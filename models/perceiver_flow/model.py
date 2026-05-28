import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


# ── Position Encoding ──────────────────────────────────────────────────────────

class FourierPositionEncoding(nn.Module):
    def __init__(self, num_frequencies=8):
        super().__init__()
        self.L = num_frequencies
        freqs = 2.0 ** torch.arange(num_frequencies).float()
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self):
        return 3 + 3 * 2 * self.L

    def forward(self, pos):
        B, N, _ = pos.shape
        angles = pos.unsqueeze(-1) * self.freqs * math.pi
        sins   = torch.sin(angles)
        coss   = torch.cos(angles)
        enc    = torch.stack([sins, coss], dim=-1).flatten(2)
        return torch.cat([pos, enc], dim=-1)


# ── Cross Attention ────────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out     = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

    def forward(self, q, kv):
        B, Nq, _ = q.shape
        B, Nk, _ = kv.shape
        H, Dh    = self.n_heads, self.d_head
        q  = self.norm_q(q)
        kv = self.norm_kv(kv)
        Q  = self.q_proj(q).view(B, Nq, H, Dh).transpose(1, 2)
        K  = self.k_proj(kv).view(B, Nk, H, Dh).transpose(1, 2)
        V  = self.v_proj(kv).view(B, Nk, H, Dh).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out  = torch.matmul(attn, V)
        out  = out.transpose(1, 2).reshape(B, Nq, H * Dh)
        return self.out(out)


# ── Perceiver Encoder ──────────────────────────────────────────────────────────

class PerceiverEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.ff1   = nn.Linear(d_model, d_ff)
        self.ff2   = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()

    def forward(self, latent, point_cloud):
        latent     = latent + self.cross_attn(latent, point_cloud)
        x, _       = self.self_attn(self.norm2(latent),
                                    self.norm2(latent),
                                    self.norm2(latent))
        latent     = latent + x
        latent     = latent + self.drop(
            self.ff2(self.act(self.ff1(self.norm3(latent))))
        )
        return latent


class PerceiverEncoder(nn.Module):
    def __init__(self, in_features, n_latent=256, d_model=128,
                 n_heads=4, n_blocks=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.latent     = nn.Parameter(torch.randn(1, n_latent, d_model) * 0.02)
        self.blocks     = nn.ModuleList([
            PerceiverEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_blocks)
        ])
        self.norm_out   = nn.LayerNorm(d_model)

    def forward(self, point_features):
        B      = point_features.shape[0]
        kv     = self.input_proj(point_features)
        latent = self.latent.expand(B, -1, -1)
        for block in self.blocks:
            latent = block(latent, kv)
        return self.norm_out(latent)


# ── Perceiver Decoder ──────────────────────────────────────────────────────────

class PerceiverDecoder(nn.Module):
    def __init__(self, pos_features, n_latent=256, d_model=128,
                 n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.query_proj  = nn.Linear(pos_features, d_model)
        self.cross_attn  = CrossAttention(d_model, n_heads, dropout)
        self.ff          = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.out_proj    = nn.Linear(d_model, 3)
        self.norm_latent = nn.LayerNorm(d_model)

    def forward(self, latent, pos_enc):
        queries = self.query_proj(pos_enc)
        out     = queries + self.cross_attn(queries, self.norm_latent(latent))
        out     = out + self.ff(out)
        return self.out_proj(out)


# ── Geometry Encoder ───────────────────────────────────────────────────────────

class GeometryEncoder(nn.Module):
    def __init__(self, d_geom=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),  nn.LayerNorm(64),  nn.GELU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, d_geom),
        )
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(d_geom),
            nn.Linear(d_geom, d_geom),
            nn.GELU(),
        )

    def forward(self, surface_pos):
        x = self.mlp(surface_pos)
        x = x.max(dim=1).values
        return self.pool_proj(x)


# ── Temporal Transformer ───────────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    def __init__(self, n_latent=256, d_model=128, d_geom=64,
                 T_in=5, T_out=5, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.n_latent  = n_latent
        self.d_model   = d_model
        self.T_in      = T_in
        self.T_out     = T_out
        self.token_dim = n_latent * d_model
        self.geom_proj = nn.Linear(d_geom, self.token_dim)
        self.pos_emb   = nn.Parameter(
            torch.randn(1, T_in + 1, self.token_dim) * 0.02)
        self.t_dim     = 256
        self.in_proj   = nn.Linear(self.token_dim, self.t_dim)
        self.out_proj  = nn.Linear(self.t_dim, self.token_dim * T_out)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoder_layer    = nn.TransformerEncoderLayer(
                d_model=self.t_dim, nhead=n_heads,
                dim_feedforward=self.t_dim * 4,
                dropout=dropout, batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers)

    def forward(self, latents, geom_emb):
        B      = latents.shape[0]
        tokens = latents.reshape(B, self.T_in, self.n_latent * self.d_model)
        geom_token = self.geom_proj(geom_emb).unsqueeze(1)
        tokens = torch.cat([geom_token, tokens], dim=1)
        tokens = tokens + self.pos_emb
        tokens = self.in_proj(tokens)
        tokens = self.transformer(tokens)
        summary = tokens[:, 0, :]
        future  = self.out_proj(summary)
        return future.reshape(B, self.T_out, self.n_latent, self.d_model)


# ── Main Model ─────────────────────────────────────────────────────────────────

class PerceiverFlow(nn.Module):
    """
    Geometry-conditioned spatiotemporal flow predictor.
    Predicts velocity_out (5 future snapshots) from velocity_in (5 past snapshots).

    Can be instantiated without arguments — loads weights from weights.pt
    in the same directory as this file.
    """

    # Fixed config matching training
    CFG = {
        "n_fourier"  : 8,
        "d_geom"     : 64,
        "n_latent"   : 256,
        "d_model"    : 128,
        "enc_heads"  : 4,
        "enc_blocks" : 2,
        "temp_heads" : 4,
        "temp_layers": 3,
        "dec_heads"  : 4,
    }

    def __init__(self):
        super().__init__()
        cfg = self.CFG

        self.pos_enc = FourierPositionEncoding(cfg["n_fourier"])
        pos_dim      = self.pos_enc.out_dim
        in_features  = 3 + pos_dim + 1

        self.geom_encoder    = GeometryEncoder(d_geom=cfg["d_geom"])
        self.spatial_encoder = PerceiverEncoder(
            in_features=in_features,
            n_latent=cfg["n_latent"],
            d_model=cfg["d_model"],
            n_heads=cfg["enc_heads"],
            n_blocks=cfg["enc_blocks"],
            d_ff=cfg["d_model"] * 2,
        )
        self.temporal        = TemporalTransformer(
            n_latent=cfg["n_latent"],
            d_model=cfg["d_model"],
            d_geom=cfg["d_geom"],
            T_in=5, T_out=5,
            n_heads=cfg["temp_heads"],
            n_layers=cfg["temp_layers"],
        )
        self.spatial_decoder = PerceiverDecoder(
            pos_features=pos_dim,
            n_latent=cfg["n_latent"],
            d_model=cfg["d_model"],
            n_heads=cfg["dec_heads"],
            d_ff=cfg["d_model"] * 2,
        )

        # ── Loading weights ─────────────────────────────
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(
            repo_id   = "hd-hg/perceiver-flow-weights",
            filename  = "weights.pt",
            repo_type = "model",
        )
        state = torch.load(weights_path, map_location="cpu")
        self.load_state_dict(state)
        print("PerceiverFlow: weights loaded from HuggingFace")

    def _normalize(self, pos, velocity_in, idcs_airfoil):
        """
        Apply same normalization as training Dataset.
        pos:          (B, N, 3)
        velocity_in:  (B, 5, N, 3)
        idcs_airfoil: list of B tensors
        Returns normalized pos, vel_in, freestream_mags
        """
        B, N, _ = pos.shape

        # Position normalization per sample
        pos_norm = torch.zeros_like(pos)
        for b in range(B):
            p      = pos[b]                                    # (N, 3)
            p_min  = p.min(0).values
            p_max  = p.max(0).values
            pos_norm[b] = 2.0 * (p - p_min) / (p_max - p_min + 1e-8) - 1.0

        # Freestream normalization per sample
        freestream_mags = []
        vel_in_norm     = torch.zeros_like(velocity_in)
        for b in range(B):
            idcs      = idcs_airfoil[b]
            bulk_mask = torch.ones(N, dtype=torch.bool, device=pos.device)
            bulk_mask[idcs] = False
            last_snap = velocity_in[b, -1]                    # (N, 3)
            bulk_vel  = last_snap[bulk_mask]                   # (N_bulk, 3)
            fs_mag    = bulk_vel.norm(dim=-1).mean().clamp(min=1e-6)
            freestream_mags.append(fs_mag)
            vel_in_norm[b] = velocity_in[b] / fs_mag

        return pos_norm, vel_in_norm, freestream_mags

    def forward(
        self,
        t            : torch.Tensor,        # (B, 10)
        pos          : torch.Tensor,        # (B, N, 3)
        idcs_airfoil : list,                # list of B variable-length tensors
        velocity_in  : torch.Tensor,        # (B, 5, N, 3)
    ) -> torch.Tensor:                      # (B, 5, N, 3)

        B, N, _ = pos.shape

        # ── Normalize inputs ───────────────────────────────────────────────
        pos_norm, vel_in_norm, freestream_mags = self._normalize(
            pos, velocity_in, idcs_airfoil)

        # ── Position encoding ──────────────────────────────────────────────
        pos_enc = self.pos_enc(pos_norm)                       # (B, N, pos_dim)

        # ── Geometry embedding ─────────────────────────────────────────────
        geom_list = []
        for b in range(B):
            idcs = idcs_airfoil[b]
            surf = pos_norm[b][idcs].unsqueeze(0)              # (1, n_surf, 3)
            geom_list.append(self.geom_encoder(surf))
        geom_emb = torch.cat(geom_list, dim=0)                 # (B, d_geom)

        # ── Spatial encoding over 5 input snapshots ────────────────────────
        is_af = torch.zeros(B, N, 1, device=pos.device)
        for b in range(B):
            is_af[b, idcs_airfoil[b], 0] = 1.0

        latent_list = []
        for t_idx in range(5):
            vel_t = vel_in_norm[:, t_idx, :, :]               # (B, N, 3)
            feats = torch.cat([vel_t, pos_enc, is_af], dim=-1)
            latent_list.append(self.spatial_encoder(feats))
        latents = torch.stack(latent_list, dim=1)              # (B, 5, n_lat, d)

        # ── Temporal prediction ────────────────────────────────────────────
        delta_z  = self.temporal(latents, geom_emb)
        z_future = latents[:, -1:, :, :] + delta_z            # (B, 5, n_lat, d)

        # ── Spatial decoding ───────────────────────────────────────────────
        out_list = []
        for t_idx in range(5):
            z_t = z_future[:, t_idx, :, :]
            out_list.append(self.spatial_decoder(z_t, pos_enc))
        vel_pred = torch.stack(out_list, dim=1)                # (B, 5, N, 3)

        # ── Hard no-slip enforcement ───────────────────────────────────────
        for b in range(B):
            vel_pred[b, :, idcs_airfoil[b], :] = 0.0

        # ── Denormalize back to original units ─────────────────────────────
        for b in range(B):
            vel_pred[b] = vel_pred[b] * freestream_mags[b]

        return vel_pred