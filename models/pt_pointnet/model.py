import math
import os

import torch
from torch.nn import LayerNorm, Linear, Module, ModuleList, ReLU, Sequential
from torch.utils.checkpoint import checkpoint
from torch_cluster import knn


class PTPointNet(Module):
    """PointNet + local Point-Transformer-style attention.

    Same inputs / encoder / global-max / decoder as KNNPointNet, but the
    per-round k-NN max-pool + MLP is replaced by a Point Transformer block:
    subtraction-form attention over k neighbors with a learned relative-position
    encoding added to both keys and values, followed by an FFN. Residual + LN
    around each sub-block.

    Per-point input: pos (3) + fourier(pos) (3 * 2 * F) + velocity_in (15) + t_start (1).
    """

    FREQS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)

    def __init__(
        self,
        num_blocks: int = 2,
        k: int = 16,
        hidden: int = 128,
        num_heads: int = 4,
        ffn_mult: int = 2,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        assert hidden % num_heads == 0

        self.num_blocks = num_blocks
        self.k = k
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.ffn_mult = ffn_mult
        self.use_checkpoint = use_checkpoint

        self.register_buffer(
            "freqs",
            2.0 * math.pi * torch.tensor(self.FREQS, dtype=torch.float32),
        )
        self.register_buffer("pos_mean", torch.zeros(3))
        self.register_buffer("pos_scale", torch.ones(3))
        self.register_buffer("vel_mean", torch.zeros(3))
        self.register_buffer("vel_std", torch.ones(3))
        in_dim = 3 + 3 * 2 * len(self.FREQS) + 15 + 1

        self.encoder = Sequential(
            Linear(in_dim, hidden),
            LayerNorm(hidden),
            ReLU(),
            Linear(hidden, hidden),
            LayerNorm(hidden),
            ReLU(),
        )

        self.blocks = ModuleList(
            [PTBlock(hidden, num_heads, ffn_mult=ffn_mult) for _ in range(num_blocks)]
        )

        self.decoder = Sequential(
            Linear(3 * hidden, 2 * hidden),
            LayerNorm(2 * hidden),
            ReLU(),
            Linear(2 * hidden, 15),
        )

        stats_path = os.path.join(os.path.dirname(__file__), "norm_stats.pt")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            for key, val in stats.items():
                getattr(self, key).copy_(val)
            print(
                f"[PTPointNet] loaded norm_stats.pt: "
                f"vel_mean={self.vel_mean.tolist()}, vel_std={self.vel_std.tolist()}"
            )
        else:
            print(f"[PTPointNet] norm_stats.pt not found — using identity normalization")

        path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

        if torch.cuda.is_available():
            self.to("cuda")

        self.eval()

    def _fourier(self, pos: torch.Tensor) -> torch.Tensor:
        angles = pos.unsqueeze(-1) * self.freqs
        feats = torch.stack([angles.sin(), angles.cos()], dim=-1)
        return feats.flatten(start_dim=2)

    def _knn_indices(self, pos_flat: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Compute k-NN once and return neighbor indices of shape (N_total, k)."""
        edge = knn(pos_flat, pos_flat, self.k, batch_x=batch, batch_y=batch)
        n_total = pos_flat.shape[0]
        return edge[1].view(n_total, self.k)

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training:
            with torch.inference_mode(), torch.autocast(device_type=pos.device.type, dtype=torch.bfloat16):
                return self._forward_impl(t, pos, idcs_airfoil, velocity_in)
        return self._forward_impl(t, pos, idcs_airfoil, velocity_in)

    def _forward_impl(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_t_in, num_pos, _ = velocity_in.shape

        pos = (pos - self.pos_mean) / self.pos_scale
        velocity_in = (velocity_in - self.vel_mean) / self.vel_std

        vel_flat = velocity_in.transpose(1, 2).reshape(batch_size, num_pos, num_t_in * 3)
        t_start = t[:, 0:1].unsqueeze(1).expand(-1, num_pos, -1)
        pos_fourier = self._fourier(pos)

        x = torch.cat([pos, pos_fourier, vel_flat, t_start], dim=2)

        point_feat = self.encoder(x)  # (B, N, H)

        feats_flat = point_feat.reshape(batch_size * num_pos, -1)
        pos_flat = pos.reshape(batch_size * num_pos, 3)
        batch_vec = torch.arange(batch_size, device=pos.device).repeat_interleave(num_pos)

        # kNN computed once, reused across blocks.
        neigh_idx = self._knn_indices(pos_flat, batch_vec)  # (N_total, k)
        # Relative positions (queries - neighbors) reused as geometric bias input.
        rel_pos = pos_flat.unsqueeze(1) - pos_flat[neigh_idx]  # (N_total, k, 3)

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                feats_flat = checkpoint(
                    block, feats_flat, neigh_idx, rel_pos, use_reentrant=False,
                )
            else:
                feats_flat = block(feats_flat, neigh_idx, rel_pos)

        neighborhood_feat = feats_flat.view(batch_size, num_pos, -1)

        global_feat = neighborhood_feat.max(dim=1).values
        global_feat = global_feat.unsqueeze(1).expand(-1, num_pos, -1)
        combined = torch.cat([point_feat, neighborhood_feat, global_feat], dim=2)

        delta = self.decoder(combined).view(batch_size, num_pos, num_t_in, 3)
        last_frame = velocity_in[:, -1, :, :]
        out = last_frame.unsqueeze(2) + delta

        out = out * self.vel_std + self.vel_mean

        for i, idcs in enumerate(idcs_airfoil):
            out[i, idcs] = 0.0

        return out.transpose(1, 2)


class PTBlock(Module):
    """Point Transformer block: subtraction attention over kNN + FFN, pre-LN, residual."""

    def __init__(self, hidden: int, num_heads: int, ffn_mult: int = 2):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads

        self.ln_attn = LayerNorm(hidden)
        self.q_proj = Linear(hidden, hidden)
        self.k_proj = Linear(hidden, hidden)
        self.v_proj = Linear(hidden, hidden)

        # Relative-position encoder δ_ij: 3 -> hidden. Added to keys and values.
        self.pos_enc = Sequential(
            Linear(3, hidden),
            ReLU(),
            Linear(hidden, hidden),
        )

        # Attention-weight MLP γ: (q - k + δ) -> per-channel weights (one per head).
        # Output dim = num_heads so weights are shared across head_dim channels.
        self.gamma = Sequential(
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, num_heads),
        )

        self.out_proj = Linear(hidden, hidden)

        self.ln_ffn = LayerNorm(hidden)
        self.ffn = Sequential(
            Linear(hidden, ffn_mult * hidden),
            ReLU(),
            Linear(ffn_mult * hidden, hidden),
        )

    def forward(
        self,
        x: torch.Tensor,        # (N_total, H)
        neigh_idx: torch.Tensor,  # (N_total, k)
        rel_pos: torch.Tensor,    # (N_total, k, 3)
    ) -> torch.Tensor:
        residual = x
        h = self.ln_attn(x)

        q = self.q_proj(h)                 # (N, H)
        k = self.k_proj(h)[neigh_idx]      # (N, k, H)
        v = self.v_proj(h)[neigh_idx]      # (N, k, H)
        delta = self.pos_enc(rel_pos)      # (N, k, H)

        # Subtraction-form pre-weights.
        pre = q.unsqueeze(1) - k + delta   # (N, k, H)
        w = self.gamma(pre)                # (N, k, num_heads)
        w = torch.softmax(w, dim=1)        # softmax over k neighbors

        # Reshape to heads and apply weights.
        n_total, k_neigh, _ = v.shape
        v_heads = (v + delta).view(n_total, k_neigh, self.num_heads, self.head_dim)
        w_heads = w.unsqueeze(-1)          # (N, k, num_heads, 1)
        out = (w_heads * v_heads).sum(dim=1)  # (N, num_heads, head_dim)
        out = out.reshape(n_total, self.hidden)
        out = self.out_proj(out)

        x = residual + out
        x = x + self.ffn(self.ln_ffn(x))
        return x
