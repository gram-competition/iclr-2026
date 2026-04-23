import math
import os

import torch
import torch.nn.functional as F
from torch.nn import Embedding, LayerNorm, Linear, Module, ModuleList, ReLU, Sequential
from torch_cluster import knn


class ABUPT(Module):
    """Anchored-Branched Universal Physics Transformer (small, pragmatic variant).

    Stages:
      1. Per-point embedding: pos + fourier(pos) + velocity_in + t_start + is_surface.
      2. Random supernode sampling: N_s from airfoil surface, N_v from volume (per batch element).
      3. Points -> supernodes message passing via k-NN on position; mean-pooled messages form latents.
      4. Approximator: branched transformer (separate surface/volume token streams,
         shared self-attn and FFN, cross-branch attention every other block).
      5. Perceiver decoder: all N point queries cross-attend to M = N_s + N_v supernode latents.
      6. Head: per-point 5-step velocity deltas, added to last input frame, hard-masked at surface.
    """

    FREQS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)

    def __init__(
        self,
        hidden: int = 192,
        num_surface_supernodes: int = 128,
        num_wake_supernodes: int = 384,
        num_far_supernodes: int = 128,
        wake_pool_frac: float = 0.2,
        num_approx_blocks: int = 12,
        num_heads: int = 4,
        encoder_k: int = 16,
        ffn_mult: int = 2,
        cross_branch_every: int = 2,
        num_decoder_blocks: int = 2,
    ):
        super().__init__()
        assert hidden % num_heads == 0

        self.hidden = hidden
        self.num_surface_supernodes = num_surface_supernodes
        self.num_wake_supernodes = num_wake_supernodes
        self.num_far_supernodes = num_far_supernodes
        self.wake_pool_frac = wake_pool_frac
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.encoder_k = encoder_k

        self.register_buffer(
            "freqs",
            2.0 * math.pi * torch.tensor(self.FREQS, dtype=torch.float32),
        )
        # Normalization buffers (identity defaults; overridden by norm_stats.pt if present,
        # or by state_dict.pt, which stores the buffers used at training time).
        self.register_buffer("pos_mean", torch.zeros(3))
        self.register_buffer("pos_scale", torch.ones(3))
        self.register_buffer("vel_mean", torch.zeros(3))
        self.register_buffer("vel_std", torch.ones(3))

        # pos (3) + fourier (3*2*F=36) + velocity_in (15) + t_start (1) + is_surface (1)
        in_dim = 3 + 3 * 2 * len(self.FREQS) + 15 + 1 + 1

        self.point_embed = Sequential(
            Linear(in_dim, hidden),
            LayerNorm(hidden),
            ReLU(),
            Linear(hidden, hidden),
            LayerNorm(hidden),
            ReLU(),
        )

        self.mp_rel_pos_embed = Sequential(
            Linear(3, hidden),
            ReLU(),
            Linear(hidden, hidden),
        )
        self.mp_message = Sequential(
            Linear(2 * hidden, hidden),
            LayerNorm(hidden),
            ReLU(),
            Linear(hidden, hidden),
        )
        # 0 = far, 1 = wake, 2 = surface
        self.type_embed = Embedding(3, hidden)
        self.supernode_ln = LayerNorm(hidden)

        self.blocks = ModuleList([
            BranchedBlock(
                hidden=hidden,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
                cross=((i + 1) % cross_branch_every == 0),
            )
            for i in range(num_approx_blocks)
        ])

        self.decoder = PerceiverDecoder(
            hidden=hidden, num_heads=num_heads, ffn_mult=ffn_mult,
            num_blocks=num_decoder_blocks,
        )

        self.head = Sequential(
            Linear(hidden, hidden),
            LayerNorm(hidden),
            ReLU(),
            Linear(hidden, 15),
        )

        stats_path = os.path.join(os.path.dirname(__file__), "norm_stats.pt")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            for key, val in stats.items():
                getattr(self, key).copy_(val)
            print(
                f"[ABUPT] loaded norm_stats.pt: "
                f"vel_mean={self.vel_mean.tolist()}, vel_std={self.vel_std.tolist()}"
            )
        else:
            print(f"[ABUPT] norm_stats.pt not found — using identity normalization")

        path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

        if torch.cuda.is_available():
            self.to("cuda")

        # Ship in eval mode; train.py re-enters training via model.train().
        self.eval()

    def _fourier(self, pos: torch.Tensor) -> torch.Tensor:
        angles = pos.unsqueeze(-1) * self.freqs
        feats = torch.stack([angles.sin(), angles.cos()], dim=-1)
        return feats.flatten(start_dim=2)

    def _sample_supernodes(
        self,
        pos: torch.Tensor,
        velocity_in: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per batch element: N_s surface + N_w wake (top-variance) + N_f far-field indices."""
        B, N, _ = pos.shape
        device = pos.device
        N_s = self.num_surface_supernodes
        N_w = self.num_wake_supernodes
        N_f = self.num_far_supernodes

        # Per-point temporal variance of the input velocity, summed over components.
        # Zero on the wall (v=0 for all t) and in the steady far-field; peaks in the wake.
        vel_var = velocity_in.float().var(dim=1).sum(dim=-1)  # (B, N)

        surf_idx = torch.empty((B, N_s), dtype=torch.long, device=device)
        wake_idx = torch.empty((B, N_w), dtype=torch.long, device=device)
        far_idx = torch.empty((B, N_f), dtype=torch.long, device=device)

        for i, airfoil_idcs in enumerate(idcs_airfoil):
            n_surface = airfoil_idcs.numel()
            if n_surface >= N_s:
                perm = torch.randperm(n_surface, device=device)[:N_s]
            else:
                perm = torch.randint(n_surface, (N_s,), device=device)
            surf_idx[i] = airfoil_idcs[perm]

            mask = torch.ones(N, dtype=torch.bool, device=device)
            mask[airfoil_idcs] = False
            vol_pool = mask.nonzero(as_tuple=False).squeeze(-1)
            n_vol = vol_pool.numel()

            scores = vel_var[i, vol_pool]
            n_wake_pool = max(N_w, int(self.wake_pool_frac * n_vol))
            n_wake_pool = min(n_wake_pool, n_vol)
            _, top = scores.topk(n_wake_pool)
            wake_pool = vol_pool[top]

            far_mask = torch.ones(n_vol, dtype=torch.bool, device=device)
            far_mask[top] = False
            far_pool = vol_pool[far_mask.nonzero(as_tuple=False).squeeze(-1)]
            n_far_pool = far_pool.numel()

            if n_wake_pool >= N_w:
                perm = torch.randperm(n_wake_pool, device=device)[:N_w]
            else:
                perm = torch.randint(n_wake_pool, (N_w,), device=device)
            wake_idx[i] = wake_pool[perm]

            if n_far_pool >= N_f:
                perm = torch.randperm(n_far_pool, device=device)[:N_f]
            else:
                perm = torch.randint(n_far_pool, (N_f,), device=device)
            far_idx[i] = far_pool[perm]

        return surf_idx, wake_idx, far_idx

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        # Trained under bf16 autocast; match that at inference to save memory
        # and reproduce eval numerics. inference_mode disables autograd bookkeeping
        # so activations aren't retained when the caller forgets torch.no_grad().
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
        B, T_in, N, _ = velocity_in.shape
        device = pos.device

        pos = (pos - self.pos_mean) / self.pos_scale
        velocity_in = (velocity_in - self.vel_mean) / self.vel_std

        vel_flat = velocity_in.transpose(1, 2).reshape(B, N, T_in * 3)
        t_start = t[:, 0:1].unsqueeze(1).expand(-1, N, -1)
        pos_fourier = self._fourier(pos)

        is_surface = torch.zeros((B, N, 1), device=device, dtype=pos.dtype)
        for i, idcs in enumerate(idcs_airfoil):
            is_surface[i, idcs, 0] = 1.0

        x = torch.cat([pos, pos_fourier, vel_flat, t_start, is_surface], dim=2)
        point_feat = self.point_embed(x)

        surf_idx, wake_idx, far_idx = self._sample_supernodes(pos, velocity_in, idcs_airfoil)
        N_s = self.num_surface_supernodes
        N_w = self.num_wake_supernodes
        N_f = self.num_far_supernodes
        N_v = N_w + N_f
        M = N_s + N_v

        batch_range = torch.arange(B, device=device).unsqueeze(-1)
        surf_pos = pos[batch_range, surf_idx]
        wake_pos = pos[batch_range, wake_idx]
        far_pos = pos[batch_range, far_idx]
        super_pos = torch.cat([surf_pos, wake_pos, far_pos], dim=1)

        super_pos_flat = super_pos.reshape(B * M, 3)
        pos_flat = pos.reshape(B * N, 3)
        point_feat_flat = point_feat.reshape(B * N, -1)
        batch_super = torch.arange(B, device=device).repeat_interleave(M)
        batch_pts = torch.arange(B, device=device).repeat_interleave(N)

        # Volume-only candidate pool for kNN. Both surface and volume supernodes
        # pool from this to avoid diluting the pooled message with v=0 neighbors
        # at the wall. Wall location is already conveyed by is_surface (per-point
        # input feature) and by the surface branch via cross-branch attention.
        vol_pool_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        for i, idcs in enumerate(idcs_airfoil):
            vol_pool_mask[i, idcs] = False
        vol_pool_global = vol_pool_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        vol_pool_pos = pos_flat[vol_pool_global]
        batch_vol_pool = batch_pts[vol_pool_global]

        edge = knn(
            vol_pool_pos, super_pos_flat, self.encoder_k,
            batch_x=batch_vol_pool, batch_y=batch_super,
        )
        neigh_local = edge[1].view(B * M, self.encoder_k)
        neigh_idx = vol_pool_global[neigh_local]  # remap to B*N indexing

        neigh_feat = point_feat_flat[neigh_idx]
        neigh_pos = pos_flat[neigh_idx]
        rel_pos = neigh_pos - super_pos_flat.unsqueeze(1)
        rel_pos_feat = self.mp_rel_pos_embed(rel_pos)

        messages = self.mp_message(torch.cat([neigh_feat, rel_pos_feat], dim=-1))
        super_feat = messages.mean(dim=1).view(B, M, -1)

        type_ids = torch.cat([
            torch.full((N_s,), 2, device=device, dtype=torch.long),
            torch.full((N_w,), 1, device=device, dtype=torch.long),
            torch.full((N_f,), 0, device=device, dtype=torch.long),
        ])
        super_feat = super_feat + self.type_embed(type_ids).unsqueeze(0)
        super_feat = self.supernode_ln(super_feat)

        surf_tok = super_feat[:, :N_s]
        vol_tok = super_feat[:, N_s:]

        for block in self.blocks:
            surf_tok, vol_tok = block(surf_tok, vol_tok)

        latents = torch.cat([surf_tok, vol_tok], dim=1)

        decoded = self.decoder(point_feat, latents)

        delta = self.head(decoded).view(B, N, T_in, 3)
        last_frame = velocity_in[:, -1, :, :]
        out = last_frame.unsqueeze(2) + delta

        out = out * self.vel_std + self.vel_mean

        for i, idcs in enumerate(idcs_airfoil):
            out[i, idcs] = 0.0

        return out.transpose(1, 2)


class BranchedBlock(Module):
    """Surface/volume streams share self-attn + FFN weights; cross-branch attn every `cross` block."""

    def __init__(self, hidden: int, num_heads: int, ffn_mult: int, cross: bool):
        super().__init__()
        assert hidden % num_heads == 0
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.cross = cross

        self.ln_self = LayerNorm(hidden)
        self.qkv = Linear(hidden, 3 * hidden)
        self.out_proj = Linear(hidden, hidden)

        if cross:
            self.ln_cross_q = LayerNorm(hidden)
            self.ln_cross_kv = LayerNorm(hidden)
            self.cross_q = Linear(hidden, hidden)
            self.cross_kv = Linear(hidden, 2 * hidden)
            self.cross_out = Linear(hidden, hidden)

        self.ln_ffn = LayerNorm(hidden)
        self.ffn = Sequential(
            Linear(hidden, ffn_mult * hidden),
            ReLU(),
            Linear(ffn_mult * hidden, hidden),
        )

    def _self_attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        qkv = self.qkv(self.ln_self(x)).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, H)
        return self.out_proj(out)

    def _cross_attn(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        B, Tq, H = q_in.shape
        Tk = kv_in.shape[1]
        q = self.cross_q(self.ln_cross_q(q_in))
        q = q.view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.cross_kv(self.ln_cross_kv(kv_in))
        kv = kv.view(B, Tk, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, Tq, H)
        return self.cross_out(out)

    def forward(self, surf: torch.Tensor, vol: torch.Tensor):
        surf = surf + self._self_attn(surf)
        vol = vol + self._self_attn(vol)

        if self.cross:
            surf_new = surf + self._cross_attn(surf, vol)
            vol_new = vol + self._cross_attn(vol, surf)
            surf, vol = surf_new, vol_new

        surf = surf + self.ffn(self.ln_ffn(surf))
        vol = vol + self.ffn(self.ln_ffn(vol))
        return surf, vol


class DecoderBlock(Module):
    """One Perceiver layer: queries cross-attend to latents, then FFN."""

    def __init__(self, hidden: int, num_heads: int, ffn_mult: int):
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads

        self.ln_q = LayerNorm(hidden)
        self.ln_kv = LayerNorm(hidden)
        self.q_proj = Linear(hidden, hidden)
        self.kv_proj = Linear(hidden, 2 * hidden)
        self.out_proj = Linear(hidden, hidden)

        self.ln_ffn = LayerNorm(hidden)
        self.ffn = Sequential(
            Linear(hidden, ffn_mult * hidden),
            ReLU(),
            Linear(ffn_mult * hidden, hidden),
        )

    def forward(self, queries: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        B, N, H = queries.shape
        M = latents.shape[1]
        q = self.q_proj(self.ln_q(queries))
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(self.ln_kv(latents))
        kv = kv.view(B, M, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, H)
        out = queries + self.out_proj(out)
        return out + self.ffn(self.ln_ffn(out))


class PerceiverDecoder(Module):
    """Stack of N point-queries-cross-attend-to-M-latents blocks."""

    def __init__(self, hidden: int, num_heads: int, ffn_mult: int, num_blocks: int = 1):
        super().__init__()
        self.blocks = ModuleList([
            DecoderBlock(hidden, num_heads, ffn_mult) for _ in range(num_blocks)
        ])

    def forward(self, queries: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            queries = block(queries, latents)
        return queries
