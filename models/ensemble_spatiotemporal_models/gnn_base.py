"""
Complete Spatiotemporal GNN for 3-D velocity field prediction.

Pipeline (per sample):
  1.  Use the full 100k-point field directly
  2.  Build k-NN graph on all positions
  3.  Encode per-node features  [pos | vel_flat | airfoil_mask]
  4.  Spatial GNN backbone  (GAT / MeshGraphNet / GraphTransformer)
  5.  Temporal attention head  →  (N, T_out, D)
  6.  Decode to 3-D velocity delta
  7.  Residual: add last input velocity
  8.  Enforce no-slip  (zero velocity on airfoil surface)
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from .graph_utils import knn_graph
from .backbones import build_backbone
from .temporal import TemporalAttentionHead


def _sanitize_airfoil_idx(idx: torch.Tensor, num_points: int, device: torch.device) -> torch.Tensor:
    if idx.numel() == 0:
        return idx.to(device=device, dtype=torch.long)
    return idx.to(device=device, dtype=torch.long).clamp_(0, num_points - 1)


def _timer_now(device: torch.device | str) -> float:
    if isinstance(device, str):
        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    else:
        use_cuda = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize(device)
    return time.perf_counter()


# -----------------------------------------------------------------------
# Fourier positional encoding (optional, helps high-freq prediction)
# -----------------------------------------------------------------------

class FourierPosEnc(nn.Module):
    def __init__(self, in_dim: int = 3, num_freqs: int = 8):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs)
        self.out_dim = in_dim + in_dim * num_freqs * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x.unsqueeze(-1) * self.freqs          # (..., D, F)
        return torch.cat([x, proj.sin().flatten(-2),
                          proj.cos().flatten(-2)], dim=-1)


# -----------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------

class SpatioTemporalGNN(nn.Module):
    """Configurable spatiotemporal GNN for velocity-field forecasting.

    Args:
        backbone:       "gat" | "meshgraphnet" | "graph_transformer"
        hidden_dim:     width of all hidden layers
        num_layers:     number of spatial GNN layers (single-scale)
        heads:          attention heads (GAT / GraphTransformer)
        k:              k for the k-NN graph
        num_sub:        deprecated; full-resolution mode always uses all points
        t_in / t_out:   input / output time steps
        use_fourier:    apply Fourier positional encoding to coordinates
        num_fourier:    number of Fourier frequency bands
        use_hierarchical: deprecated; unsupported in full-resolution mode
        dropout:        dropout rate
        interp_k:       deprecated; interpolation is disabled in full-resolution mode
    """

    def __init__(
        self,
        backbone: str = "meshgraphnet",
        hidden_dim: int = 128,
        num_layers: int = 8,
        heads: int = 4,
        k: int = 16,
        num_sub: int = 8192,
        t_in: int = 5,
        t_out: int = 5,
        use_fourier: bool = True,
        num_fourier: int = 8,
        use_hierarchical: bool = False,
        dropout: float = 0.1,
        interp_k: int = 3,
    ):
        super().__init__()
        self.k = k
        self.num_sub = num_sub
        self.t_in = t_in
        self.t_out = t_out
        self.interp_k = interp_k
        self.use_hierarchical = use_hierarchical
        self.enable_timing = False
        self._last_timers: dict[str, float] = {}

        if use_hierarchical:
            raise ValueError(
                "use_hierarchical is no longer supported: the model now runs at full 100k resolution only."
            )

        # --- positional encoding ---
        if use_fourier:
            self.pos_enc = FourierPosEnc(3, num_fourier)
            pos_dim = self.pos_enc.out_dim
        else:
            self.pos_enc = None
            pos_dim = 3

        # input = pos_encoding + flattened velocity + airfoil mask
        in_dim = pos_dim + t_in * 3 + 1

        # --- node encoder ---
        self.node_enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # --- spatial backbone ---
        bb_kwargs = dict(hidden_dim=hidden_dim, num_layers=num_layers,
                         heads=heads, dropout=dropout)
        self.spatial = build_backbone(backbone, **bb_kwargs)

        # --- temporal head ---
        self.temporal = TemporalAttentionHead(
            hidden_dim, t_out=t_out, temporal_dim=hidden_dim,
            num_heads=heads, dropout=dropout,
        )

        # --- decoder: hidden → 3-D velocity delta per timestep ---
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    # ------------------------------------------------------------------
    # Per-sample forward
    # ------------------------------------------------------------------

    def _forward_single(
        self,
        pos: torch.Tensor,          # (N, 3)
        vel_in: torch.Tensor,       # (T_in, N, 3)
        airfoil_idx: torch.Tensor,  # (M,)
    ) -> tuple[torch.Tensor, dict[str, float]]:              # (T_out, N, 3)

        N = pos.size(0)
        device = pos.device
        timings: dict[str, float] = {}
        t_prev = _timer_now(device) if self.enable_timing else 0.0
        airfoil_idx = _sanitize_airfoil_idx(airfoil_idx, N, device)

        # 1. airfoil mask
        mask = pos.new_zeros(N, 1)
        mask[airfoil_idx] = 1.0

        # 2. Full-resolution path: keep all 100k points.
        pos_s = pos
        vel_s = vel_in
        mask_s = mask
        if self.enable_timing:
            t_now = _timer_now(device)
            timings["sample"] = t_now - t_prev
            t_prev = t_now

        # 3. k-NN graph on full-resolution points
        neighbors, rel_pos, dists = knn_graph(pos_s, self.k)
        if self.enable_timing:
            t_now = _timer_now(device)
            timings["graph"] = t_now - t_prev
            t_prev = t_now

        # 4. build node features
        pos_feat = self.pos_enc(pos_s) if self.pos_enc else pos_s
        vel_flat = vel_s.permute(1, 0, 2).reshape(pos_s.size(0), -1)  # (Ns, T*3)
        x = torch.cat([pos_feat, vel_flat, mask_s], dim=-1)
        x = self.node_enc(x)                                          # (Ns, D)
        if self.enable_timing:
            t_now = _timer_now(device)
            timings["encode"] = t_now - t_prev
            t_prev = t_now

        # 5. spatial backbone
        x = self.spatial(x, neighbors, rel_pos, dists)                 # (N, D)
        if self.enable_timing:
            t_now = _timer_now(device)
            timings["spatial"] = t_now - t_prev
            t_prev = t_now

        # 6. temporal head → (Ns, T_out, D)
        x = self.temporal(x)
        if self.enable_timing:
            t_now = _timer_now(device)
            timings["temporal"] = t_now - t_prev
            t_prev = t_now

        # 7. decode → delta velocities (Ns, T_out, 3)
        x = self.decoder(x)

        # 8. residual prediction: add last observed velocity
        last_vel = vel_s[-1]  # (Ns, 3)
        x = x + last_vel.unsqueeze(1)                                 # (Ns, T_out, 3)
        if self.enable_timing:
            t_now = _timer_now(device)
            timings["decode"] = t_now - t_prev
            t_prev = t_now

        # 9. Already at full resolution  →  (T_out, N, 3)
        x_full = x.permute(1, 0, 2)
        if self.enable_timing:
            t_now = _timer_now(device)
            timings["interp"] = t_now - t_prev
            t_prev = t_now

        # 10. enforce no-slip boundary on airfoil surface
        x_full[:, airfoil_idx] = 0.0
        if self.enable_timing:
            t_now = _timer_now(device)
            timings["post"] = t_now - t_prev
            timings["forward_total"] = sum(timings.values())

        return x_full, timings

    # ------------------------------------------------------------------
    # Batched forward (competition-compatible signature)
    # ------------------------------------------------------------------

    def forward(
        self,
        t: torch.Tensor,                       # (B, 10)
        pos: torch.Tensor,                      # (B, N, 3)
        idcs_airfoil: list[torch.Tensor],       # list of B tensors
        velocity_in: torch.Tensor,              # (B, T_in, N, 3)
    ) -> torch.Tensor:                          # (B, T_out, N, 3)

        B = velocity_in.size(0)
        outputs = []
        totals: dict[str, float] = {}
        for b in range(B):
            n_pts = pos[b].shape[0]
            out, timings = self._forward_single(
                pos[b], velocity_in[b],
                _sanitize_airfoil_idx(idcs_airfoil[b], n_pts, pos.device),
            )
            outputs.append(out)
            if self.enable_timing:
                for k, v in timings.items():
                    totals[k] = totals.get(k, 0.0) + float(v)

        if self.enable_timing:
            self._last_timers = {k: v / max(B, 1) for k, v in totals.items()}
        else:
            self._last_timers = {}

        return torch.stack(outputs)  # (B, T_out, N, 3)
