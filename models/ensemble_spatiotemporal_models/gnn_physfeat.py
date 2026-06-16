
from __future__ import annotations

import time

import torch
import torch.nn as nn

from .graph_utils import knn_graph
from .backbones import build_backbone
from .gnn_base import FourierPosEnc
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


@torch.no_grad()
def _point_surface_distance_and_direction(
    pos: torch.Tensor,
    airfoil_idx: torch.Tensor,
    chunk: int = 4096,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dist (N,1), unit direction from pos toward nearest airfoil point (N,3)."""
    N = pos.size(0)
    idx = _sanitize_airfoil_idx(airfoil_idx, N, pos.device)
    if idx.numel() == 0:
        zeros_dist = torch.zeros(N, 1, device=pos.device, dtype=pos.dtype)
        zeros_dir = torch.zeros(N, 3, device=pos.device, dtype=pos.dtype)
        return zeros_dist, zeros_dir
    af = pos.index_select(0, idx)
    dist_parts: list[torch.Tensor] = []
    nn_parts: list[torch.Tensor] = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d = torch.cdist(pos[s:e], af)
        mn, j = d.min(dim=1)
        dist_parts.append(mn)
        nn_parts.append(j)
    dist = torch.cat(dist_parts, dim=0).unsqueeze(-1)
    j_all = torch.cat(nn_parts, dim=0)
    nearest = af[j_all]
    vec = nearest - pos
    unit = vec / (dist.clamp(min=eps))
    unit = torch.where(dist < eps, torch.zeros_like(unit), unit)
    return dist, unit


class SpatioTemporalGNNPhysFeat(nn.Module):
    """SpatioTemporalGNN + continuous surface proximity (distance + direction)."""

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
                "use_hierarchical is no longer supported: full 100k resolution only."
            )

        if use_fourier:
            self.pos_enc = FourierPosEnc(3, num_fourier)
            pos_dim = self.pos_enc.out_dim
        else:
            self.pos_enc = None
            pos_dim = 3

        geom_extra = 4
        in_dim = pos_dim + t_in * 3 + 1 + geom_extra

        self.node_enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.spatial = build_backbone(
            backbone,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
        self.temporal = TemporalAttentionHead(
            hidden_dim,
            t_out=t_out,
            temporal_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def _forward_single(
        self,
        pos: torch.Tensor,
        vel_in: torch.Tensor,
        airfoil_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        N = pos.size(0)
        device = pos.device
        timings: dict[str, float] = {}
        t_prev = _timer_now(device) if self.enable_timing else 0.0
        airfoil_idx = _sanitize_airfoil_idx(airfoil_idx, N, device)

        mask = pos.new_zeros(N, 1)
        mask[airfoil_idx.long()] = 1.0

        dist, dir_u = _point_surface_distance_and_direction(pos, airfoil_idx)

        pos_s = pos
        vel_s = vel_in
        mask_s = mask
        geom_s = torch.cat([dist, dir_u], dim=-1)

        if self.enable_timing:
            t_now = _timer_now(device)
            timings["sample"] = t_now - t_prev
            t_prev = t_now

        neighbors, rel_pos, dists = knn_graph(pos_s, self.k)
        if self.enable_timing:
            t_now = _timer_now(device)
            timings["graph"] = t_now - t_prev
            t_prev = t_now

        pos_feat = self.pos_enc(pos_s) if self.pos_enc else pos_s
        vel_flat = vel_s.permute(1, 0, 2).reshape(pos_s.size(0), -1)
        x = torch.cat([pos_feat, vel_flat, mask_s, geom_s], dim=-1)
        x = self.node_enc(x)
        if self.enable_timing:
            t_now = _timer_now(device)
            timings["encode"] = t_now - t_prev
            t_prev = t_now

        x = self.spatial(x, neighbors, rel_pos, dists)
        if self.enable_timing:
            t_now = _timer_now(device)
            timings["spatial"] = t_now - t_prev
            t_prev = t_now

        x = self.temporal(x)
        if self.enable_timing:
            t_now = _timer_now(device)
            timings["temporal"] = t_now - t_prev
            t_prev = t_now

        x = self.decoder(x)
        last_vel = vel_s[-1]
        x = x + last_vel.unsqueeze(1)
        x_full = x.permute(1, 0, 2)

        if self.enable_timing:
            t_now = _timer_now(device)
            timings["decode"] = t_now - t_prev
            t_prev = t_now

        x_full[:, airfoil_idx.long()] = 0.0
        if self.enable_timing:
            t_now = _timer_now(device)
            timings["post"] = t_now - t_prev
            timings["forward_total"] = sum(timings.values())

        return x_full, timings

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        B = velocity_in.size(0)
        outputs = []
        totals: dict[str, float] = {}
        for b in range(B):
            n_pts = pos[b].shape[0]
            out, timings = self._forward_single(
                pos[b],
                velocity_in[b],
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

        return torch.stack(outputs)
