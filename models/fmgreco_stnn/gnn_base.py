"""Spatiotemporal GNN: k-NN graph, MGN backbone, temporal head, residual + no-slip."""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn

from .backbones import MeshGraphNetBackbone
from .graph_utils import airfoil_boundary_features, knn_graph
from .temporal import TemporalAttentionHead


def _sanitize_airfoil_idx(
    idx: torch.Tensor, num_points: int, device: torch.device
) -> torch.Tensor:
    if idx.numel() == 0:
        return idx.to(device=device, dtype=torch.long)
    return idx.to(device=device, dtype=torch.long).clamp_(0, num_points - 1)


class FourierPosEnc(nn.Module):
    def __init__(self, in_dim: int = 3, num_freqs: int = 8):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs)
        self.out_dim = in_dim + in_dim * num_freqs * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x.unsqueeze(-1) * self.freqs
        return torch.cat(
            [x, proj.sin().flatten(-2), proj.cos().flatten(-2)], dim=-1
        )


class DeltaDecoder(nn.Module):
    """Maps (N, t_out, td) -> (N, t_out, 3) optional SIREN-style sin stack on the delta."""

    def __init__(
        self,
        td: int,
        t_out: int,
        hidden_dim: int,
        *,
        use_siren: bool,
        siren_omega0: float,
        dropout: float,
    ):
        super().__init__()
        self.use_siren = use_siren
        self.siren_omega0 = siren_omega0
        if use_siren:
            h = max(32, hidden_dim)
            self.w1 = nn.Linear(td, h)
            self.w2 = nn.Linear(h, h)
            self.out = nn.Linear(h, 3)
            self.drop = nn.Dropout(dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(td, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),
            )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not self.use_siren:
            return self.mlp(h)
        x = self.siren_omega0 * self.w1(h)
        x = torch.sin(x)
        x = self.drop(x)
        x = torch.sin(self.w2(x))
        return self.out(x)


class SpatioTemporalGNN(nn.Module):
    """
    Competition entry: full-res k-NN + MeshGraphNet + temporal attention.
    Constructor must be callable with no args; loads ``state_dict.pt`` if present.

    Incremental features (graph path):
    - Airfoil boundary features: log1p(min dist) + unit direction to nearest surface
      sample (chunked cdist; surface may be subsampled).
    - Residual on ``velocity_in`` baseline: last frame (default) or mean over input time.
    - Optional sin-heavy head on the predicted delta (SIREN-style first layer).
    """

    def __init__(
        self,
        hidden_dim: int = 96,
        num_layers: int = 4,
        heads: int = 4,
        k: int = 12,
        t_in: int = 5,
        t_out: int = 5,
        use_fourier: bool = True,
        num_fourier: int = 6,
        dropout: float = 0.1,
        use_boundary_features: bool = True,
        max_airfoil_samples: int = 4096,
        boundary_feature_chunk: int = 8192,
        residual_baseline: str = "last",
        use_siren_head: bool = False,
        siren_omega0: float = 30.0,
        num_heads: Optional[int] = None,
        num_t_in: Optional[int] = None,
        num_t_out: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        _ = kwargs  # trainer passes latent_dim, num_modes, knn_query_chunk_size, etc.
        if num_heads is not None:
            heads = num_heads
        if num_t_in is not None:
            t_in = num_t_in
        if num_t_out is not None:
            t_out = num_t_out

        self.k = k
        self.t_in = t_in
        self.t_out = t_out
        self.use_boundary_features = use_boundary_features
        self.max_airfoil_samples = max_airfoil_samples
        self.boundary_feature_chunk = boundary_feature_chunk
        if residual_baseline not in ("last", "mean"):
            raise ValueError("residual_baseline must be 'last' or 'mean'")
        self.residual_baseline = residual_baseline
        self.use_siren_head = use_siren_head

        if use_fourier:
            self.pos_enc = FourierPosEnc(3, num_fourier)
            pos_dim = self.pos_enc.out_dim
        else:
            self.pos_enc = None
            pos_dim = 3

        b_dim = 4 if use_boundary_features else 0
        in_dim = pos_dim + t_in * 3 + 1 + b_dim
        self.node_enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.spatial = MeshGraphNetBackbone(
            hidden_dim, num_layers=num_layers, dropout=dropout
        )
        self.temporal = TemporalAttentionHead(
            hidden_dim,
            t_out=t_out,
            temporal_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
        )
        self.temporal_dim = hidden_dim
        self.decoder = DeltaDecoder(
            self.temporal_dim,
            t_out,
            hidden_dim,
            use_siren=use_siren_head,
            siren_omega0=siren_omega0,
            dropout=dropout,
        )

        weights = os.path.join(
            os.path.dirname(__file__), "state_dict.pt"
        )
        if os.path.isfile(weights):
            try:
                ck = torch.load(weights, map_location="cpu", weights_only=True)
            except TypeError:
                ck = torch.load(weights, map_location="cpu", weights_only=True)
            state = ck.get("model_state_dict", ck)
            self.load_state_dict(state, strict=False)

    def _velocity_baseline(self, vel_in: torch.Tensor) -> torch.Tensor:
        """(t_in, N, 3) -> (N, 3) fixed prior for residual."""
        if self.residual_baseline == "mean":
            return vel_in.mean(dim=0)
        return vel_in[-1]

    def _forward_single(
        self,
        pos: torch.Tensor,
        vel_in: torch.Tensor,
        airfoil_idx: torch.Tensor,
    ) -> torch.Tensor:
        n = pos.size(0)
        device = pos.device
        airfoil_idx = _sanitize_airfoil_idx(airfoil_idx, n, device)
        mask = pos.new_zeros(n, 1)
        mask[airfoil_idx] = 1.0

        neighbors, rel_pos, dists = knn_graph(pos, self.k)

        pos_feat = self.pos_enc(pos) if self.pos_enc else pos
        vel_flat = vel_in.permute(1, 0, 2).reshape(n, -1)
        parts = [pos_feat, vel_flat, mask]
        if self.use_boundary_features:
            bfeat = airfoil_boundary_features(
                pos,
                airfoil_idx,
                max_airfoil_samples=self.max_airfoil_samples,
                chunk_size=self.boundary_feature_chunk,
            )
            parts.append(bfeat)
        x = torch.cat(parts, dim=-1)
        x = self.node_enc(x)
        x = self.spatial(x, neighbors, rel_pos, dists)
        x = self.temporal(x)
        x = self.decoder(x)
        last_vel = self._velocity_baseline(vel_in)
        x = x + last_vel.unsqueeze(1)
        x_full = x.permute(1, 0, 2)
        x_full[:, airfoil_idx] = 0.0
        return x_full

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
        velocity_mean: Optional[torch.Tensor] = None,
        velocity_std: Optional[torch.Tensor] = None,
        *,
        wall_distance: Optional[torch.Tensor] = None,
        surface_frame: Optional[torch.Tensor] = None,
        knn_indices: Optional[torch.Tensor] = None,
        return_knn_indices: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        _ = (t, velocity_mean, velocity_std, wall_distance, surface_frame, knn_indices)
        b = velocity_in.size(0)
        outs = []
        for bi in range(b):
            n_pts = pos[bi].shape[0]
            out = self._forward_single(
                pos[bi],
                velocity_in[bi],
                _sanitize_airfoil_idx(idcs_airfoil[bi], n_pts, pos.device),
            )
            outs.append(out)
        stacked = torch.stack(outs, dim=0)
        if return_knn_indices:
            return stacked, None
        return stacked
