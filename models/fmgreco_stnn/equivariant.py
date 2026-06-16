"""
SO(3)-equivariant message passing on k-NN graphs (torch-only, no PyG).

Vector channels v ∈ R^{N×d_v×3} transform as 3-vectors under rotations; scalar
channels s ∈ R^{N×d_s} are invariant. Edge messages use only invariant pair
features (distances, scalar states) and decompose neighbor vectors into
components parallel / perpendicular to the edge direction r̂_ij, which preserves
equivariance. Reflections are *not* symmetries of this construction (chiral
geometries can be distinguished), unlike full E(3) parity-even models.

Optional pressure channels can be concatenated in ``gnn_base`` (see ``use_pressure``);
here ``s`` remains a learned invariant unless you rely solely on that backbone path.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EquivariantVelocityLift(nn.Module):
    """Lift u ∈ R^{N×3} to v ∈ R^{N×d_v×3} with learned per-channel positive scalars."""

    def __init__(self, d_v: int):
        super().__init__()
        self.d_v = d_v
        self.log_scale = nn.Parameter(torch.zeros(d_v))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (N, 3) -> (N, d_v, 3), equivariant under SO(3)."""
        w = self.log_scale.exp().view(1, -1, 1)
        return u.unsqueeze(1) * w


class SO3EquivariantKNNLayer(nn.Module):
    """One k-NN message-passing step on (s, v) with parallel/perp vector mixing."""

    def __init__(self, d_s: int, d_v: int, dropout: float = 0.1):
        super().__init__()
        pair_in = 2 * d_s + 1
        mid = max(32, pair_in)
        self.gate_mlp = nn.Sequential(
            nn.Linear(pair_in, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Linear(mid, 2 * d_v),
        )
        self.scalar_msg_mlp = nn.Sequential(
            nn.Linear(pair_in, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Linear(mid, d_s),
        )
        self.dropout = nn.Dropout(dropout)
        self.s_norm = nn.LayerNorm(d_s)

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        neighbors: torch.Tensor,
        rel_pos: torch.Tensor,
        dists: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        s: (N, d_s), v: (N, d_v, 3)
        neighbors: (N, k), rel_pos: (N, k, 3), dists: (N, k)
        """
        eps = 1e-8
        _, k = neighbors.shape
        s_j = s[neighbors]
        v_j = v[neighbors]
        dist_sq = (dists.unsqueeze(-1) + eps).pow(2)
        s_i = s.unsqueeze(1).expand(-1, k, -1)
        pair = torch.cat([s_i, s_j, dist_sq], dim=-1)

        gates = self.dropout(self.gate_mlp(pair))
        g_par, g_perp = gates.chunk(2, dim=-1)

        r_hat = rel_pos / (dists.unsqueeze(-1).clamp_min(eps))
        dot = (v_j * r_hat.unsqueeze(2)).sum(dim=-1)
        v_par = r_hat.unsqueeze(2) * dot.unsqueeze(-1)
        v_perp = v_j - v_par
        msg_v = g_par.unsqueeze(-1) * v_par + g_perp.unsqueeze(-1) * v_perp
        agg_v = msg_v.mean(dim=1)

        delta_s = self.scalar_msg_mlp(pair).mean(dim=1)
        s_out = self.s_norm(s + delta_s)
        v_out = v + agg_v
        return s_out, v_out


class SO3EquivariantBackbone(nn.Module):
    """Stack of SO3-equivariant k-NN layers; keeps (s, v) throughout."""

    def __init__(
        self,
        d_s: int,
        d_v: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            SO3EquivariantKNNLayer(d_s, d_v, dropout=dropout)
            for _ in range(num_layers)
        )

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        neighbors: torch.Tensor,
        rel_pos: torch.Tensor,
        dists: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            s, v = layer(s, v, neighbors, rel_pos, dists)
        return s, v


class SO3ToHiddenMerge(nn.Module):
    """Map invariant scalars + vector magnitudes to a single hidden vector for the temporal head."""

    def __init__(self, d_s: int, d_v: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_s + d_v, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        inv = (v * v).sum(dim=-1).sqrt()
        return self.net(torch.cat([s, inv], dim=-1))
