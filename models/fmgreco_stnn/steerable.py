"""
Steerable (E(3)-equivariant) k-NN message passing via e3nn irreps + tensor products.

Node features live in a fixed ``o3.Irreps`` (scalars 0e, vectors 1o, rank-2 2e, ...).
Edge attributes are real spherical harmonics :math:`Y_\ell(\hat r_{ij})` up to ``lmax``.
Each layer applies a fully connected tensor product
:math:`m_{ij} = f_j \\otimes Y(\\hat r_{ij})`, averaged over neighbors, with a residual.

This is a stronger geometric prior than the hand-built SO(3) parallel/perp mixer in
``equivariant.py`` (full steerable channels, Clebsch–Gordan mixing).

Reflection parity follows e3nn's standard ``e``/``o`` labels; the construction is
E(3)-equivariant when irreps and harmonics are chosen consistently.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3


def hidden_dim_to_irreps(hidden_dim: int) -> o3.Irreps:
    """
    Pack ``hidden_dim`` channels into 0e + 1o + 2e multiplets with exact total dimension.

    Greedy: use as many 2e (dim 5) blocks as fit, then 1o (dim 3), remainder 0e.
    """
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be positive")
    remaining = hidden_dim
    nt = remaining // 5
    remaining -= nt * 5
    nv = remaining // 3
    remaining -= nv * 3
    ns = remaining
    return o3.Irreps(f"{ns}x0e + {nv}x1o + {nt}x2e")


def irreps_invariant_dim(irreps: o3.Irreps) -> int:
    """Dimension of rotation-invariant summary (l=0 blocks + per-multiplet norms)."""
    n = 0
    for mul, ir in irreps:
        if ir.l == 0:
            n += mul * ir.dim
        else:
            n += mul
    return n


def irreps_invariant_concat(x: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    """(N, irreps.dim) -> (N, irreps_invariant_dim) — SO(3)-invariant node features."""
    i = 0
    outs: list[torch.Tensor] = []
    for mul, ir in irreps:
        d = mul * ir.dim
        chunk = x[:, i : i + d]
        if ir.l == 0:
            outs.append(chunk)
        else:
            chunk = chunk.reshape(x.shape[0], mul, ir.dim)
            outs.append(chunk.norm(dim=-1))
        i += d
    return torch.cat(outs, dim=-1)


def _double_multiplicities(irreps: o3.Irreps) -> o3.Irreps:
    """Widen irreps so TP(node, Y_l) → mid almost always has nonzero paths."""
    return o3.Irreps([(mul * 2, ir) for mul, ir in irreps])


class SteerableKNNLayer(nn.Module):
    """One residual steerable message-passing step on a k-NN graph."""

    def __init__(self, irreps_node: o3.Irreps, lmax: int, dropout: float):
        super().__init__()
        self.irreps_node = irreps_node
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.irreps_mid = _double_multiplicities(irreps_node)
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_node,
            self.irreps_sh,
            self.irreps_mid,
            shared_weights=True,
        )
        self.compress = o3.Linear(self.irreps_mid, irreps_node)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.Tensor,
        rel_pos: torch.Tensor,
        dists: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (N, irreps.dim), neighbors: (N, k), rel_pos: (N, k, 3), dists: (N, k).
        """
        dtype = x.dtype
        x = x.float()
        rel_pos = rel_pos.float()
        dists = dists.float()

        n, k, _ = rel_pos.shape
        eps = 1e-8
        hat = rel_pos / dists.unsqueeze(-1).clamp_min(eps)
        flat_hat = hat.reshape(-1, 3)
        sh = o3.spherical_harmonics(
            self.irreps_sh,
            flat_hat,
            normalize=True,
            normalization="component",
        )
        sh = sh.reshape(n, k, -1)

        x_j = x[neighbors]
        flat_xj = x_j.reshape(-1, self.irreps_node.dim)
        flat_sh = sh.reshape(-1, self.irreps_sh.dim)
        msg = self.compress(self.tp(flat_xj, flat_sh))
        msg = msg.reshape(n, k, -1).mean(dim=1)

        out = x + self.drop(msg)
        return out.to(dtype)


class SteerableEquivariantBackbone(nn.Module):
    """Stack of steerable k-NN layers."""

    def __init__(
        self,
        irreps: o3.Irreps,
        num_layers: int,
        lmax: int,
        dropout: float,
    ):
        super().__init__()
        self.irreps = irreps
        self.layers = nn.ModuleList(
            [
                SteerableKNNLayer(irreps, lmax=lmax, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.Tensor,
        rel_pos: torch.Tensor,
        dists: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, neighbors, rel_pos, dists)
        return x


class SteerableToHiddenMerge(nn.Module):
    """Map steerable irreps to isotropic ``hidden_dim`` channels for the temporal head."""

    def __init__(self, irreps: o3.Irreps, hidden_dim: int):
        super().__init__()
        self.irreps = irreps
        inv_dim = irreps_invariant_dim(irreps)
        self.net = nn.Sequential(
            nn.Linear(inv_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv = irreps_invariant_concat(x, self.irreps)
        return self.net(inv)
