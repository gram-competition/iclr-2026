"""MeshGraphNet-style spatial encoder (edge + node updates)."""

from __future__ import annotations

import torch
import torch.nn as nn


class EdgeEncoder(nn.Module):
    """Project relative offsets + distance into hidden edge features (dim per edge)."""

    def __init__(self, hidden_dim: int, edge_in: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, rel_pos: torch.Tensor, dists: torch.Tensor) -> torch.Tensor:
        raw = torch.cat([rel_pos, dists.unsqueeze(-1)], dim=-1)
        return self.net(raw)


class ResidualMGNLayer(nn.Module):
    """MeshGraphNet-style block with explicit residuals on edge and node streams."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, k, _ = edge_feat.shape
        x_i = x.unsqueeze(1).expand(-1, k, -1)
        x_j = x[neighbors]
        edge_input = torch.cat([x_i, x_j, edge_feat], dim=-1)
        edge_feat = edge_feat + self.edge_mlp(edge_input)
        msg = edge_feat.mean(dim=1)
        x = x + self.node_mlp(torch.cat([x, msg], dim=-1))
        return x, edge_feat


class MeshGraphNetBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_enc = EdgeEncoder(hidden_dim)
        self.layers = nn.ModuleList(
            ResidualMGNLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.Tensor,
        rel_pos: torch.Tensor,
        dists: torch.Tensor,
    ) -> torch.Tensor:
        edge_feat = self.edge_enc(rel_pos, dists)
        for layer in self.layers:
            x, edge_feat = layer(x, neighbors, edge_feat)
        return x
