"""Variance-gated GNN corrector with EdgeConv V2 backbone."""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


def _batched_knn_graph(pos: torch.Tensor, k: int, batch: torch.Tensor,
                       chunk_size: int = 2048) -> torch.Tensor:
    sources, targets = [], []
    for b_id in batch.unique():
        mask = batch == b_id
        idx_global = mask.nonzero(as_tuple=False).squeeze(-1)
        pts = pos[mask]
        M = pts.shape[0]
        knn_local = pts.new_empty(M, k, dtype=torch.long)
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            dists = torch.cdist(pts[start:end], pts)
            dists[:, start:end].fill_diagonal_(float("inf"))
            _, knn_local[start:end] = dists.topk(k, dim=-1, largest=False)
        src = idx_global[knn_local.reshape(-1)]
        tgt = idx_global.unsqueeze(-1).expand(M, k).reshape(-1)
        sources.append(src)
        targets.append(tgt)
    return torch.stack([torch.cat(targets), torch.cat(sources)], dim=0)


def _edge_attr(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    rel = pos[edge_index[1]] - pos[edge_index[0]]
    dist = rel.norm(dim=-1, keepdim=True)
    return torch.cat([rel, dist], dim=-1)


def _make_head(hidden_dim: int) -> nn.Sequential:
    head = nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 3),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class _EdgeConvV2Block(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 4):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.norm(out)

    def message(self, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))


class EdgeConvV2Backbone(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(_EdgeConvV2Block(node_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.convs.append(_EdgeConvV2Block(hidden_dim, hidden_dim))
        self.head = _make_head(hidden_dim)

    def forward(self, x, edge_index, pos_cat):
        edge_attr = _edge_attr(pos_cat, edge_index)
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index, edge_attr)
            if i > 0:
                x_new = x_new + x
            x = x_new
        return self.head(x)


class WakeCorrectorGNN(nn.Module):
    """Variance-gated GNN corrector with EdgeConv V2 backbone."""

    def __init__(self, node_dim: int = 10, hidden_dim: int = 192,
                 n_layers: int = 10, k: int = 16,
                 top_fraction: float = 0.4,
                 mask_sharpness: float = 5.0) -> None:
        super().__init__()
        self.k = k
        self.top_fraction = top_fraction
        self.mask_sharpness = mask_sharpness
        self.mp = EdgeConvV2Backbone(
            node_dim=node_dim, hidden_dim=hidden_dim, n_layers=n_layers,
        )

    def forward(self, u_base: torch.Tensor, pos: torch.Tensor,
                velocity_in: torch.Tensor,
                airfoil_mask: torch.Tensor) -> torch.Tensor:
        B, K, N, C = u_base.shape

        var_per_point = velocity_in.var(dim=1).sum(dim=-1)
        var_z = (var_per_point - var_per_point.mean(1, keepdim=True)) / (
            var_per_point.std(1, keepdim=True) + 1e-8
        )
        soft_mask = torch.sigmoid(self.mask_sharpness * var_z)
        soft_mask = soft_mask * (1 - airfoil_mask)

        n_select = max(int(self.top_fraction * N), self.k + 1)
        _, top_idx = var_per_point.topk(n_select, dim=1)

        pos_parts, batch_parts, mask_parts = [], [], []
        for b in range(B):
            idx = top_idx[b]
            pos_parts.append(pos[b, idx])
            batch_parts.append(idx.new_full((n_select,), b))
            mask_parts.append(soft_mask[b, idx])

        pos_cat = torch.cat(pos_parts, dim=0)
        batch_cat = torch.cat(batch_parts, dim=0)
        mask_cat = torch.cat(mask_parts, dim=0).unsqueeze(-1)

        edge_index = _batched_knn_graph(pos_cat, k=self.k, batch=batch_cat)

        v_last = velocity_in[:, -1]
        var_feat = var_per_point.unsqueeze(-1)
        delta_full = u_base.new_zeros(B, K, N, C)

        for k_step in range(K):
            feat_parts = []
            for b in range(B):
                idx = top_idx[b]
                feat_parts.append(torch.cat([
                    u_base[b, k_step, idx],
                    v_last[b, idx],
                    pos[b, idx],
                    var_feat[b, idx],
                ], dim=-1))

            x = torch.cat(feat_parts, dim=0)
            delta = self.mp(x, edge_index, pos_cat)
            delta = delta * mask_cat

            for b in range(B):
                idx = top_idx[b]
                start, end = b * n_select, (b + 1) * n_select
                delta_full[b, k_step, idx] = delta[start:end]

        result = u_base + delta_full
        result = result * (1 - airfoil_mask[:, None, :, None])
        return result
