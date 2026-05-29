from __future__ import annotations

import os
from typing import List, Optional

import torch
from torch.utils.checkpoint import checkpoint as _grad_ckpt
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, ReLU, Sequential
from torch_cluster import fps, knn, knn_graph
from torch_geometric.utils import scatter

from .features import compute_geometry_features_single


class GraphConvBlock(Module):
    """KNN graph convolution with edge features (relative position + distance)."""

    _EDGE_DIM = 4  # rel_pos(3) + distance(1)

    def __init__(self, hidden_dim: int, k_neighbors: int = 16, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors

        self.linear_agg = Linear(hidden_dim * 2 + self._EDGE_DIM, hidden_dim)
        self.norm = LayerNorm(hidden_dim)
        self.activation = ReLU()
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """
        Args:
            x: (num_pos, hidden_dim)
            pos: (num_pos, 3)
        Returns:
            (num_pos, hidden_dim)
        """
        N = pos.shape[0]
        k = min(self.k_neighbors, N - 1)

        edge_index = knn_graph(pos, k=k, loop=False)
        src, dst = edge_index

        x_nbr_agg = scatter(x[src], dst, dim=0, dim_size=N, reduce="mean")

        rel_pos = pos[src] - pos[dst]
        edge_dist = rel_pos.norm(dim=-1, keepdim=True)
        edge_feat_agg = scatter(
            torch.cat([rel_pos, edge_dist], dim=-1),
            dst,
            dim=0,
            dim_size=N,
            reduce="mean",
        )

        combined = torch.cat([x, x_nbr_agg, edge_feat_agg], dim=-1)
        return self.dropout(self.activation(self.norm(self.linear_agg(combined))))


class GraphTransformerBlock(Module):
    """Graph transformer with multi-head attention and geometric edge bias.

    Attention scores include a Graphormer-style per-head bias derived from
    relative position and distance, letting each head specialize to different
    spatial scales.
    """

    _EDGE_DIM = 4  # rel_pos(3) + distance(1)

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        k_neighbors: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.k_neighbors = k_neighbors
        self.scale = self.head_dim**-0.5

        self.norm1 = LayerNorm(hidden_dim)
        self.q_proj = Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_proj = Linear(self._EDGE_DIM, num_heads)
        self.out_proj = Linear(hidden_dim, hidden_dim)

        self.norm2 = LayerNorm(hidden_dim)
        self.ffn = Sequential(
            Linear(hidden_dim, hidden_dim * 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim * 2, hidden_dim),
            Dropout(dropout),
        )
        self.attn_dropout = Dropout(dropout)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """
        Args:
            x: (N, hidden_dim)
            pos: (N, 3)
        Returns:
            (N, hidden_dim)
        """
        N = pos.shape[0]
        k = min(self.k_neighbors, N - 1)

        edge_index = knn_graph(pos, k=k, loop=False)
        src, dst = edge_index  # src=neighbor, dst=query

        # Per-head attention bias from geometric edge features
        rel_pos = pos[src] - pos[dst]
        edge_dist = rel_pos.norm(dim=-1, keepdim=True)
        edge_bias = self.edge_proj(torch.cat([rel_pos, edge_dist], dim=-1))  # (E, H)

        # Pre-norm multi-head attention
        h = self.norm1(x)
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(h).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(h).view(N, self.num_heads, self.head_dim)

        # Attention score: Q[dst] · K[src] * scale + edge_bias
        # Use einsum for efficient batched dot product
        attn = (
            torch.einsum("ehd,ehd->eh", Q[dst], K[src]) * self.scale + edge_bias
        )  # (E, H)

        # Numerically stable scatter softmax per destination node
        attn_max = scatter(attn, dst, dim=0, dim_size=N, reduce="max")
        attn = torch.exp(attn - attn_max[dst])
        attn_sum = scatter(attn, dst, dim=0, dim_size=N, reduce="sum").clamp(min=1e-6)
        attn = self.attn_dropout(attn / attn_sum[dst])  # (E, H)

        # Weighted value aggregation
        weighted_V = (V[src] * attn.unsqueeze(-1)).reshape(
            -1, self.num_heads * self.head_dim
        )
        agg = scatter(weighted_V, dst, dim=0, dim_size=N, reduce="sum")

        x = x + self.out_proj(agg)
        x = x + self.ffn(self.norm2(x))
        return x


def _knn_interpolate(
    x_coarse: Tensor,
    pos_coarse: Tensor,
    pos_fine: Tensor,
    k: int = 3,
) -> Tensor:
    """Inverse-distance weighted interpolation from a coarse to a fine point cloud."""
    k_eff = min(k, pos_coarse.shape[0])
    # knn(x, y, k): edge_index[0] = y indices (fine queries), edge_index[1] = x indices (coarse neighbors)
    assign = knn(pos_coarse, pos_fine, k=k_eff)  # [2, N_fine * k_eff]
    fine_idx, coarse_idx = assign[0], assign[1]

    dist = (
        (pos_fine[fine_idx] - pos_coarse[coarse_idx])
        .norm(dim=-1, keepdim=True)
        .clamp(min=1e-8)
    )
    weight = 1.0 / dist
    weight_sum = scatter(
        weight, fine_idx, dim=0, dim_size=pos_fine.shape[0], reduce="sum"
    ).clamp(min=1e-8)
    weight = weight / weight_sum[fine_idx]

    return scatter(
        x_coarse[coarse_idx] * weight,
        fine_idx,
        dim=0,
        dim_size=pos_fine.shape[0],
        reduce="sum",
    )


class DeltaGraph(Module):
    """Multi-scale graph transformer for airflow delta prediction.

    Architecture:
      1. Linear input projection at full resolution (L0).
      2. FPS downsample to L1 (~8 % of points); 2 x GraphTransformerBlock.
            3. Decoder: interpolate L1 → L0 with skip fusion.
            4. Linear output head predicts velocity delta for all 5 output timesteps.
            5. Add delta to linear baseline extrapolation; zero out airfoil points.
    """

    num_t_in = 5
    num_t_out = 5

    hidden_dim = 256
    dropout_probability = 0.1

    adjacency_radius = 0.02
    max_surface_points = 512
    surface_chunk_size = 8192
    normal_k_neighbors = 16

    graph_k_neighbors = 16
    num_heads = 4

    # Two-level subsampling (applied independently per sample in the batch loop)
    l1_ratio = 0.08  # ~8k points from 100k full resolution
    interp_k = 3  # neighbors for knn interpolation in decoder

    l1_layers = 2

    # Set True to trade computation for memory (recomputes activations on backward)
    gradient_checkpointing = False

    def __init__(self, load_weights: bool = True):
        super().__init__()

        # Features: pos(3), velocity_in_flat(15), time(10), geometry(7)
        in_dim = 3 + self.num_t_in * 3 + (self.num_t_in + self.num_t_out) + 7
        out_dim = self.num_t_out * 3

        self.linear_in = Linear(in_dim, self.hidden_dim)
        self.norm_in = LayerNorm(self.hidden_dim)
        self.activation = ReLU()
        self.dropout = Dropout(self.dropout_probability)

        self.l1_blocks = torch.nn.ModuleList(
            [
                GraphTransformerBlock(
                    self.hidden_dim,
                    self.num_heads,
                    self.graph_k_neighbors,
                    self.dropout_probability,
                )
                for _ in range(self.l1_layers)
            ]
        )

        # Decoder fusion: concatenate skip + interpolated → hidden
        self.fuse_l0 = Linear(self.hidden_dim * 2, self.hidden_dim)
        self.norm_fuse_l0 = LayerNorm(self.hidden_dim)

        self.linear_mid = Linear(self.hidden_dim, self.hidden_dim)
        self.norm_mid = LayerNorm(self.hidden_dim)
        self.linear_out = Linear(self.hidden_dim, out_dim)

        if load_weights:
            path_state_dict = os.path.join(os.path.dirname(__file__), "state_dict.pt")
            if not os.path.exists(path_state_dict):
                raise FileNotFoundError(
                    "Missing model weights at "
                    f"{path_state_dict}. Create them before using DeltaGraph()."
                )
            state_dict = torch.load(path_state_dict, map_location="cpu")
            self.load_state_dict(state_dict)

    def _baseline_extrapolation(self, t: Tensor, velocity_in: Tensor) -> Tensor:
        t_in = t[:, : self.num_t_in]
        t_out = t[:, self.num_t_in : self.num_t_in + self.num_t_out]

        dt = (t_in[:, -1] - t_in[:, -2]).clamp(min=1e-6)
        slope = (velocity_in[:, -1] - velocity_in[:, -2]) / dt[:, None, None]

        delta_t = t_out - t_in[:, -1:]
        return velocity_in[:, -1:] + slope[:, None] * delta_t[:, :, None, None]

    def _compute_geometry_batch(
        self, pos: Tensor, idcs_airfoil: List[Tensor]
    ) -> Tensor:
        features = []
        for pos_i, idcs_i in zip(pos, idcs_airfoil):
            feat_i = compute_geometry_features_single(
                pos=pos_i,
                idcs_airfoil=idcs_i.to(pos_i.device),
                adjacency_radius=self.adjacency_radius,
                max_surface_points=self.max_surface_points,
                chunk_size=self.surface_chunk_size,
                normal_k_neighbors=self.normal_k_neighbors,
            )
            features.append(feat_i)
        return torch.stack(features, dim=0)

    def _build_airfoil_mask_batch(
        self,
        batch_size: int,
        num_pos: int,
        idcs_airfoil: List[Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        mask = torch.zeros((batch_size, 1, num_pos, 1), device=device, dtype=dtype)
        for batch_idx, idcs in enumerate(idcs_airfoil):
            if idcs.numel() == 0:
                continue
            mask[batch_idx, 0, idcs.long(), 0] = 1.0
        return mask

    def forward(
        self,
        t: Tensor,
        pos: Tensor,
        idcs_airfoil: List[Tensor],
        velocity_in: Tensor,
        geom_feat: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            t: (batch, 10)
            pos: (batch, num_pos, 3)
            idcs_airfoil: list of airfoil index tensors, one per sample
            velocity_in: (batch, 5, num_pos, 3)
            geom_feat: optional precomputed geometry features (batch, num_pos, 7).
                       When provided, skips on-the-fly geometry computation.
        Returns:
            velocity_out: (batch, 5, num_pos, 3)
        """
        batch_size, num_t_in, num_pos, _ = velocity_in.shape
        if num_t_in != self.num_t_in:
            raise ValueError(
                f"Expected {self.num_t_in} input timesteps but got {num_t_in}."
            )

        baseline = self._baseline_extrapolation(t=t, velocity_in=velocity_in)

        velocity_flat = velocity_in.transpose(1, 2).reshape(
            batch_size, num_pos, num_t_in * 3
        )
        time_feat = t[:, None, :].expand(batch_size, num_pos, t.shape[1])

        if geom_feat is None:
            with torch.no_grad():
                geom_feat = self._compute_geometry_batch(
                    pos=pos, idcs_airfoil=idcs_airfoil
                )

        geom_feat = geom_feat.to(dtype=velocity_in.dtype, device=velocity_in.device)

        # Per-point input features
        x_input = torch.cat((pos, velocity_flat, time_feat, geom_feat), dim=-1)
        x_input = self.dropout(self.activation(self.norm_in(self.linear_in(x_input))))

        # Multi-scale graph processing per sample (variable idcs_airfoil precludes batching)
        x_out = []
        for batch_idx in range(batch_size):
            x_s = x_input[batch_idx]  # (num_pos, hidden_dim)  — L0 skip features
            pos_s = pos[batch_idx]  # (num_pos, 3)

            # Encoder:
            l1_idcs = fps(pos_s, ratio=self.l1_ratio, random_start=self.training)
            pos_l1 = pos_s[l1_idcs]
            x_l1 = x_s[l1_idcs]

            for block in self.l1_blocks:
                if self.gradient_checkpointing and self.training:
                    x_l1 = _grad_ckpt(block, x_l1, pos_l1, use_reentrant=False)
                else:
                    x_l1 = block(x_l1, pos_l1)

            # Decoder:
            # L1 → L0: interpolate medium features and fuse with L0 skip
            x_l1_up = _knn_interpolate(x_l1, pos_l1, pos_s, k=self.interp_k)
            x_s = self.dropout(
                self.activation(
                    self.norm_fuse_l0(self.fuse_l0(torch.cat([x_s, x_l1_up], dim=-1)))
                )
            )

            x_out.append(x_s)

        x = torch.stack(x_out, dim=0)
        x = self.dropout(self.activation(self.norm_mid(self.linear_mid(x))))
        delta = (
            self.linear_out(x)
            .view(batch_size, num_pos, self.num_t_out, 3)
            .transpose(1, 2)
        )

        velocity_out = baseline + delta

        airfoil_mask = self._build_airfoil_mask_batch(
            batch_size=batch_size,
            num_pos=num_pos,
            idcs_airfoil=idcs_airfoil,
            device=velocity_out.device,
            dtype=velocity_out.dtype,
        )
        return velocity_out * (1.0 - airfoil_mask)
