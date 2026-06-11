"""Self-contained building blocks for the Spatiotemporal MNO model.

This module intentionally duplicates the small set of layers and geometry
helpers the model needs so that ``models/spatiotemporal_mno`` has no import
dependency on any other model package (e.g. ``models/mlp``). Keeping the
package self-contained makes the submission easy to merge and reuse in
isolation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_cluster import knn_graph as torch_cluster_knn_graph

    HAS_TORCH_CLUSTER = True
except ImportError:
    torch_cluster_knn_graph = None
    HAS_TORCH_CLUSTER = False


def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    num_hidden_layers: int = 2,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(num_hidden_layers - 1):
        layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.GELU()))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class GlobalDimensionShrinkageAttention(nn.Module):
    """Project point tokens to a low-rank mode space, attend globally, then lift back."""

    def __init__(self, latent_dim: int, num_modes: int, num_heads: int):
        super().__init__()
        self.down_projector = _make_mlp(latent_dim, latent_dim, num_modes)
        self.mode_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.up_projector = _make_mlp(latent_dim, latent_dim, num_modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_scores = torch.softmax(self.down_projector(x), dim=1)
        modes = torch.einsum("bnm,bnd->bmd", down_scores, x)

        attended_modes, _ = self.mode_attention(
            modes,
            modes,
            modes,
            need_weights=False,
        )

        up_scores = torch.softmax(self.up_projector(x), dim=-1)
        return torch.einsum("bnm,bmd->bnd", up_scores, attended_modes)


class LocalGraphAttention(nn.Module):
    """Attention over Euclidean kNN graph for unstructured point clouds."""

    def __init__(self, latent_dim: int, k: int, query_chunk_size: int = 2048):
        super().__init__()
        self.k = k
        self.query_chunk_size = query_chunk_size
        self.pos_encoding = _make_mlp(3, latent_dim, latent_dim)
        self.attn_kernel = _make_mlp(latent_dim, latent_dim, latent_dim)
        self.value_mlp = _make_mlp(latent_dim, latent_dim, latent_dim)

    @staticmethod
    def _batched_gather(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch_size, num_pos, feat_dim = features.shape
        _, num_query, k = indices.shape

        offsets = (
            torch.arange(batch_size, device=features.device, dtype=torch.long)
            .view(batch_size, 1, 1)
            .mul(num_pos)
        )
        flat_indices = (indices + offsets).reshape(-1)
        flat_features = features.reshape(batch_size * num_pos, feat_dim)
        gathered = flat_features.index_select(0, flat_indices)
        return gathered.view(batch_size, num_query, k, feat_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        knn_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_pos, _ = x.shape
        out = torch.empty_like(x)

        for start in range(0, num_pos, self.query_chunk_size):
            end = min(start + self.query_chunk_size, num_pos)
            idx_chunk = knn_indices[:, start:end, :]

            center_x = x[:, start:end, :].unsqueeze(2).expand(-1, -1, self.k, -1)
            center_pos = pos[:, start:end, :].unsqueeze(2)
            neigh_x = self._batched_gather(x, idx_chunk)
            neigh_pos = self._batched_gather(pos, idx_chunk)

            rel_pos = neigh_pos - center_pos
            delta = self.pos_encoding(rel_pos)

            attn_input = center_x - neigh_x + delta
            attn_weights = torch.softmax(self.attn_kernel(attn_input), dim=2)

            values = self.value_mlp(neigh_x) + delta
            out[:, start:end, :] = (attn_weights * values).sum(dim=2)

        return out


class MicroPointWiseAttention(nn.Module):
    """Point-wise token reweighting using MLP scores normalized across points."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.score_mlp = _make_mlp(latent_dim, latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize across sequence length N to match Micro attention formulation.
        score_p = torch.softmax(self.score_mlp(x), dim=1)
        return x + score_p * x


class MNOBlock(nn.Module):
    """Single Multiscale Neural Operator block with parallel branches and fusion."""

    def __init__(
        self,
        latent_dim: int,
        num_modes: int,
        num_heads: int,
        k: int,
        graph_query_chunk_size: int,
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(latent_dim)

        self.global_attention = GlobalDimensionShrinkageAttention(
            latent_dim=latent_dim,
            num_modes=num_modes,
            num_heads=num_heads,
        )
        self.local_attention = LocalGraphAttention(
            latent_dim=latent_dim,
            k=k,
            query_chunk_size=graph_query_chunk_size,
        )
        self.micro_attention = MicroPointWiseAttention(latent_dim=latent_dim)

        self.fusion_mlp = _make_mlp(latent_dim, 2 * latent_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        knn_indices: torch.Tensor,
    ) -> torch.Tensor:
        x_norm = self.pre_norm(x)

        x_global = self.global_attention(x_norm)
        x_local = self.local_attention(x_norm, pos, knn_indices)
        x_micro = self.micro_attention(x_norm)

        fused = x_global + x_local + x_micro
        return x + self.fusion_mlp(fused)


class FourierTimeEmbedding(nn.Module):
    """Map scalar timestamps to high-dimensional sinusoidal features.

    For each of ``num_freqs`` learnable frequencies, produces ``[sin(ω·t), cos(ω·t)]``,
    giving a ``2 * num_freqs``-dimensional embedding per scalar input.
    The frequencies are initialised log-linearly and kept trainable so the
    model can adapt the spectral coverage during training.
    """

    def __init__(self, num_freqs: int = 16):
        super().__init__()
        self.num_freqs = num_freqs
        self.out_dim = 2 * num_freqs
        # Log-linear init spanning a wide range of temporal scales.
        init_freqs = torch.linspace(0.0, 4.0, num_freqs).exp()  # ~1 … ~55
        self.freqs = nn.Parameter(init_freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``t``: arbitrary shape ``(...)``.  Returns ``(..., 2*num_freqs)``."""
        # (..., 1) * (num_freqs,) -> (..., num_freqs)
        angles = t.unsqueeze(-1) * self.freqs
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


def build_airfoil_mask(
    idcs_airfoil: list[torch.Tensor],
    batch_size: int,
    num_pos: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.zeros((batch_size, num_pos, 1), device=device, dtype=dtype)
    for b, indices in enumerate(idcs_airfoil):
        if indices.numel() == 0:
            continue
        mask[b, indices.to(device=device, dtype=torch.long), 0] = 1.0
    return mask


def compute_wall_distance(
    pos: torch.Tensor,
    idcs_airfoil: list[torch.Tensor],
) -> torch.Tensor:
    """Compute min Euclidean distance from each point to the nearest airfoil point.

    Returns a ``(B, N, 1)`` tensor of wall distances, log-transformed for
    better gradient behaviour: ``log(1 + d)``.
    """
    batch_size, num_pos, _ = pos.shape
    wall_dist = torch.zeros(
        (batch_size, num_pos, 1), device=pos.device, dtype=pos.dtype
    )
    for b, indices in enumerate(idcs_airfoil):
        if indices.numel() == 0:
            continue
        airfoil_pos = pos[b, indices.to(device=pos.device, dtype=torch.long), :]  # (A, 3)
        # cdist: (N, A) -> min over airfoil points
        dists = torch.cdist(pos[b].unsqueeze(0), airfoil_pos.unsqueeze(0)).squeeze(0)  # (N, A)
        min_dist = dists.min(dim=-1).values  # (N,)
        wall_dist[b, :, 0] = torch.log1p(min_dist)
    return wall_dist


def compute_surface_frame(
    pos: torch.Tensor,
    idcs_airfoil: list[torch.Tensor],
) -> torch.Tensor:
    """Compute local surface coordinate frame on the fly (inference path).

    Returns a ``(B, N, 9)`` tensor: ``[n, t1, t2]`` per point.
    """
    batch_size, num_pos, _ = pos.shape
    frame = torch.zeros(
        (batch_size, num_pos, 9), device=pos.device, dtype=pos.dtype
    )
    for b, indices in enumerate(idcs_airfoil):
        if indices.numel() == 0:
            frame[b, :, 2] = 1.0   # default normal = z
            frame[b, :, 3] = 1.0   # default t1 = x
            frame[b, :, 7] = 1.0   # default t2 = y
            continue
        airfoil_pos = pos[b, indices.to(device=pos.device, dtype=torch.long), :]  # (A, 3)
        dists = torch.cdist(pos[b].unsqueeze(0), airfoil_pos.unsqueeze(0)).squeeze(0)  # (N, A)
        min_dist, min_idx = dists.min(dim=-1)
        nearest = airfoil_pos[min_idx]  # (N, 3)
        diff = pos[b] - nearest
        dist_clamped = min_dist.unsqueeze(-1).clamp(min=1e-12)
        normals = diff / dist_clamped

        # On-surface points: default normal to z-up
        on_surface = min_dist < 1e-8
        if on_surface.any():
            normals[on_surface] = torch.tensor(
                [0.0, 0.0, 1.0], device=pos.device, dtype=pos.dtype
            )

        normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        ref = torch.zeros_like(normals)
        ref[:, 0] = 1.0
        parallel_mask = normals[:, 0].abs() > 0.9
        ref[parallel_mask, 0] = 0.0
        ref[parallel_mask, 1] = 1.0

        t1 = torch.cross(normals, ref, dim=-1)
        t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        t2 = torch.cross(normals, t1, dim=-1)
        t2 = t2 / t2.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        frame[b, :, 0:3] = normals
        frame[b, :, 3:6] = t1
        frame[b, :, 6:9] = t2
    return frame
