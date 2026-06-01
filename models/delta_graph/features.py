from __future__ import annotations

from typing import Tuple

import torch


def _geometry_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def build_airfoil_mask(
    num_pos: int,
    idcs_airfoil: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.zeros((num_pos,), device=device, dtype=dtype)
    if idcs_airfoil.numel() > 0:
        mask[idcs_airfoil.long()] = 1.0
    return mask.unsqueeze(-1)


def sample_surface_points(
    surface_pos: torch.Tensor,
    max_surface_points: int,
) -> torch.Tensor:
    if surface_pos.shape[0] <= max_surface_points:
        return surface_pos

    idcs = (
        torch.linspace(
            0,
            surface_pos.shape[0] - 1,
            steps=max_surface_points,
            device=surface_pos.device,
        )
        .round()
        .long()
    )
    return surface_pos[idcs]


def nearest_surface_distance_and_index(
    pos: torch.Tensor,
    surface_pos: torch.Tensor,
    chunk_size: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    compute_dtype = _geometry_compute_dtype(pos.dtype)
    pos_work = pos.to(dtype=compute_dtype)
    surface_pos_work = surface_pos.to(dtype=compute_dtype)

    num_pos = pos_work.shape[0]
    min_dist = torch.empty((num_pos,), device=pos.device, dtype=compute_dtype)
    min_idx = torch.empty((num_pos,), device=pos.device, dtype=torch.long)

    with torch.autocast(device_type=pos.device.type, enabled=False):
        for start in range(0, num_pos, chunk_size):
            stop = min(start + chunk_size, num_pos)
            dists = torch.cdist(pos_work[start:stop], surface_pos_work)
            min_chunk, min_idcs_chunk = torch.min(dists, dim=1)
            min_dist[start:stop] = min_chunk
            min_idx[start:stop] = min_idcs_chunk

    return min_dist.to(dtype=pos.dtype).unsqueeze(-1), min_idx


def compute_surface_normals(
    surface_pos: torch.Tensor,
    k_neighbors: int = 16,
) -> torch.Tensor:
    compute_dtype = _geometry_compute_dtype(surface_pos.dtype)
    surface_pos_work = surface_pos.to(dtype=compute_dtype)

    num_surface = surface_pos.shape[0]
    if num_surface < 4:
        normals = torch.zeros_like(surface_pos_work)
        normals[:, 2] = 1.0
        return normals.to(dtype=surface_pos.dtype)

    k_eff = min(k_neighbors, num_surface - 1)
    with torch.autocast(device_type=surface_pos.device.type, enabled=False):
        dists = torch.cdist(surface_pos_work, surface_pos_work)
        dists.fill_diagonal_(float("inf"))
        knn_idcs = torch.topk(dists, k=k_eff, dim=1, largest=False).indices

        neighbor_pos = surface_pos_work[knn_idcs]
        centered = neighbor_pos - neighbor_pos.mean(dim=1, keepdim=True)

        cov = torch.matmul(centered.transpose(1, 2), centered)
        _, eigvecs = torch.linalg.eigh(cov)

        normals = eigvecs[:, :, 0]
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    return normals.to(dtype=surface_pos.dtype)


def compute_geometry_features_single(
    pos: torch.Tensor,
    idcs_airfoil: torch.Tensor,
    adjacency_radius: float,
    max_surface_points: int,
    chunk_size: int,
    normal_k_neighbors: int,
) -> torch.Tensor:
    num_pos = pos.shape[0]
    device = pos.device
    dtype = pos.dtype
    compute_dtype = _geometry_compute_dtype(dtype)

    airfoil_mask = build_airfoil_mask(num_pos, idcs_airfoil, device, compute_dtype)
    if idcs_airfoil.numel() == 0:
        zeros = torch.zeros((num_pos, 1), device=device, dtype=compute_dtype)
        normals = torch.zeros((num_pos, 3), device=device, dtype=compute_dtype)
        normals[:, 2] = 1.0
        return torch.cat((zeros, zeros, airfoil_mask, normals, zeros), dim=-1).to(
            dtype=dtype
        )

    surface_pos = pos[idcs_airfoil.long()].to(dtype=compute_dtype)
    surface_pos = sample_surface_points(
        surface_pos, max_surface_points=max_surface_points
    )

    surface_normals = compute_surface_normals(
        surface_pos,
        k_neighbors=normal_k_neighbors,
    )

    dist_to_surface, nearest_idcs = nearest_surface_distance_and_index(
        pos,
        surface_pos,
        chunk_size=chunk_size,
    )

    nearest_surface_pos = surface_pos[nearest_idcs]
    nearest_normals = surface_normals[nearest_idcs]

    disp = pos - nearest_surface_pos
    disp_norm = disp / (disp.norm(dim=-1, keepdim=True) + 1e-8)
    cos_angle = (disp_norm * nearest_normals).sum(dim=-1, keepdim=True)

    adjacency_feature = (dist_to_surface <= adjacency_radius).to(dtype)
    adjacency_feature = torch.maximum(adjacency_feature, airfoil_mask)

    # [distance, adjacency, airfoil_mask, nearest_normal_xyz, cos(angle)]
    return torch.cat(
        (dist_to_surface, adjacency_feature, airfoil_mask, nearest_normals, cos_angle),
        dim=-1,
    ).to(dtype=dtype)
