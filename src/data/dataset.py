from __future__ import annotations

import hashlib
import math
import os
import pickle
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset, Sampler


def build_full_knn_indices(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Build exact kNN indices on CPU for a fixed point cloud."""
    num_pos = pos.size(0)
    k_eff = min(k, max(1, num_pos - 1))
    tree = cKDTree(pos.numpy())
    _, indices_np = tree.query(pos.numpy(), k=k_eff + 1)
    indices_np = np.asarray(indices_np)
    if indices_np.ndim == 1:
        indices_np = indices_np[:, None]
    indices = torch.from_numpy(indices_np[:, 1 : k_eff + 1]).long()
    if k_eff < k:
        pad = indices[:, -1:].expand(-1, k - k_eff)
        indices = torch.cat((indices, pad), dim=-1)
    return indices


def remap_knn_indices(
    full_knn_indices: torch.Tensor,
    point_indices: torch.Tensor,
    *,
    num_pos_original: int,
) -> torch.Tensor:
    mapping = torch.full((num_pos_original,), -1, dtype=torch.long)
    mapping[point_indices] = torch.arange(point_indices.numel(), dtype=torch.long)
    remapped = mapping[full_knn_indices.index_select(0, point_indices).long()]
    self_indices = torch.arange(point_indices.numel(), dtype=torch.long).view(-1, 1)
    return torch.where(remapped >= 0, remapped, self_indices)


def build_dense_crop_indices(
    pos: torch.Tensor,
    target_num_pos: int,
    *,
    random_crop: bool,
    seed: int | None,
) -> torch.Tensor:
    num_pos = pos.size(0)
    if target_num_pos >= num_pos:
        return torch.arange(num_pos, dtype=torch.long)

    if random_crop:
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        center_idx = torch.randint(0, num_pos, (1,), generator=generator).item()
        center_pos = pos[center_idx]
        distances = (pos - center_pos).square().sum(dim=-1)
        idcs = torch.topk(
            distances,
            k=target_num_pos,
            largest=False,
        ).indices
        return idcs.sort().values

    idcs = torch.linspace(0, num_pos - 1, steps=target_num_pos)
    return idcs.round().to(dtype=torch.long)


def sample_evenly_spaced_indices(indices: torch.Tensor, target_count: int) -> torch.Tensor:
    if target_count <= 0 or indices.numel() == 0:
        return indices.new_empty((0,), dtype=torch.long)
    if target_count >= indices.numel():
        return indices

    sample_positions = torch.linspace(
        0,
        indices.numel() - 1,
        steps=target_count,
    ).round().long()
    return indices.index_select(0, sample_positions)


def build_boundary_biased_indices(
    pos: torch.Tensor,
    idcs_airfoil: torch.Tensor,
    target_num_pos: int,
    *,
    boundary_point_fraction: float,
    seed: int,
) -> torch.Tensor:
    """Select *target_num_pos* points with a guaranteed boundary-layer focus.

    ``boundary_point_fraction`` of the budget is reserved for the airfoil region,
    but it is split between:
    - a sparse subset of exact airfoil points so boundary geometry and no-slip
      anchors remain available to the model, and
    - an off-wall shell of points layered around the airfoil so the sampled cloud
      captures the boundary layer instead of collapsing onto the wall itself.

    The remainder is sampled uniformly at random from the remaining points.
    """
    num_pos = pos.size(0)
    if target_num_pos >= num_pos:
        return torch.arange(num_pos, dtype=torch.long)

    airfoil_indices = idcs_airfoil.long().unique(sorted=True)
    if airfoil_indices.numel() == 0:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(num_pos, generator=generator)
        return perm[:target_num_pos].sort().values

    boundary_budget = min(target_num_pos, max(1, int(target_num_pos * boundary_point_fraction)))

    # Keep a sparse set of true wall points so the sampled cloud still contains
    # explicit boundary anchors for the hard no-slip mask.
    surface_budget = min(airfoil_indices.numel(), max(1, boundary_budget // 4))
    shell_budget = max(0, boundary_budget - surface_budget)
    surface_indices = sample_evenly_spaced_indices(airfoil_indices, surface_budget)

    # Distance of every mesh point to the nearest airfoil surface point (KDTree: O(N log A)).
    airfoil_pos = pos[airfoil_indices].numpy()
    tree = cKDTree(airfoil_pos)
    min_dist_np, nearest_airfoil_idx_np = tree.query(pos.numpy(), k=1)

    # Build an off-wall shell by excluding exact airfoil points, grouping volume
    # points by their nearest airfoil anchor, then interleaving layers across the
    # surface. This avoids spending the whole budget on a tiny set of wall-adjacent
    # points in one dense region.
    shell_mask = torch.ones(num_pos, dtype=torch.bool)
    shell_mask[airfoil_indices] = False
    shell_candidate_ids = shell_mask.nonzero(as_tuple=False).squeeze(1).cpu().numpy()
    shell_candidate_dist = min_dist_np[shell_candidate_ids]
    shell_candidate_group = nearest_airfoil_idx_np[shell_candidate_ids]

    positive_distance_mask = shell_candidate_dist > 0.0
    shell_candidate_ids = shell_candidate_ids[positive_distance_mask]
    shell_candidate_dist = shell_candidate_dist[positive_distance_mask]
    shell_candidate_group = shell_candidate_group[positive_distance_mask]

    if shell_budget > 0 and shell_candidate_ids.size > 0:
        group_then_dist_order = np.lexsort((shell_candidate_dist, shell_candidate_group))
        shell_candidate_ids = shell_candidate_ids[group_then_dist_order]
        shell_candidate_dist = shell_candidate_dist[group_then_dist_order]
        shell_candidate_group = shell_candidate_group[group_then_dist_order]

        _, group_starts, group_counts = np.unique(
            shell_candidate_group,
            return_index=True,
            return_counts=True,
        )
        rank_in_group = np.arange(shell_candidate_ids.size) - np.repeat(group_starts, group_counts)
        layer_then_dist_order = np.lexsort((shell_candidate_dist, rank_in_group))
        shell_indices_np = shell_candidate_ids[layer_then_dist_order[:shell_budget]]
        shell_indices = torch.from_numpy(shell_indices_np).long()
    else:
        shell_indices = airfoil_indices.new_empty((0,), dtype=torch.long)

    selected_boundary = torch.cat((surface_indices, shell_indices))[:boundary_budget]
    freestream_budget = target_num_pos - selected_boundary.numel()

    # Sample the remaining budget uniformly from the points not already reserved.
    mask = torch.ones(num_pos, dtype=torch.bool)
    mask[selected_boundary] = False
    remaining_pool = mask.nonzero(as_tuple=False).squeeze(1)

    generator = torch.Generator().manual_seed(seed)
    if freestream_budget >= remaining_pool.numel():
        far_indices = remaining_pool
    else:
        perm = torch.randperm(remaining_pool.numel(), generator=generator)
        far_indices = remaining_pool[perm[:freestream_budget]]

    combined = torch.cat([selected_boundary, far_indices])
    return combined.sort().values


def compute_surface_frame(
    pos: torch.Tensor,
    idcs_airfoil: torch.Tensor,
) -> torch.Tensor:
    """Compute a local surface coordinate frame for every mesh point.

    Returns an ``(N, 9)`` tensor of ``[n_x, n_y, n_z, t1_x, t1_y, t1_z, t2_x, t2_y, t2_z]``
    where *n* is the wall-normal direction (pointing away from the surface),
    and *t1*, *t2* are two orthogonal wall-tangent vectors.

    For points away from the airfoil the normal is estimated as the direction
    from the nearest surface point to the field point.  For airfoil points
    themselves (distance ≈ 0) the normal is estimated from a local PCA of nearby
    surface points so it is always well-defined.
    """
    num_pos = pos.size(0)
    frame = torch.zeros((num_pos, 9), dtype=pos.dtype)

    if idcs_airfoil.numel() == 0:
        # No airfoil → all normals default to z-up, tangents to x/y.
        frame[:, 2] = 1.0   # n = (0,0,1)
        frame[:, 3] = 1.0   # t1 = (1,0,0)
        frame[:, 7] = 1.0   # t2 = (0,1,0)
        return frame

    airfoil_pos_np = pos[idcs_airfoil.long()].numpy()
    tree = cKDTree(airfoil_pos_np)
    min_dist_np, nearest_idx_np = tree.query(pos.numpy(), k=1)
    nearest_airfoil_pos = torch.from_numpy(airfoil_pos_np[nearest_idx_np]).float()

    # --- Normal vectors ---
    diff = pos - nearest_airfoil_pos  # (N, 3)
    dist = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp(min=1e-12)
    normals = diff / dist  # (N, 3)

    # For on-surface points (dist ≈ 0), estimate normal from PCA of local
    # k nearest surface neighbours.
    on_surface_mask = (min_dist_np < 1e-8)
    on_surface_ids = np.where(on_surface_mask)[0]

    if on_surface_ids.size > 0:
        k_pca = min(8, airfoil_pos_np.shape[0])
        _, pca_nbr_idx = tree.query(pos.numpy()[on_surface_ids], k=k_pca)
        pca_nbr_idx = np.atleast_2d(pca_nbr_idx)
        for i, pt_idx in enumerate(on_surface_ids):
            nbr_pts = torch.from_numpy(airfoil_pos_np[pca_nbr_idx[i]]).float()
            centroid = nbr_pts.mean(dim=0, keepdim=True)
            centered = nbr_pts - centroid
            cov = centered.T @ centered
            eigvals, eigvecs = torch.linalg.eigh(cov)
            # Smallest eigenvalue eigenvector → surface normal direction
            normals[pt_idx] = eigvecs[:, 0]

    # Ensure unit normals
    normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True).clamp(min=1e-12)

    # --- Tangent vectors via Gram-Schmidt with a reference ---
    # Choose a reference vector that is not parallel to the normal.
    ref = torch.zeros_like(normals)
    ref[:, 0] = 1.0  # default reference = x-axis
    # Where normal is almost parallel to x-axis, switch to y-axis.
    parallel_mask = normals[:, 0].abs() > 0.9
    ref[parallel_mask, 0] = 0.0
    ref[parallel_mask, 1] = 1.0

    t1 = torch.cross(normals, ref, dim=-1)
    t1 = t1 / torch.linalg.norm(t1, dim=-1, keepdim=True).clamp(min=1e-12)
    t2 = torch.cross(normals, t1, dim=-1)
    t2 = t2 / torch.linalg.norm(t2, dim=-1, keepdim=True).clamp(min=1e-12)

    frame[:, 0:3] = normals
    frame[:, 3:6] = t1
    frame[:, 6:9] = t2
    return frame


def remap_airfoil_indices(
    idcs_airfoil: torch.Tensor,
    point_indices: torch.Tensor,
    *,
    num_pos_original: int,
) -> torch.Tensor:
    mapping = torch.full((num_pos_original,), -1, dtype=torch.long)
    mapping[point_indices] = torch.arange(point_indices.numel(), dtype=torch.long)
    idcs_remapped = mapping[idcs_airfoil.long()]
    return idcs_remapped[idcs_remapped >= 0]


def compute_velocity_standardization(
    velocity_in: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute normalization stats from known inputs only to avoid target leakage.
    mean = velocity_in.mean(dim=(0, 1), keepdim=True)
    std = velocity_in.std(dim=(0, 1), unbiased=False, keepdim=True).clamp_min(eps)
    return mean, std


def scale_velocity(
    velocity: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (velocity - mean) / std


def load_pressure_inputs_from_npz(
    data: Any,
    *,
    t_in: int,
    num_points: int,
) -> torch.Tensor:
    """
    Load per-node pressure for the ``t_in`` conditioning frames.

    Accepts ``pressure_in`` (``(t_in, N)``) or full ``pressure`` (``(T, N)`` with
    ``T >= t_in``), optional last dimension ``(..., 1)`` which is squeezed.
    Returns zeros when no pressure key exists (model may still use zero placeholders).
    """
    if "pressure_in" in data:
        arr = np.asarray(data["pressure_in"], dtype=np.float32)
    elif "pressure" in data:
        arr = np.asarray(data["pressure"], dtype=np.float32)
    else:
        return torch.zeros(t_in, num_points, dtype=torch.float32)

    if arr.ndim == 3:
        arr = arr.squeeze(-1)
    if arr.ndim != 2:
        raise ValueError(
            "pressure_in / pressure must be 2D (T, N) or 3D (T, N, 1); "
            f"got shape {arr.shape}."
        )
    if int(arr.shape[1]) != int(num_points):
        raise ValueError(
            f"pressure node count {arr.shape[1]} does not match pos N={num_points}."
        )
    if int(arr.shape[0]) < int(t_in):
        raise ValueError(
            f"pressure has T={arr.shape[0]} but velocity_in requires t_in={t_in}."
        )
    arr = arr[:t_in]
    return torch.from_numpy(np.copy(arr))


def unscale_velocity_batch(
    velocity: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    mean = mean.unsqueeze(1).unsqueeze(1)
    std = std.unsqueeze(1).unsqueeze(1)
    return velocity * std + mean


def augment_xy_plane_isometry(
    pos: torch.Tensor,
    velocity_in: torch.Tensor,
    velocity_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Random x-flip (50%) and random z-axis rotation on pos and velocity vectors.
    Apply to *raw* coordinates/velocities before per-sample standardization.
    """
    dev, dt = pos.device, pos.dtype
    flip = pos.new_tensor([-1.0, 1.0, 1.0], device=dev, dtype=dt)
    if random.random() < 0.5:
        pos = pos * flip
        velocity_in = velocity_in * flip
        velocity_out = velocity_out * flip
    theta = random.uniform(0.0, 2.0 * math.pi)
    c = math.cos(theta)
    s = math.sin(theta)
    r = pos.new_tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dt)
    pos = pos @ r.T
    velocity_in = velocity_in @ r.T
    velocity_out = velocity_out @ r.T
    return pos, velocity_in, velocity_out


class WarpedIFWDataset(Dataset):
    def __init__(
        self,
        files: Sequence[Path],
        *,
        num_points: int,
        random_crop: bool,
        seed: int,
        scaler_eps: float,
        boundary_point_fraction: float = 0.0,
        precompute_knn: bool = False,
        knn_cache_dir: str | Path | None = None,
        knn_k: int = 16,
        knn_include_neighbors: bool = False,
        knn_overflow_points: int = 0,
        augment_xy_plane_isometry: bool = False,
    ):
        self.files = list(files)
        self.num_points = num_points
        self.random_crop = random_crop
        self.seed = seed
        self.scaler_eps = scaler_eps
        self.boundary_point_fraction = boundary_point_fraction
        self.precompute_knn = precompute_knn
        self.knn_cache_dir = Path(knn_cache_dir) if knn_cache_dir else None
        self.knn_k = knn_k
        self.knn_include_neighbors = knn_include_neighbors
        self.knn_overflow_points = max(0, knn_overflow_points)
        self.augment_xy_plane_isometry = augment_xy_plane_isometry
        self._cached_point_indices: dict[int, torch.Tensor] = {}
        self._cached_knn_indices: dict[int, torch.Tensor] = {}

        if self.precompute_knn and self.knn_cache_dir is None:
            raise ValueError("knn_cache_dir is required when precompute_knn=True.")
        if self.knn_cache_dir is not None:
            self.knn_cache_dir.mkdir(parents=True, exist_ok=True)

        # Precompute boundary-biased indices for all samples so __getitem__ is fast.
        if boundary_point_fraction > 0.0:
            total = len(self.files)
            print(f"Precomputing boundary-biased indices for {total} samples ...")
            for idx, f in enumerate(self.files):
                with np.load(f) as d:
                    pos_np = d["pos"]
                    idcs_np = d["idcs_airfoil"]
                pos_t = torch.from_numpy(pos_np).float()
                idcs_t = torch.from_numpy(idcs_np).long()
                if num_points < pos_t.size(0):
                    self._cached_point_indices[idx] = build_boundary_biased_indices(
                        pos_t, idcs_t, num_points,
                        boundary_point_fraction=boundary_point_fraction,
                        seed=seed + idx,
                    )
                if (idx + 1) % 100 == 0 or idx + 1 == total:
                    print(f"  [{idx + 1}/{total}]")
            print(f"  -> cached {len(self._cached_point_indices)} index sets")

        if self.precompute_knn:
            total = len(self.files)
            print(f"Precomputing/loading kNN lookup tables for {total} samples (k={self.knn_k}) ...")
            for idx, path in enumerate(self.files):
                self._cached_knn_indices[idx] = self._load_or_build_knn_indices(path)
                if (idx + 1) % 100 == 0 or idx + 1 == total:
                    print(f"  [{idx + 1}/{total}]")
            print(f"  -> cached {len(self._cached_knn_indices)} kNN index sets")

    def _knn_cache_path(self, path: Path) -> Path:
        assert self.knn_cache_dir is not None
        resolved = str(path.resolve())
        digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]
        return self.knn_cache_dir / f"{path.stem}_{digest}_k{self.knn_k}.pt"

    def _load_or_build_knn_indices(self, path: Path) -> torch.Tensor:
        cache_path = self._knn_cache_path(path)
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
                if (
                    isinstance(payload, dict)
                    and int(payload.get("k", -1)) == self.knn_k
                    and "knn_indices" in payload
                ):
                    return payload["knn_indices"]
            except (EOFError, RuntimeError, OSError, ValueError, pickle.UnpicklingError):
                print(f"Invalid kNN cache file, rebuilding: {cache_path}")
                try:
                    cache_path.unlink()
                except FileNotFoundError:
                    pass

        with np.load(path) as data:
            pos = torch.from_numpy(data["pos"]).float()
        knn_indices = build_full_knn_indices(pos, self.knn_k)
        knn_indices = knn_indices.to(torch.int32)
        payload = {
            "k": self.knn_k,
            "num_points": int(pos.size(0)),
            "source": str(path.resolve()),
            "knn_indices": knn_indices,
        }
        tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return knn_indices

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        with np.load(path) as data:
            t = torch.from_numpy(data["t"]).float()
            pos = torch.from_numpy(data["pos"]).float()
            idcs_airfoil = torch.from_numpy(data["idcs_airfoil"]).long()
            velocity_in = torch.from_numpy(data["velocity_in"]).float()
            velocity_out = torch.from_numpy(data["velocity_out"]).float()
            pressure_in = load_pressure_inputs_from_npz(
                data,
                t_in=int(velocity_in.shape[0]),
                num_points=int(pos.size(0)),
            )

        num_pos_original = pos.size(0)
        if index in self._cached_point_indices:
            point_indices = self._cached_point_indices[index]
        elif self.boundary_point_fraction > 0.0 and self.num_points < num_pos_original:
            point_indices = build_boundary_biased_indices(
                pos,
                idcs_airfoil,
                self.num_points,
                boundary_point_fraction=self.boundary_point_fraction,
                seed=self.seed + index,
            )
            self._cached_point_indices[index] = point_indices
        else:
            point_indices = build_dense_crop_indices(
                pos,
                self.num_points,
                random_crop=self.random_crop,
                seed=None if self.random_crop else self.seed + index,
            )
        full_knn_indices = self._cached_knn_indices.get(index)
        if (
            full_knn_indices is not None
            and self.knn_include_neighbors
            and point_indices.numel() < num_pos_original
        ):
            crop_mask = torch.zeros(num_pos_original, dtype=torch.bool)
            crop_mask[point_indices] = True
            neighbor_indices = full_knn_indices.index_select(0, point_indices).reshape(-1).long()
            overflow_indices = neighbor_indices[~crop_mask.index_select(0, neighbor_indices)]
            overflow_indices = overflow_indices.unique(sorted=True)

            if self.knn_overflow_points > 0:
                overflow_indices = overflow_indices[: self.knn_overflow_points]
                if overflow_indices.numel() < self.knn_overflow_points:
                    available_indices = torch.arange(num_pos_original, dtype=torch.long)
                    available_indices = available_indices[~crop_mask]
                    available_indices = available_indices[
                        ~torch.isin(available_indices, overflow_indices)
                    ]
                    fill_count = min(
                        self.knn_overflow_points - overflow_indices.numel(),
                        available_indices.numel(),
                    )
                    if fill_count > 0:
                        overflow_indices = torch.cat(
                            (overflow_indices, available_indices[:fill_count])
                        )

            point_indices = torch.cat((point_indices, overflow_indices)).unique(sorted=True)

        pos = pos.index_select(0, point_indices)
        velocity_in = velocity_in.index_select(1, point_indices)
        velocity_out = velocity_out.index_select(1, point_indices)
        pressure_in = pressure_in.index_select(1, point_indices)
        idcs_airfoil = remap_airfoil_indices(
            idcs_airfoil,
            point_indices,
            num_pos_original=num_pos_original,
        )

        if self.augment_xy_plane_isometry:
            pos, velocity_in, velocity_out = augment_xy_plane_isometry(
                pos, velocity_in, velocity_out
            )

        velocity_mean, velocity_std = compute_velocity_standardization(
            velocity_in,
            eps=self.scaler_eps,
        )
        velocity_in = scale_velocity(velocity_in, velocity_mean, velocity_std)
        velocity_out = scale_velocity(velocity_out, velocity_mean, velocity_std)

        pm = pressure_in.mean()
        ps = pressure_in.std(unbiased=False).clamp_min(self.scaler_eps)
        pressure_in = (pressure_in - pm) / ps

        # Precompute raw wall distance (min Euclidean distance to airfoil)
        # so training avoids redundant cdist calls in model + loss.
        # Uses cKDTree for O(N log A) instead of O(N*A) brute-force cdist.
        if idcs_airfoil.numel() > 0:
            airfoil_pos_np = pos[idcs_airfoil.long()].numpy()
            tree = cKDTree(airfoil_pos_np)
            min_dist_np, _ = tree.query(pos.numpy(), k=1)
            wall_distance = torch.from_numpy(min_dist_np).float()
        else:
            wall_distance = torch.zeros(pos.size(0), dtype=pos.dtype)

        # Precompute local surface coordinate frame (normal + 2 tangents).
        surface_frame = compute_surface_frame(pos, idcs_airfoil)
        if full_knn_indices is not None:
            knn_indices = remap_knn_indices(
                full_knn_indices,
                point_indices,
                num_pos_original=num_pos_original,
            )
        else:
            knn_indices = torch.empty((0, 0), dtype=torch.long)

        return (
            t,
            pos,
            idcs_airfoil,
            velocity_in,
            velocity_out,
            pressure_in,
            velocity_mean.squeeze(0).squeeze(0),
            velocity_std.squeeze(0).squeeze(0),
            wall_distance,
            surface_frame,
            knn_indices,
        )


def collate_fn(batch):
    (
        t,
        pos,
        idcs_airfoil,
        velocity_in,
        velocity_out,
        pressure_in,
        velocity_mean,
        velocity_std,
        wall_distance,
        surface_frame,
        knn_indices,
    ) = zip(*batch)
    has_knn = knn_indices[0].numel() > 0
    return (
        torch.stack(t, dim=0),
        torch.stack(pos, dim=0),
        list(idcs_airfoil),
        torch.stack(velocity_in, dim=0),
        torch.stack(velocity_out, dim=0),
        torch.stack(pressure_in, dim=0),
        torch.stack(velocity_mean, dim=0),
        torch.stack(velocity_std, dim=0),
        torch.stack(wall_distance, dim=0),
        torch.stack(surface_frame, dim=0),
        torch.stack(knn_indices, dim=0) if has_knn else None,
    )


def split_train_val_test(
    files: Sequence[Path],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    if val_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("val_fraction and test_fraction must be > 0.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.")

    grouped: dict[str, list[Path]] = {}
    for path in files:
        group_key = path.stem.rsplit("-", maxsplit=1)[0]
        grouped.setdefault(group_key, []).append(path)

    group_keys = list(grouped.keys())
    if len(group_keys) <= 2:
        only_split = sorted(files)
        return only_split, only_split, only_split

    rng = random.Random(seed)
    rng.shuffle(group_keys)

    num_groups = len(group_keys)
    num_val_groups = int(round(num_groups * val_fraction))
    num_val_groups = max(1, min(num_val_groups, num_groups - 2))

    remaining_for_test = num_groups - num_val_groups
    num_test_groups = int(round(num_groups * test_fraction))
    num_test_groups = max(1, min(num_test_groups, remaining_for_test - 1))

    val_keys = set(group_keys[:num_val_groups])
    test_keys = set(group_keys[num_val_groups : num_val_groups + num_test_groups])
    train_files: list[Path] = []
    val_files: list[Path] = []
    test_files: list[Path] = []
    for key in group_keys:
        if key in val_keys:
            val_files.extend(grouped[key])
        elif key in test_keys:
            test_files.extend(grouped[key])
        else:
            train_files.extend(grouped[key])

    return sorted(train_files), sorted(val_files), sorted(test_files)


def split_train_val_kfold(
    files: Sequence[Path],
    *,
    num_folds: int,
    fold_index: int,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Group-level k-fold split (same grouping rule as :func:`split_train_val_test`).

    Groups are shuffled with ``random.Random(seed)``, then assigned to folds in
    round-robin order so sizes stay balanced. Validation is the ``fold_index`` fold;
    training uses all other folds. Returns ``test_files`` as an empty list (no
    held-out test in pure k-fold; use validation metrics per fold).

    Run eight separate trainings with ``fold_index=0..7`` and ``num_folds=8`` for
    8-way cross-validation-style coverage on 8 GPUs (sequential or parallel jobs).
    """
    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {num_folds}.")
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError(
            f"fold_index must be in [0, {num_folds}), got {fold_index}."
        )

    grouped: dict[str, list[Path]] = {}
    for path in files:
        group_key = path.stem.rsplit("-", maxsplit=1)[0]
        grouped.setdefault(group_key, []).append(path)

    group_keys = list(grouped.keys())
    if len(group_keys) <= num_folds:
        raise ValueError(
            f"k-fold needs more trajectory groups than folds: "
            f"got {len(group_keys)} groups and num_folds={num_folds}."
        )

    rng = random.Random(seed)
    rng.shuffle(group_keys)

    fold_keys: list[set[str]] = [set() for _ in range(num_folds)]
    for i, key in enumerate(group_keys):
        fold_keys[i % num_folds].add(key)

    val_keys = fold_keys[fold_index]
    train_files: list[Path] = []
    val_files: list[Path] = []
    for key in group_keys:
        if key in val_keys:
            val_files.extend(grouped[key])
        else:
            train_files.extend(grouped[key])

    return sorted(train_files), sorted(val_files), []


def resolve_overfit_file(
    files: Sequence[Path],
    *,
    overfit_file: str,
    overfit_index: int,
) -> Path:
    if overfit_file:
        candidate = Path(overfit_file)
        if candidate.exists():
            return candidate.resolve()

        matches = [
            path
            for path in files
            if path.name == overfit_file or path.stem == overfit_file
        ]
        if not matches:
            raise FileNotFoundError(
                f"Could not resolve --overfit-file={overfit_file} in dataset list."
            )
        return matches[0]

    if overfit_index < 0 or overfit_index >= len(files):
        raise IndexError(
            f"--overfit-index must be in [0, {len(files) - 1}], got {overfit_index}."
        )
    return files[overfit_index]


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    sampler: Sampler | None = None,
) -> DataLoader:
    loader_shuffle = shuffle if sampler is None else False
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=loader_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    return DataLoader(**loader_kwargs)
