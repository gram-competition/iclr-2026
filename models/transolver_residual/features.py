"""
Per-point feature computation for TransolverResidual.

Base feature vector (52 channels):
    pos_normalized   ( 3)  -- position in [0,1]^3  (per-sample bounding box)
    velocity_in      (15)  -- 5 input snapshots flattened (5 × 3)
    poly_residual    (15)  -- velocity_in minus polynomial fit on input window
    temporal_mean    ( 3)  -- mean velocity over 5 input timesteps per component
    temporal_std     ( 3)  -- std  velocity over 5 input timesteps per component
    is_airfoil       ( 1)  -- 1 if point is on airfoil surface, 0 otherwise
    dist_to_airfoil  ( 1)  -- distance to nearest airfoil surface point
    upstream_dist    ( 1)  -- signed x-offset to nearest surface point
                             (positive = downstream, negative = upstream)
    t_values         (10)  -- all 10 time values broadcast to every point
                             (global temporal context)
    velocity_differences (12) -- Δv_t = v_t - v_{t-1} for t=1..4
                          
    local_features   (15)  -- mean velocity of the k=8 nearest spatial neighbours.
                          
"""

import torch
import numpy as np
from models.transolver_residual.polynomial import poly_fit_residual

# Number of surface points subsampled for distance computation.
# The full surface has ~20k points; 2048 gives a very good approximation
# while keeping the (N, K) distance matrix small.
_N_SURF_SAMPLE = 2048
_DIST_CHUNK    = 5000   # points processed at once in distance loop

KNN_K = 8               # number of nearest neighbours for local feature


def get_feature_dim(
    use_local_feats: bool = False,
    use_temporal_deltas: bool = False,
) -> int:
    """Return the feature vector length for the given flag combination."""
    dim = 52
    if use_local_feats:
        dim += 15   # neighbour mean velocity: 5 timesteps × 3 components
    if use_temporal_deltas:
        dim += 12   # velocity differences: 4 Δt × 3 components
    return dim


def precompute_distance_features(
    pos: np.ndarray,
    idcs_airfoil: np.ndarray,
) -> tuple:
    """
    Precompute distance-based features for a single sample.  Runs on CPU/numpy
    so it can be called from DataLoader worker processes.

    Args:
        pos          : (N, 3)  float32 numpy array
        idcs_airfoil : (K,)    int64  numpy array

    Returns:
        is_airfoil      : (N,)  float32 -- binary surface flag
        dist_to_airfoil : (N,)  float32 -- Euclidean distance to nearest surface pt
        upstream_dist   : (N,)  float32 -- signed x-offset to nearest surface pt
    """
    N = len(pos)
    is_airfoil_arr      = np.zeros(N, dtype=np.float32)
    dist_to_airfoil_arr = np.zeros(N, dtype=np.float32)
    upstream_dist_arr   = np.zeros(N, dtype=np.float32)

    is_airfoil_arr[idcs_airfoil] = 1.0

    # Subsample surface points
    K = len(idcs_airfoil)
    if K > _N_SURF_SAMPLE:
        sub_idx = np.random.choice(K, _N_SURF_SAMPLE, replace=False)
        surf_pts = pos[idcs_airfoil[sub_idx]]          # (_N_SURF_SAMPLE, 3)
    else:
        surf_pts = pos[idcs_airfoil]                   # (K, 3)

    # Chunked nearest-surface search
    for i in range(0, N, _DIST_CHUNK):
        chunk = pos[i : i + _DIST_CHUNK]               # (C, 3)
        diffs = chunk[:, None, :] - surf_pts[None, :, :]  # (C, K, 3)
        dists = np.linalg.norm(diffs, axis=-1)         # (C, K)
        idx_min = dists.argmin(axis=-1)                # (C,)
        c = np.arange(len(chunk))
        dist_to_airfoil_arr[i : i + _DIST_CHUNK] = dists[c, idx_min]
        upstream_dist_arr[i : i + _DIST_CHUNK]   = diffs[c, idx_min, 0]

    return is_airfoil_arr, dist_to_airfoil_arr, upstream_dist_arr


def precompute_knn(pos: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """
    Compute k nearest spatial neighbours for every point using cKDTree.
    Self is excluded from the result.

    Args:
        pos : (N, 3)  float32 numpy array
        k   : number of neighbours (default KNN_K=8)

    Returns:
        knn_idx : (N, k)  int32 -- indices of k nearest neighbours per point
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k + 1)   # k+1: first result is the point itself
    return idx[:, 1:].astype(np.int32)  # (N, k), drop self


def compute_features(
    pos: torch.Tensor,
    velocity_in: torch.Tensor,
    idcs_airfoil: list,
    t: torch.Tensor,
    poly_degree: int = 2,
    dist_feats: list = None,
    use_local_feats: bool = False,
    use_temporal_deltas: bool = False,
    knn_feats: list = None,
) -> torch.Tensor:
    """
    Compute per-point input features for the Transolver encoder.

    Args:
        pos                  : (B, N, 3)
        velocity_in          : (B, 5, N, 3)
        idcs_airfoil         : list of B tensors, each of shape (K_b,)
        t                    : (B, 10)
        poly_degree          : degree used for polynomial fit residual
        dist_feats           : optional list of B tuples
                               (is_airfoil, dist_to_airfoil, upstream_dist) -- each (N,)
                               If provided, skips the nearest-surface computation.
        use_local_feats      : if True, append mean neighbour velocity (15 ch)
        use_temporal_deltas  : if True, append velocity differences (12 ch)
        knn_feats            : list of B tensors (N, k) int64, or None.
                               Required when use_local_feats=True.

    Returns:
        features : (B, N, D)  where D = get_feature_dim(use_local_feats, use_temporal_deltas)
    """
    B, T_in, N, _ = velocity_in.shape
    device = pos.device
    dtype  = velocity_in.dtype

    # ── 1. Normalized position ──────────────────────────────────────────────
    pos_min = pos.min(dim=1, keepdim=True).values   # (B, 1, 3)
    pos_max = pos.max(dim=1, keepdim=True).values   # (B, 1, 3)
    pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-8)  # (B, N, 3)

    # ── 2. Velocity input flattened ─────────────────────────────────────────
    vel_flat = velocity_in.permute(0, 2, 1, 3).reshape(B, N, T_in * 3)  # (B, N, 15)

    # ── 3. Polynomial residual on the input window ──────────────────────────
    poly_res = poly_fit_residual(velocity_in, t, degree=poly_degree)     # (B, 5, N, 3)
    poly_res_flat = poly_res.permute(0, 2, 1, 3).reshape(B, N, T_in * 3)  # (B, N, 15)

    # ── 4. Temporal statistics ───────────────────────────────────────────────
    temporal_mean = velocity_in.mean(dim=1)                              # (B, N, 3)
    temporal_std  = velocity_in.std(dim=1, unbiased=False)              # (B, N, 3)

    # ── 5. Airfoil flag and distance features ───────────────────────────────
    is_airfoil      = torch.zeros(B, N, 1, dtype=dtype, device=device)
    dist_to_airfoil = torch.zeros(B, N, 1, dtype=dtype, device=device)
    upstream_dist   = torch.zeros(B, N, 1, dtype=dtype, device=device)

    if dist_feats is not None:
        # Fast path: precomputed tensors from the Dataset
        for b in range(B):
            ia, dist, xsign = dist_feats[b]
            is_airfoil[b, :, 0]      = ia.to(device=device, dtype=dtype)
            dist_to_airfoil[b, :, 0] = dist.to(device=device, dtype=dtype)
            upstream_dist[b, :, 0]   = xsign.to(device=device, dtype=dtype)
    else:
        # Fallback: compute on-the-fly (slow -- avoid during training)
        for b in range(B):
            idcs = idcs_airfoil[b].to(device)
            is_airfoil[b, idcs, 0] = 1.0

            K_b = len(idcs)
            if K_b > _N_SURF_SAMPLE:
                sub = idcs[torch.randperm(K_b, device=device)[:_N_SURF_SAMPLE]]
            else:
                sub = idcs
            surf_pts = pos[b, sub]

            dist_buf  = torch.empty(N, dtype=dtype, device=device)
            xsign_buf = torch.empty(N, dtype=dtype, device=device)
            for i in range(0, N, _DIST_CHUNK):
                chunk = pos[b, i : i + _DIST_CHUNK]
                diffs = chunk.unsqueeze(1) - surf_pts.unsqueeze(0)
                dists = diffs.norm(dim=-1)
                d_min, idx_min = dists.min(dim=-1)
                dist_buf[i : i + _DIST_CHUNK]  = d_min
                xsign_buf[i : i + _DIST_CHUNK] = diffs[
                    torch.arange(len(chunk), device=device), idx_min, 0
                ]
            dist_to_airfoil[b, :, 0] = dist_buf
            upstream_dist[b, :, 0]   = xsign_buf

    # ── 6. Global time conditioning ─────────────────────────────────────────
    t_feat = t.unsqueeze(1).expand(B, N, 10)        # (B, N, 10)

    # ── 7. Base feature concatenation ───────────────────────────────────────
    parts = [
        pos_norm,           # (B, N,  3)
        vel_flat,           # (B, N, 15)
        poly_res_flat,      # (B, N, 15)
        temporal_mean,      # (B, N,  3)
        temporal_std,       # (B, N,  3)
        is_airfoil,         # (B, N,  1)
        dist_to_airfoil,    # (B, N,  1)
        upstream_dist,      # (B, N,  1)
        t_feat,             # (B, N, 10)
    ]

    # ── 8. Optional: local neighbour mean velocity (+15 ch) ─────────────────
    if use_local_feats:
        local_mean = torch.zeros(B, N, T_in * 3, dtype=dtype, device=device)
        if knn_feats is not None:
            for b in range(B):
                if knn_feats[b] is not None:
                    kidx = knn_feats[b].to(device=device, dtype=torch.long)  # (N, k)
                    # velocity_in[b]: (5, N, 3)
                    # index N dim with (N, k) → (5, N, k, 3)
                    nbr_vel = velocity_in[b][:, kidx, :]            # (5, N, k, 3)
                    nbr_mean = nbr_vel.mean(dim=2)                   # (5, N, 3)
                    local_mean[b] = nbr_mean.permute(1, 0, 2).reshape(N, T_in * 3)
        parts.append(local_mean)   # (B, N, 15)

    # ── 9. Optional: temporal velocity differences (+12 ch) ─────────────────
    if use_temporal_deltas:
        # Δv_t = v_t - v_{t-1}  for t = 1..4  →  (B, 4, N, 3)
        dv = velocity_in[:, 1:] - velocity_in[:, :-1]
        dv_flat = dv.permute(0, 2, 1, 3).reshape(B, N, (T_in - 1) * 3)  # (B, N, 12)
        parts.append(dv_flat)

    return torch.cat(parts, dim=-1)


# Base feature dimension with both optional flags off.
FEATURE_DIM = 52
