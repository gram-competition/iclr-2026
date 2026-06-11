
import logging

import torch
import torch.nn.functional as F

try:
    import numpy as np
    from scipy.spatial import cKDTree

    _HAS_CKDTREE = True
except Exception:
    np = None
    cKDTree = None
    _HAS_CKDTREE = False

_KNN_BACKEND_LOGGED = False
_SAMPLE_BACKEND_LOGGED = False
_INTERP_BACKEND_LOGGED = False


def _log_knn_backend_once(backend: str, pos: torch.Tensor, k: int) -> None:
    global _KNN_BACKEND_LOGGED
    if _KNN_BACKEND_LOGGED:
        return

    msg = (
        f"knn_graph backend: {backend} "
        f"(N={pos.size(0)}, k={k}, device={pos.device}, scipy={_HAS_CKDTREE})"
    )
    logger = logging.getLogger("GNN")
    if logger.handlers:
        logger.info(msg)
    else:
        print(f"[graph_utils] {msg}")
    _KNN_BACKEND_LOGGED = True


def _log_sample_backend_once(mode: str, pos: torch.Tensor, num_samples: int) -> None:
    global _SAMPLE_BACKEND_LOGGED
    if _SAMPLE_BACKEND_LOGGED:
        return

    msg = (
        f"subsample backend: {mode} "
        f"(N={pos.size(0)}, num_samples={num_samples}, device={pos.device})"
    )
    logger = logging.getLogger("GNN")
    if logger.handlers:
        logger.info(msg)
    else:
        print(f"[graph_utils] {msg}")
    _SAMPLE_BACKEND_LOGGED = True


def _log_interp_backend_once(
    backend: str,
    pos_sub: torch.Tensor,
    pos_full: torch.Tensor,
    k: int,
) -> None:
    global _INTERP_BACKEND_LOGGED
    if _INTERP_BACKEND_LOGGED:
        return

    msg = (
        f"knn_interpolate backend: {backend} "
        f"(N_sub={pos_sub.size(0)}, N_full={pos_full.size(0)}, "
        f"k={k}, device={pos_sub.device}, scipy={_HAS_CKDTREE})"
    )
    logger = logging.getLogger("GNN")
    if logger.handlers:
        logger.info(msg)
    else:
        print(f"[graph_utils] {msg}")
    _INTERP_BACKEND_LOGGED = True


# ---------------------------------------------------------------------------
# Fast spatial subsampling
# ---------------------------------------------------------------------------

def _voxel_grid_representatives(
    pos: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Pick one representative per occupied voxel.

    The voxel size is chosen so the number of occupied voxels is typically
    around ``num_samples`` for roughly uniform point clouds. Representatives are
    deterministic (first point after voxel-key sort).
    """
    n = pos.size(0)
    if num_samples >= n:
        return torch.arange(n, device=pos.device)

    mins = pos.min(dim=0).values
    maxs = pos.max(dim=0).values
    extent = (maxs - mins).clamp(min=1e-6)
    cell_size = (extent.prod() / max(num_samples, 1)) ** (1.0 / 3.0)
    cell_size = cell_size.clamp(min=1e-6)

    coords = torch.floor((pos - mins) / cell_size).to(torch.long)
    grid_shape = coords.max(dim=0).values + 1
    s1 = grid_shape[1] + 1
    s2 = grid_shape[2] + 1
    keys = coords[:, 0] * s1 * s2 + coords[:, 1] * s2 + coords[:, 2]

    order = torch.argsort(keys)
    keys_sorted = keys[order]
    first_mask = torch.ones_like(keys_sorted, dtype=torch.bool)
    first_mask[1:] = keys_sorted[1:] != keys_sorted[:-1]
    reps = order[first_mask]

    if reps.numel() > num_samples:
        pick = torch.linspace(
            0,
            reps.numel() - 1,
            steps=num_samples,
            device=pos.device,
        ).round().to(torch.long)
        reps = reps[pick]

    return reps


def _fill_to_budget(
    chosen: torch.Tensor,
    total_points: int,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    if chosen.numel() >= num_samples:
        return chosen[:num_samples]

    mask = torch.ones(total_points, dtype=torch.bool, device=device)
    mask[chosen] = False
    remaining = torch.nonzero(mask, as_tuple=False).flatten()
    need = num_samples - chosen.numel()
    if need >= remaining.numel():
        return torch.cat([chosen, remaining], dim=0)

    pick = torch.linspace(
        0,
        remaining.numel() - 1,
        steps=need,
        device=device,
    ).round().to(torch.long)
    return torch.cat([chosen, remaining[pick]], dim=0)


def farthest_point_sample(
    pos: torch.Tensor,
    num_samples: int,
    priority_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fast boundary-aware spatial subsampling.

    This keeps airfoil / boundary points first (when ``priority_idx`` is given),
    then uses a voxel-grid coverage heuristic on the remaining points. It is
    much cheaper than iterative FPS and better aligned with this CFD task where
    preserving boundary information matters.

    Args:
        pos: (N, 3) point positions.
        num_samples: how many points to select.
        priority_idx: indices that should be preserved first, e.g. airfoil points.

    Returns:
        (num_samples,) long tensor of selected indices.
    """
    N = pos.size(0)
    if num_samples >= N:
        return torch.arange(N, device=pos.device)
    _log_sample_backend_once("boundary-aware voxel-grid", pos, num_samples)

    if priority_idx is None or priority_idx.numel() == 0:
        chosen = _voxel_grid_representatives(pos, num_samples)
        return _fill_to_budget(chosen, N, num_samples, pos.device)

    priority_idx = priority_idx.to(device=pos.device, dtype=torch.long)
    priority_idx = priority_idx[(priority_idx >= 0) & (priority_idx < N)]
    priority_idx = torch.unique(priority_idx, sorted=False)

    if priority_idx.numel() >= num_samples:
        priority_pos = pos[priority_idx]
        chosen_local = _voxel_grid_representatives(priority_pos, num_samples)
        return priority_idx[chosen_local]

    keep = priority_idx
    mask = torch.ones(N, dtype=torch.bool, device=pos.device)
    mask[keep] = False
    remaining = torch.nonzero(mask, as_tuple=False).flatten()
    remaining_need = num_samples - keep.numel()
    remaining_local = _voxel_grid_representatives(pos[remaining], remaining_need)
    chosen = torch.cat([keep, remaining[remaining_local]], dim=0)
    return _fill_to_budget(chosen, N, num_samples, pos.device)


# ---------------------------------------------------------------------------
# k-NN graph (dense neighbour format)
# ---------------------------------------------------------------------------

def _knn_graph_ckdtree(
    pos: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact k-NN using SciPy's cKDTree on CPU."""
    pos_cpu = pos.detach().to("cpu", dtype=torch.float32).contiguous()
    pos_np = pos_cpu.numpy()
    tree = cKDTree(pos_np)
    dists, nn_idx = tree.query(pos_np, k=k + 1, workers=-1)

    if k == 1:
        dists = dists[:, None]
        nn_idx = nn_idx[:, None]

    nn_idx = nn_idx[:, 1:]
    dists = dists[:, 1:]

    nn_idx_t = torch.from_numpy(np.asarray(nn_idx)).to(device=pos.device, dtype=torch.long)
    dists_t = torch.from_numpy(np.asarray(dists)).to(device=pos.device, dtype=pos.dtype)
    rel_pos = pos[nn_idx_t] - pos.unsqueeze(1)
    return nn_idx_t, rel_pos, dists_t


def knn_graph(pos: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build k-NN graph in *dense* format.

    Args:
        pos: (N, 3)
        k: number of neighbours

    Returns:
        neighbors: (N, k)  indices of neighbours
        rel_pos:   (N, k, 3)  relative positions  (neighbour - center)
        dists:     (N, k)  Euclidean distances
    """
    if k >= pos.size(0):
        raise ValueError(f"k ({k}) must be < number of points ({pos.size(0)})")

    if _HAS_CKDTREE:
        _log_knn_backend_once("scipy.cKDTree", pos, k)
        return _knn_graph_ckdtree(pos, k)

    _log_knn_backend_once("torch.cdist", pos, k)
    pw = torch.cdist(pos, pos)  # (N, N)
    # k+1 because the closest point is itself
    _, nn_idx = pw.topk(k + 1, largest=False, dim=-1)
    nn_idx = nn_idx[:, 1:]  # drop self-loop  (N, k)

    rel_pos = pos[nn_idx] - pos.unsqueeze(1)  # (N, k, 3)
    dists = rel_pos.norm(dim=-1)  # (N, k)
    return nn_idx, rel_pos, dists


# ---------------------------------------------------------------------------
# k-NN interpolation (sub → full resolution)
# ---------------------------------------------------------------------------

def knn_interpolate(
    feat_sub: torch.Tensor,
    pos_sub: torch.Tensor,
    pos_full: torch.Tensor,
    k: int = 3,
    chunk_size: int = 10_000,
) -> torch.Tensor:
    """Inverse-distance–weighted interpolation via k nearest subsampled points.

    Args:
        feat_sub: (N_sub, D) or (T, N_sub, D)
        pos_sub:  (N_sub, 3)
        pos_full: (N, 3)
        k:        number of neighbours for interpolation
        chunk_size: process full points in chunks to limit memory

    Returns:
        feat_full: (N, D) or (T, N, D)
    """
    if k > pos_sub.size(0):
        raise ValueError(f"k ({k}) must be <= number of subsampled points ({pos_sub.size(0)})")

    has_time = feat_sub.dim() == 3

    if _HAS_CKDTREE:
        _log_interp_backend_once("scipy.cKDTree", pos_sub, pos_full, k)
        pos_sub_cpu = pos_sub.detach().to("cpu", dtype=torch.float32).contiguous()
        pos_full_cpu = pos_full.detach().to("cpu", dtype=torch.float32).contiguous()
        tree = cKDTree(pos_sub_cpu.numpy())
        nn_dist, nn_idx = tree.query(pos_full_cpu.numpy(), k=k, workers=-1)

        if k == 1:
            nn_dist = nn_dist[:, None]
            nn_idx = nn_idx[:, None]

        nn_idx_t = torch.from_numpy(np.asarray(nn_idx)).to(
            device=feat_sub.device, dtype=torch.long
        )
        nn_dist_t = torch.from_numpy(np.asarray(nn_dist)).to(
            device=feat_sub.device, dtype=feat_sub.dtype
        ).clamp(min=1e-8)

        w = 1.0 / nn_dist_t
        w = w / w.sum(dim=1, keepdim=True)

        if has_time:
            nn_feat = feat_sub[:, nn_idx_t, :]  # (T, N, k, D)
            return (w.unsqueeze(0).unsqueeze(-1) * nn_feat).sum(dim=2)

        nn_feat = feat_sub[nn_idx_t]  # (N, k, D)
        return (w.unsqueeze(-1) * nn_feat).sum(dim=1)

    _log_interp_backend_once("torch.cdist", pos_sub, pos_full, k)
    if has_time:
        T = feat_sub.size(0)
        results = [
            knn_interpolate(feat_sub[t], pos_sub, pos_full, k, chunk_size)
            for t in range(T)
        ]
        return torch.stack(results)

    N_full = pos_full.size(0)
    D = feat_sub.size(-1)
    parts: list[torch.Tensor] = []

    for start in range(0, N_full, chunk_size):
        end = min(start + chunk_size, N_full)
        chunk_pos = pos_full[start:end]  # (C, 3)

        pw = torch.cdist(chunk_pos, pos_sub)  # (C, N_sub)
        _, nn_idx = pw.topk(k, largest=False, dim=-1)  # (C, k)
        nn_dist = pw.gather(1, nn_idx).clamp(min=1e-8)  # (C, k)

        w = 1.0 / nn_dist  # (C, k)
        w = w / w.sum(dim=1, keepdim=True)  # normalise

        nn_feat = feat_sub[nn_idx]  # (C, k, D)
        interp = (w.unsqueeze(-1) * nn_feat).sum(dim=1)  # (C, D)
        parts.append(interp)

    return torch.cat(parts, dim=0)  # (N, D)


# ---------------------------------------------------------------------------
# Scatter helpers (used by GNN layers)
# ---------------------------------------------------------------------------

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum-aggregate *src* rows into *dim_size* buckets given by *index*."""
    shape = list(src.shape)
    shape[0] = dim_size
    out = src.new_zeros(shape)
    idx = index
    for _ in range(src.dim() - 1):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    s = scatter_sum(src, index, dim_size)
    cnt = scatter_sum(torch.ones_like(src[:, :1]), index, dim_size).clamp(min=1)
    return s / cnt


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Grouped softmax: softmax of *src* within groups defined by *index*."""
    mx = scatter_sum(src, index, dim_size)  # crude upper bound (sum ≥ max)
    # proper max via scatter_reduce
    mx = src.new_full((dim_size, *src.shape[1:]), float("-inf"))
    idx = index
    for _ in range(src.dim() - 1):
        idx = idx.unsqueeze(-1)
    idx_e = idx.expand_as(src)
    mx.scatter_reduce_(0, idx_e, src, reduce="amax", include_self=False)

    src = (src - mx[index]).exp()
    denom = scatter_sum(src, index, dim_size)
    return src / denom[index].clamp(min=1e-8)
