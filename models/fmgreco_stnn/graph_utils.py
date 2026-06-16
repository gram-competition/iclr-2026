"""k-NN graph construction for point clouds (SciPy on CPU for large N)."""

from __future__ import annotations

import torch

try:
    from scipy.spatial import cKDTree

    _HAS_CKDTREE = True
except Exception:
    cKDTree = None
    _HAS_CKDTREE = False


def knn_graph(
    pos: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        pos: (N, 3)
        k: neighbours (excluding self)

    Returns:
        neighbors: (N, k) int64
        rel_pos: (N, k, 3) neighbour - center
        dists: (N, k)

    Note:
        SciPy ``cKDTree`` runs on CPU: this allocates a float32 CPU copy of ``pos`` per
        call (and per rank under DDP). For ~100k points that is modest; on 8-GPU nodes,
        peak host RAM is roughly eight concurrent copies only if all ranks query at
        once—typically well within server-class memory, but worth monitoring on small VMs.
    """
    n = pos.size(0)
    if k >= n:
        raise ValueError(f"k ({k}) must be < N ({n})")

    if _HAS_CKDTREE:
        pos_cpu = pos.detach().to("cpu", dtype=torch.float32).contiguous()
        tree = cKDTree(pos_cpu.numpy())
        dists_np, nn_idx = tree.query(pos_cpu.numpy(), k=k + 1, workers=-1)
        if k == 1:
            dists_np = dists_np.reshape(-1, 1)
            nn_idx = nn_idx.reshape(-1, 1)
        nn_idx = nn_idx[:, 1:]
        dists_np = dists_np[:, 1:]
        nn_t = torch.as_tensor(nn_idx, device=pos.device, dtype=torch.long)
        dists_t = torch.as_tensor(dists_np, device=pos.device, dtype=pos.dtype)
        rel = pos[nn_t] - pos.unsqueeze(1)
        return nn_t, rel, dists_t

    # Fallback: O(N^2) — only viable for small clouds
    if n > 4096:
        raise RuntimeError(
            "Install scipy for k-NN on large point clouds (pip install scipy)."
        )
    pw = torch.cdist(pos, pos)
    _, nn_idx = pw.topk(k + 1, largest=False, dim=-1)
    nn_idx = nn_idx[:, 1:]
    rel = pos[nn_idx] - pos.unsqueeze(1)
    dists_t = rel.norm(dim=-1)
    return nn_idx, rel, dists_t


def _subsample_surface(surface_pos: torch.Tensor, max_points: int) -> torch.Tensor:
    n = surface_pos.size(0)
    if n <= max_points or max_points <= 0:
        return surface_pos
    idcs = (
        torch.linspace(0, n - 1, steps=max_points, device=surface_pos.device)
        .round()
        .long()
    )
    return surface_pos[idcs]


def _promote_compute_dtype(d: torch.dtype) -> torch.dtype:
    if d in (torch.float16, torch.bfloat16):
        return torch.float32
    return d


def airfoil_boundary_features(
    pos: torch.Tensor,
    idcs_airfoil: torch.Tensor,
    *,
    max_airfoil_samples: int = 4096,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """
    Per volume point: [log1p(d), d_hat] where d is Euclidean distance to the
    nearest *sampled* airfoil point, d_hat = (p - p_s) / (||d||+eps) in R^3.

    Returns (N, 4) with the **same dtype as** ``pos`` (internal ``cdist`` uses fp32 when
    ``pos`` is half/bfloat16, then results are cast back—see ``_promote_compute_dtype``).

    If there is no airfoil, returns zeros.
    """
    n = pos.size(0)
    device, dtype = pos.device, pos.dtype
    if idcs_airfoil is None or idcs_airfoil.numel() == 0:
        return pos.new_zeros((n, 4))

    out = pos.new_zeros((n, 9))
    idcs = idcs_airfoil.long().view(-1).clamp_(0, n - 1)
    surface = pos.index_select(0, idcs).detach()
    surface = _subsample_surface(surface, max_airfoil_samples)
    if surface.size(0) == 0:
        return out

    compute_dtype = _promote_compute_dtype(dtype)
    pos_w = pos.to(dtype=compute_dtype)
    surface_w = surface.to(dtype=compute_dtype)

    min_dist = torch.empty((n,), device=device, dtype=compute_dtype)
    min_idcs = torch.empty((n,), device=device, dtype=torch.long)

    with torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            dists = torch.cdist(pos_w[start:stop], surface_w)
            md, mid = dists.min(dim=1)
            min_dist[start:stop] = md
            min_idcs[start:stop] = mid

    nearest = surface_w[min_idcs]
    disp = pos_w - nearest
    disp = disp / (disp.norm(dim=-1, keepdim=True) + 1e-8)
    raw_dist = min_dist.unsqueeze(-1)
    nearest2 = surface_w[min_idcs]
    disp_raw = pos_w - nearest2
    extra = torch.cat([raw_dist, disp_raw, torch.log1p(min_dist).unsqueeze(-1) ** 2], dim=-1)
    out = torch.cat((torch.log1p(min_dist).unsqueeze(-1), disp, extra), dim=-1)
    return out.to(dtype=dtype)


def knn_flux_divergence(
    pos: torch.Tensor,
    u: torch.Tensor,
    neighbors: torch.Tensor,
    rel_pos: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Scalar divergence-like field from neighbor velocity differences (training prior).

    For each node i, uses k neighbors j with e_ij = x_j - x_i and
    u_ij = u_j - u_i (same shape (N,3)):

        d_i = (1/k) * sum_j (u_ij · e_ij) / (||e_ij||^2 + eps)

    This is a graph finite-difference proxy; validate on smooth fields before
    relying on the penalty. Returns (N,).
    """
    n, k, _ = rel_pos.shape
    if n == 0:
        return pos.new_zeros((0,))

    u_n = u[neighbors]
    u_c = u.unsqueeze(1)
    du = u_n - u_c
    denom = rel_pos.pow(2).sum(-1) + eps
    inner = (du * rel_pos).sum(-1) / denom
    return inner.mean(dim=-1)


def knn_flux_divergence_loss(
    pos: torch.Tensor,
    u: torch.Tensor,
    k: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Mean-squared `knn_flux_divergence` (penalize squared divergence field).

    ``u`` is (N, 3) on the same position cloud as ``pos``. Optional *training* term
    (validate the discrete operator on ground-truth fields first).
    """
    if u.size(0) <= 1 or k < 1:
        return u.sum() * 0.0
    k_eff = min(k, u.size(0) - 1)
    nbr, rel, _ = knn_graph(pos, k_eff)
    div = knn_flux_divergence(pos, u, nbr, rel, eps=eps)
    return div.pow(2).mean()
