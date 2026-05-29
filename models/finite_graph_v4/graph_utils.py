"""Directional kNN graph construction (numpy, inference copy).

Used at inference time to pick, for every non-surface centre node:
  - k isotropic nearest neighbours
  - k nearest upstream neighbours (displacement opposite to v_t4)
  - k nearest downstream neighbours (displacement aligned with v_t4)
from a larger precomputed kNN pool. The outputs drive the three directional
heads of FiniteGraphV4.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def build_knn_pool_np(pos: np.ndarray, k_pool: int = 256) -> np.ndarray:
    tree = cKDTree(pos)
    _, nbrs = tree.query(pos, k=k_pool + 1)
    return nbrs[:, 1:].astype(np.int64)


def build_directional_graphs_np(
    pos: np.ndarray,           # (N, 3) float32
    v_t4: np.ndarray,          # (N, 3) float32
    airfoil_mask: np.ndarray,  # (N,) bool
    nbrs_pool: np.ndarray,     # (N, K_pool) int64
    k: int = 16,
    chunk_size: int = 10000,
) -> dict[str, np.ndarray]:
    """Build iso / upstream / downstream neighbour index arrays."""
    N = pos.shape[0]
    K_pool = nbrs_pool.shape[1]
    speed = np.linalg.norm(v_t4, axis=1)

    candidate = (~airfoil_mask) & (K_pool >= k)
    iso_all = nbrs_pool[:, :k].copy()

    eps_speed = 1e-6
    stagnation = speed < eps_speed
    v_hat = np.zeros_like(v_t4)
    moving = ~stagnation
    v_hat[moving] = v_t4[moving] / speed[moving, None]

    up_all = np.empty((N, k), dtype=np.int64)
    down_all = np.empty((N, k), dtype=np.int64)
    up_valid = np.zeros(N, dtype=bool)
    down_valid = np.zeros(N, dtype=bool)
    SENTINEL = np.float32(1e30)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        sl = slice(start, end)
        n = end - start

        nb = nbrs_pool[sl]
        dp = pos[nb] - pos[sl, None, :]
        dist = np.linalg.norm(dp, axis=2)
        proj = np.einsum("nki,ni->nk", dp, v_hat[sl])

        rows = np.arange(n)[:, None]

        up_dist = np.where(proj < 0.0, dist, SENTINEL)
        up_order = np.argpartition(up_dist, k, axis=1)[:, :k]
        up_refsort = np.take_along_axis(up_dist, up_order, axis=1).argsort(axis=1)
        up_final = np.take_along_axis(up_order, up_refsort, axis=1)
        up_all[sl] = nb[rows, up_final]
        up_valid[sl] = np.take_along_axis(up_dist, up_final, axis=1)[:, -1] < SENTINEL

        down_dist = np.where(proj > 0.0, dist, SENTINEL)
        down_order = np.argpartition(down_dist, k, axis=1)[:, :k]
        down_refsort = np.take_along_axis(down_dist, down_order, axis=1).argsort(axis=1)
        down_final = np.take_along_axis(down_order, down_refsort, axis=1)
        down_all[sl] = nb[rows, down_final]
        down_valid[sl] = np.take_along_axis(down_dist, down_final, axis=1)[:, -1] < SENTINEL

    up_all[stagnation] = iso_all[stagnation]
    down_all[stagnation] = iso_all[stagnation]
    up_valid[stagnation] = True
    down_valid[stagnation] = True

    valid = candidate & up_valid & down_valid
    node_ids = np.where(valid)[0].astype(np.int64)
    return {
        "iso": iso_all[valid],
        "up": up_all[valid],
        "down": down_all[valid],
        "valid": valid,
        "node_ids": node_ids,
    }
