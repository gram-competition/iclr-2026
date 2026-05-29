"""Feature engineering for FiniteGraphV4 (competition-inference subset).

Assembles the (N, 26) input feature matrix expected by the network from a
single sample's ``pos`` / ``velocity_in`` / ``idcs_airfoil``. This is the
same 26-channel layout used by FiniteGraphV3 — v4 only changes the network
architecture, not the input features. The on-disk geometry cache is
removed so the package never writes into the host repository at
evaluation time.

Channel layout (must match the checkpoint that was trained on it):
     0..11  : 4 velocity deltas v[1..4] - v[0]
    12..14  : v at t4 (baseline for residual prediction)
    15      : wall distance
    16      : signed wall distance (sign from v_t4 . to_wall_hat)
    17..19  : normalised position
    20      : airfoil indicator (0/1)
    21..23  : vorticity at t4 (from kNN LSQ, log-compressed)
    24      : Q-criterion at t4 (log-compressed)
    25      : strain-rate magnitude at t4 (log-compressed)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


N_INPUT_CH = 26

DELTA_CH_START = 0
V_T4_CH_START  = 12
WALL_DIST_CH   = 15
SIGNED_WD_CH   = 16
POS_N_CH_START = 17
AIRFOIL_IND_CH = 20
VORT_CH_START  = 21
Q_CRIT_CH      = 24
STRAIN_MAG_CH  = 25

KNN_K = 16
DT = 0.001


def compute_wall_geometry(
    pos: np.ndarray, idcs_airfoil: np.ndarray,
) -> dict[str, np.ndarray]:
    """Wall distance, unit vector to nearest wall point, and a kNN graph."""
    surf_tree = cKDTree(pos[idcs_airfoil])
    d, nn_local = surf_tree.query(pos, k=1)
    nearest = pos[idcs_airfoil][nn_local]
    vec = nearest - pos
    d = d.astype(np.float32)
    norm = np.linalg.norm(vec, axis=1, keepdims=True).clip(min=1e-8)
    vec_hat = (vec / norm).astype(np.float32)

    full_tree = cKDTree(pos)
    _, nbrs = full_tree.query(pos, k=KNN_K + 1)
    nbrs = nbrs[:, 1:].astype(np.int64)
    return {"wall_dist": d, "to_wall_hat": vec_hat, "nbrs": nbrs}


def _gradient_tensor_np(
    v: np.ndarray, pos: np.ndarray, nbrs: np.ndarray, reg: float = 1e-3,
) -> np.ndarray:
    dx = pos[nbrs] - pos[:, None, :]
    dv = v[nbrs] - v[:, None, :]
    A = np.einsum("nki,nkj->nij", dx, dx)
    B = np.einsum("nki,nkj->nij", dx, dv)
    diag_mean = np.trace(A, axis1=1, axis2=2) / 3.0
    lam = (reg * np.maximum(diag_mean, 1e-8))[:, None, None]
    A_reg = A + np.eye(3, dtype=A.dtype)[None] * lam
    return np.linalg.solve(A_reg, B)


def _vorticity_from_G(G: np.ndarray, log_scale: float = 100.0) -> np.ndarray:
    wx = G[:, 1, 2] - G[:, 2, 1]
    wy = G[:, 2, 0] - G[:, 0, 2]
    wz = G[:, 0, 1] - G[:, 1, 0]
    vort = np.stack([wx, wy, wz], axis=1).astype(np.float32)
    return (np.sign(vort) * np.log1p(np.abs(vort) / log_scale)).astype(np.float32)


def _q_and_strain_from_G(
    G: np.ndarray,
    q_log_scale: float = 1e4,
    s_log_scale: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    Gt = G.swapaxes(-1, -2)
    S = 0.5 * (G + Gt)
    Om = 0.5 * (G - Gt)
    S_norm2 = (S ** 2).sum(axis=(1, 2))
    O_norm2 = (Om ** 2).sum(axis=(1, 2))
    Q = 0.5 * (O_norm2 - S_norm2)
    S_mag = np.sqrt(S_norm2)
    q_feat = (np.sign(Q) * np.log1p(np.abs(Q) / q_log_scale)).astype(np.float32)
    s_feat = np.log1p(S_mag / s_log_scale).astype(np.float32)
    return q_feat, s_feat


def _normalise_positions(pos: np.ndarray) -> np.ndarray:
    centre = pos.mean(axis=0, keepdims=True)
    p = pos - centre
    scale = float(np.abs(p).max()) or 1.0
    return (p / scale).astype(np.float32)


def build_features(
    pos: np.ndarray,               # (N, 3)
    velocity_in: np.ndarray,       # (5, N, 3)
    idcs_airfoil: np.ndarray,      # (M,)
) -> np.ndarray:
    """Return the (N, 26) float32 feature matrix for one sample."""
    geom = compute_wall_geometry(pos, idcs_airfoil)
    wall_dist = geom["wall_dist"]
    to_wall_hat = geom["to_wall_hat"]
    nbrs = geom["nbrs"]

    v0 = velocity_in[0]
    deltas = velocity_in[1:] - v0                                   # (4, N, 3)
    delta_feat = deltas.transpose(1, 0, 2).reshape(pos.shape[0], -1)

    v_t4 = velocity_in[4]

    sgn = np.einsum("ni,ni->n", v_t4, to_wall_hat)
    sgn = np.sign(sgn).astype(np.float32)
    signed_wall_dist = (sgn * wall_dist).astype(np.float32)

    airfoil_ind = np.zeros(pos.shape[0], dtype=np.float32)
    airfoil_ind[idcs_airfoil] = 1.0

    pos_n = _normalise_positions(pos)

    G = _gradient_tensor_np(
        v_t4.astype(np.float32), pos.astype(np.float32), nbrs,
    )
    vort = _vorticity_from_G(G)
    q_crit, strain_mag = _q_and_strain_from_G(G)

    feats = np.concatenate([
        delta_feat,                   # 12
        v_t4,                         #  3
        wall_dist[:, None],           #  1
        signed_wall_dist[:, None],    #  1
        pos_n,                        #  3
        airfoil_ind[:, None],         #  1
        vort,                         #  3
        q_crit[:, None],              #  1
        strain_mag[:, None],          #  1
    ], axis=1).astype(np.float32)

    assert feats.shape[1] == N_INPUT_CH, feats.shape
    return feats
