"""
Post-training evaluation and visualisation for TransolverResidual.

Usage:
    ./venv/bin/python utils/eval_vis.py                        # random val sample
    ./venv/bin/python utils/eval_vis.py --sim_id 1025_1        # specific sim
    ./venv/bin/python utils/eval_vis.py --n_samples 10         # eval N val samples
    ./venv/bin/python utils/eval_vis.py --save_dir out/        # save figures to disk
Produces:
  1. Per-timestep prediction vs ground-truth vs error  (CFD-style XZ slices)
  2. Relative L2 per timestep bar chart (model vs polynomial baseline)
  3. Error distribution violin plot across val samples
"""

import argparse
import os
import sys
import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from models.transolver_residual import TransolverResidual
from models.transolver_residual.features import precompute_distance_features
from models.transolver_residual.polynomial import poly_extrapolate
from utils.dataloader import GRaMDataset

DATA_DIR = os.path.join(ROOT, "gram_data")


# ── helpers ───────────────────────────────────────────────────────────────────

def get_split_membership(path: str, data_dir: str = DATA_DIR,
                         train_fraction: float = 0.9, seed: int = 42) -> str:
    """
    Returns 'train', 'val', or 'unknown' for a given .npz path,
    using the same deterministic split as training (seed=42, 90/10).
    """
    all_files = sorted(f for f in glob.glob(os.path.join(data_dir, "*.npz"))
                       if ".distcache" not in f)
    if not all_files:
        return "unknown"

    n_train  = int(len(all_files) * train_fraction)
    rng      = torch.Generator().manual_seed(seed)
    indices  = torch.randperm(len(all_files), generator=rng).tolist()
    train_set = set(indices[:n_train])

    abs_path = os.path.abspath(path)
    for i, f in enumerate(all_files):
        if os.path.abspath(f) == abs_path:
            return "train" if i in train_set else "val"
    return "unknown"


def _load_sample(path: str, device):
    data         = np.load(path)
    velocity_in  = torch.from_numpy(data["velocity_in"]).unsqueeze(0).to(device)   # (1,5,N,3)
    pos          = torch.from_numpy(data["pos"]).unsqueeze(0).to(device)            # (1,N,3)
    t            = torch.from_numpy(data["t"]).unsqueeze(0).to(device)             # (1,10)
    velocity_out = torch.from_numpy(data["velocity_out"]).unsqueeze(0).to(device)  # (1,5,N,3)
    idcs_airfoil = [torch.from_numpy(data["idcs_airfoil"].astype(np.int64))]

    # Distance features
    cache_path = path.replace(".npz", ".distcache.npz")
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        ia, dist, xsign = cache["ia"], cache["dist"], cache["xsign"]
    else:
        ia, dist, xsign = precompute_distance_features(
            data["pos"], data["idcs_airfoil"].astype(np.int64)
        )
    dist_feats = [(torch.from_numpy(ia), torch.from_numpy(dist), torch.from_numpy(xsign))]

    # k-NN features (loaded if cache exists, else None)
    knn_path = path.replace(".npz", ".knncache.npz")
    if os.path.exists(knn_path):
        knn_feats = [torch.from_numpy(np.load(knn_path)["knn_idx"])]
    else:
        knn_feats = [None]

    return velocity_in, pos, t, velocity_out, idcs_airfoil, dist_feats, knn_feats


def _rel_l2(pred, target):
    """pred, target: (5, N, 3) numpy arrays  →  scalar per timestep list"""
    losses = []
    for i in range(pred.shape[0]):
        p = pred[i].reshape(-1)
        g = target[i].reshape(-1)
        losses.append(float(np.linalg.norm(p - g) / (np.linalg.norm(g) + 1e-8)))
    return losses


def _interp_xz(pos_np, scalar, grid_shape=(400, 200), y_tol=0.05):
    """Project to XZ plane (Y near 0) and interpolate to a regular grid."""
    mask = np.abs(pos_np[:, 1]) < y_tol
    if mask.sum() < 100:
        y_tol *= 3
        mask = np.abs(pos_np[:, 1]) < y_tol

    pts    = pos_np[mask, :][:, [0, 2]]   # XZ coords
    values = scalar[mask]

    xi = np.linspace(pos_np[:, 0].min(), pos_np[:, 0].max(), grid_shape[0])
    zi = np.linspace(pos_np[:, 2].min(), pos_np[:, 2].max(), grid_shape[1])
    XI, ZI = np.meshgrid(xi, zi, indexing="ij")

    grid = griddata(pts, values, (XI, ZI), method="linear")
    return grid, xi, zi


def _foil_mask(pos_np, idcs_airfoil_np):
    """Boolean mask of surface points."""
    m = np.zeros(len(pos_np), dtype=bool)
    m[idcs_airfoil_np] = True
    return m


# ── Figure 1: prediction grid ─────────────────────────────────────────────────

def plot_correction_grid(
    pos_np, idcs_np, pred_np, gt_np, poly_np, t_np,
    component=0, save_path=None,
):
    """
    4-column grid: GT | Prediction | Model error | Polynomial error
    This makes clear what the model learned on top of the polynomial.
    """
    comp_labels = ["Ux", "Uy", "Uz"]
    T = pred_np.shape[0]

    fig, axes = plt.subplots(T, 4, figsize=(20, 3 * T), facecolor="#111111")
    fig.suptitle(
        f"Component {comp_labels[component]} — GT / Pred / Model error / Poly error",
        color="white", fontsize=12, y=1.01,
    )

    vel_all = np.concatenate([gt_np[:, :, component].ravel(),
                               pred_np[:, :, component].ravel()])
    vmin, vmax = np.percentile(vel_all, 1), np.percentile(vel_all, 99)

    for row, ts in enumerate(range(T)):
        gt_field   = gt_np[ts, :, component]
        pred_field = pred_np[ts, :, component]
        poly_field = poly_np[ts, :, component]

        gt_grid,    xi, zi = _interp_xz(pos_np, gt_field)
        pred_grid,  _,  _  = _interp_xz(pos_np, pred_field)
        merr_grid,  _,  _  = _interp_xz(pos_np, pred_field - gt_field)
        perr_grid,  _,  _  = _interp_xz(pos_np, poly_field - gt_field)

        extent = [xi[0], xi[-1], zi[0], zi[-1]]
        e_scale = np.nanpercentile(np.abs(perr_grid), 98)  # same scale for both errors

        for col, (grid, cmap, vlo, vhi, title) in enumerate([
            (gt_grid,   "turbo",  vmin,     vmax,    f"GT t={t_np[5+ts]:.3f}"),
            (pred_grid, "turbo",  vmin,     vmax,    f"Pred t={t_np[5+ts]:.3f}"),
            (merr_grid, "RdBu_r", -e_scale, e_scale, f"Model err  (max={e_scale:.3f})"),
            (perr_grid, "RdBu_r", -e_scale, e_scale, f"Poly err   (max={e_scale:.3f})"),
        ]):
            ax = axes[row, col]
            ax.set_facecolor("#111111")
            im = ax.imshow(grid.T, origin="lower", aspect="auto",
                           extent=extent, cmap=cmap, vmin=vlo, vmax=vhi,
                           interpolation="bilinear")
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.yaxis.set_tick_params(
                color="white", labelcolor="white")
            ax.set_title(title, color="white", fontsize=8)
            ax.tick_params(colors="white", labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
            if col == 0:
                ax.set_ylabel(f"t{5+ts}", color="white", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#111111")
        print(f"Saved: {save_path}")
    return fig


def plot_prediction_grid(
    pos_np, idcs_np, pred_np, gt_np, t_np,
    component=0, save_path=None,
):
    """
    5 rows (timesteps t5–t9) × 3 cols (GT | Prediction | Error).
    component: 0=Ux, 1=Uy, 2=Uz
    """
    comp_labels = ["Ux", "Uy", "Uz"]
    T = pred_np.shape[0]

    fig, axes = plt.subplots(T, 3, figsize=(15, 3 * T), facecolor="#111111")
    fig.suptitle(f"Velocity component {comp_labels[component]}  —  t5 to t9",
                 color="white", fontsize=13, y=1.01)

    # Shared velocity colorscale across GT + Pred
    vel_all = np.concatenate([gt_np[:, :, component].ravel(),
                               pred_np[:, :, component].ravel()])
    vmin, vmax = np.percentile(vel_all, 1), np.percentile(vel_all, 99)
    err_abs    = np.abs(pred_np - gt_np)[:, :, component].ravel()
    emax       = np.percentile(err_abs, 99)

    for row, ts in enumerate(range(T)):
        gt_field   = gt_np[ts, :, component]
        pred_field = pred_np[ts, :, component]
        err_field  = pred_field - gt_field

        gt_grid,   xi, zi = _interp_xz(pos_np, gt_field)
        pred_grid, _,  _  = _interp_xz(pos_np, pred_field)
        err_grid,  _,  _  = _interp_xz(pos_np, err_field)

        extent = [xi[0], xi[-1], zi[0], zi[-1]]

        # Tight error scale: use actual error range, not global percentile
        # so small residuals are visible even when GT and pred look identical
        e_row = np.nanpercentile(np.abs(err_grid), 98)

        for col, (grid, cmap, vlo, vhi, title) in enumerate([
            (gt_grid,   "turbo",  vmin, vmax,   f"GT  t={t_np[5+ts]:.3f}"),
            (pred_grid, "turbo",  vmin, vmax,   f"Pred t={t_np[5+ts]:.3f}"),
            (err_grid,  "RdBu_r", -e_row, e_row, f"Error (×1)  max={e_row:.3f}"),
        ]):
            ax = axes[row, col]
            ax.set_facecolor("#111111")
            im = ax.imshow(
                grid.T, origin="lower", aspect="auto",
                extent=extent, cmap=cmap,
                vmin=vlo, vmax=vhi, interpolation="bilinear",
            )
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.yaxis.set_tick_params(color="white", labelcolor="white")
            ax.set_title(title, color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
            if col == 0:
                ax.set_ylabel(f"t{5+ts}", color="white", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#111111")
        print(f"Saved: {save_path}")
    return fig


# ── Figure 2: per-timestep bar chart ─────────────────────────────────────────

def plot_timestep_comparison(rel_l2_model, rel_l2_poly, save_path=None):
    """Bar chart: model vs polynomial baseline, per output timestep."""
    ts    = [f"t{5+i}" for i in range(5)]
    x     = np.arange(5)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#111111")
    ax.set_facecolor("#111111")

    bars1 = ax.bar(x - width/2, rel_l2_poly,  width, label="Polynomial baseline", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, rel_l2_model, width, label="TransolverResidual",  color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ts, color="white")
    ax.set_ylabel("Relative L2", color="white")
    ax.set_title("Per-timestep relative L2: model vs polynomial baseline", color="white")
    ax.legend(facecolor="#222222", labelcolor="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    # Annotate improvement
    for xi, (pm, pp) in enumerate(zip(rel_l2_model, rel_l2_poly)):
        gain = (pp - pm) / pp * 100
        color = "#55CC55" if gain > 0 else "#CC5555"
        ax.text(xi, max(pm, pp) + 0.01, f"{gain:+.1f}%", ha="center",
                color=color, fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#111111")
        print(f"Saved: {save_path}")
    return fig


# ── Figure 3: velocity magnitude error map ────────────────────────────────────

def plot_error_map(pos_np, pred_np, gt_np, timestep=0, save_path=None):
    """XZ error map for velocity magnitude at a single timestep."""
    pred_mag = np.linalg.norm(pred_np[timestep], axis=-1)
    gt_mag   = np.linalg.norm(gt_np[timestep],   axis=-1)
    err      = np.abs(pred_mag - gt_mag)

    gt_grid,   xi, zi = _interp_xz(pos_np, gt_mag)
    pred_grid, _,  _  = _interp_xz(pos_np, pred_mag)
    err_grid,  _,  _  = _interp_xz(pos_np, err)

    extent = [xi[0], xi[-1], zi[0], zi[-1]]
    vmax   = np.nanpercentile(np.abs(np.concatenate([gt_grid.ravel(), pred_grid.ravel()])), 99)
    emax   = np.nanpercentile(err_grid.ravel()[~np.isnan(err_grid.ravel())], 99)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#111111")
    fig.suptitle(f"|U| at t{5+timestep}", color="white", fontsize=12)

    for ax, (grid, cmap, vlo, vhi, title) in zip(axes, [
        (gt_grid,   "turbo", 0, vmax, "Ground truth |U|"),
        (pred_grid, "turbo", 0, vmax, "Predicted |U|"),
        (err_grid,  "hot",   0, emax, "Absolute error"),
    ]):
        ax.set_facecolor("#111111")
        im = ax.imshow(grid.T, origin="lower", aspect="auto",
                       extent=extent, cmap=cmap, vmin=vlo, vmax=vhi,
                       interpolation="bilinear")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.yaxis.set_tick_params(color="white", labelcolor="white")
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#111111")
        print(f"Saved: {save_path}")
    return fig


# ── Slice assignment visualisation ───────────────────────────────────────────

def _slice_entropy_stats(sw: np.ndarray, layer_label: str = "") -> np.ndarray:
    """
    Compute and print entropy stats for slice weights.

    sw : (H, N, M) float32 numpy array
    Returns entropy (N,) averaged over heads.
    """
    eps     = 1e-9
    M       = sw.shape[-1]
    max_ent = np.log(M)
    entropy = -(sw * np.log(sw + eps)).sum(axis=-1).mean(axis=0)  # (N,)
    mean_frac = entropy.mean() / max_ent

    tag = f" — {layer_label}" if layer_label else ""
    print(f"\nSlice assignment entropy{tag}:")
    print(f"  M={M} slices  |  max entropy = {max_ent:.3f} nats")
    print(f"  Mean   = {entropy.mean():.3f} nats  ({mean_frac*100:.1f}% of max)")
    print(f"  Median = {np.median(entropy):.3f} nats")
    if mean_frac < 0.40:
        verdict = "Sharp — slices have learned distinct physical regimes."
    elif mean_frac < 0.65:
        verdict = "Moderate — some spatial structure present."
    else:
        verdict = "MUSHY (>65% of max) — slices barely differentiated; model may be underfitting."
    print(f"  → {verdict}")
    return entropy


@torch.no_grad()
@torch.no_grad()
def _extract_slice_weights(model, pos, velocity_in, idcs_airfoil, t,
                            dist_feats, knn_feats, layer_idx: int):
    """
    Run the model encoder + blocks up to layer_idx and return (B, H, N, M)
    slice weight tensor on CPU.
    """
    from models.transolver_residual.features import compute_features

    device = next(model.parameters()).device
    feats  = compute_features(
        pos, velocity_in, idcs_airfoil, t,
        poly_degree          = model.poly_degree,
        dist_feats           = dist_feats,
        use_local_feats      = model.use_local_feats,
        use_temporal_deltas  = model.use_temporal_deltas,
        knn_feats            = knn_feats,
    )
    x = model.encoder(feats)           # (B, N, C)

    n_blocks = len(model.blocks)
    target   = n_blocks + layer_idx if layer_idx < 0 else min(layer_idx, n_blocks - 1)
    for i, block in enumerate(model.blocks):
        if i == target:
            break
        x = block(x)

    block  = model.blocks[target]
    attn   = block.attn
    B, N, C = x.shape
    normed = block.norm1(x)

    x_mid = (attn.proj_x(normed)
               .reshape(B, N, attn.heads, attn.dim_head)
               .permute(0, 2, 1, 3))                     # (B, H, N, dim_head)
    temp  = attn.temperature.clamp(0.1, 5.0)
    sw    = torch.softmax(attn.proj_slice(x_mid) / temp, dim=-1)   # (B, H, N, M)
    return sw[0].cpu().float().numpy(), target   # (H, N, M), int


@torch.no_grad()
def plot_slice_assignments(model, pos_np, velocity_in, pos, t, idcs_airfoil,
                           dist_feats, knn_feats=None, layer_idx=0, save_path=None):
    """
    Visualise Physics-Attention slice assignments.  Produces two figures:

    Figure A — Per-slice weight heatmaps (one panel per slice, averaged over
               heads, interpolated to XZ grid).  This is the same visualization
               as Figure 1 in the Transolver paper.  Each panel shows which
               region of the flow belongs to that slice.  Spatially coherent
               blobs = attention is working; uniform noise = mushy.

    Figure B — Entropy map + histogram.  Low entropy = point is sharply
               assigned to one slice (freestream behaves predictably).
               High entropy = point is spread across many slices (wake,
               boundary layer — legitimately ambiguous physics).
    """
    model.eval()

    sw, target = _extract_slice_weights(
        model, pos, velocity_in, idcs_airfoil, t,
        dist_feats, knn_feats, layer_idx,
    )
    H, N, M = sw.shape

    # Mean over heads → (N, M) — one weight per point per slice
    sw_mean = sw.mean(axis=0)   # (N, M)

    # ── Entropy stats (always printed) ────────────────────────────────────────
    entropy = _slice_entropy_stats(sw, layer_label=f"layer {target}")
    max_ent = np.log(M)

    # ── Figure A: Per-slice weight heatmaps ───────────────────────────────────
    # Lay out M slices in a grid, each showing where that slice concentrates.
    # Inspired directly by Fig 1 of Wu et al. (Transolver, ICML 2024).
    n_cols = 8
    n_rows = (M + n_cols - 1) // n_cols

    fig_a, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2.2 * n_rows),
        facecolor="#0a0a0a",
    )
    axes = np.array(axes).reshape(n_rows, n_cols)
    fig_a.suptitle(
        f"Per-slice weight maps — layer {target}  "
        f"({M} slices, avg over {H} heads)\n"
        f"Bright region = points assigned to this slice  "
        f"|  structure = working,  uniform noise = mushy",
        color="white", fontsize=10, y=1.01,
    )

    extent = [pos_np[:, 0].min(), pos_np[:, 0].max(),
              pos_np[:, 2].min(), pos_np[:, 2].max()]

    # Sort slices by how much "mass" they carry (descending) so the active
    # slices appear first and dead/degenerate ones come last.
    slice_mass = sw_mean.sum(axis=0)               # (M,) — total weight across all points
    sorted_slices = np.argsort(slice_mass)[::-1]   # most active first

    for panel_idx in range(n_rows * n_cols):
        r, c = divmod(panel_idx, n_cols)
        ax   = axes[r, c]
        ax.set_facecolor("#0a0a0a")

        if panel_idx >= M:
            ax.set_visible(False)
            continue

        m   = sorted_slices[panel_idx]
        wts = sw_mean[:, m]                        # (N,) weight of slice m at each point
        grid, xi, zi = _interp_xz(pos_np, wts)

        im = ax.imshow(
            grid.T, origin="lower", aspect="auto",
            extent=extent, cmap="hot",
            vmin=0, vmax=wts.max(),
            interpolation="bilinear",
        )
        mass_pct = slice_mass[m] / slice_mass.sum() * 100
        ax.set_title(f"s{m}  ({mass_pct:.1f}%)", color="white", fontsize=6, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        p = save_path.replace(".png", f"_slices_layer{target}.png")
        fig_a.savefig(p, dpi=130, bbox_inches="tight", facecolor="#0a0a0a")
        print(f"Saved: {p}")

    # ── Figure B: Entropy heatmap + histogram ─────────────────────────────────
    fig_b, axes2 = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111111")
    fig_b.suptitle(
        f"Slice assignment entropy — layer {target}  "
        f"(low=sharp, high=mushy)   mean={entropy.mean():.3f} / max={max_ent:.3f} "
        f"({entropy.mean()/max_ent*100:.1f}%)",
        color="white", fontsize=10,
    )

    # Left panel: entropy heatmap on XZ grid
    ax = axes2[0]
    ax.set_facecolor("#111111")
    ent_grid, xi, zi = _interp_xz(pos_np, entropy)
    im = ax.imshow(
        ent_grid.T, origin="lower", aspect="auto",
        extent=extent, cmap="plasma",
        vmin=0, vmax=max_ent, interpolation="bilinear",
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("Entropy (nats)", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    ax.set_title("Per-point entropy (interpolated XZ)", color="white", fontsize=9)
    ax.set_xlabel("X", color="white", fontsize=8)
    ax.set_ylabel("Z", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")

    # Right panel: histogram
    ax2 = axes2[1]
    ax2.set_facecolor("#111111")
    ax2.hist(entropy, bins=80, color="#DD8452", edgecolor="none", alpha=0.85)
    ax2.axvline(x=max_ent, color="#CC5555", lw=1.5, ls="--",
                label=f"Max entropy = {max_ent:.2f}")
    ax2.axvline(x=entropy.mean(), color="#55CC55", lw=1.5, ls="--",
                label=f"Mean = {entropy.mean():.2f}  ({entropy.mean()/max_ent*100:.0f}%)")
    ax2.set_xlabel("Entropy (nats)", color="white")
    ax2.set_ylabel("Point count", color="white")
    ax2.set_title("Entropy distribution", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#222222", labelcolor="white", fontsize=9)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#444444")

    plt.tight_layout()
    if save_path:
        p = save_path.replace(".png", f"_entropy_layer{target}.png")
        fig_b.savefig(p, dpi=130, bbox_inches="tight", facecolor="#111111")
        print(f"Saved: {p}")

    return fig_a, fig_b


# ── Multi-sample evaluation ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate_n_samples(model, files, n, device, save_dir=None):
    """
    Run the model on n randomly chosen samples, collect per-timestep losses,
    and plot a distribution.
    """
    import random
    chosen = random.sample(files, min(n, len(files)))

    all_model = []
    all_poly  = []

    for path in chosen:
        velocity_in, pos, t, velocity_out, idcs_airfoil, dist_feats, knn_feats = _load_sample(path, device)

        pred = model(t, pos, idcs_airfoil, velocity_in, dist_feats, knn_feats)
        poly = poly_extrapolate(velocity_in, t, degree=2)

        gt_np   = velocity_out[0].cpu().numpy()   # (5, N, 3)
        pred_np = pred[0].cpu().numpy()
        poly_np = poly[0].cpu().numpy()

        all_model.append(_rel_l2(pred_np, gt_np))
        all_poly.append(_rel_l2(poly_np, gt_np))

    all_model = np.array(all_model)   # (n, 5)
    all_poly  = np.array(all_poly)

    # Summary table
    print("\nRelative L2 summary (mean ± std over {} samples)".format(len(chosen)))
    print(f"{'timestep':>10}  {'poly':>10}  {'model':>10}  {'improvement':>12}")
    for i in range(5):
        pm = all_model[:, i].mean()
        pp = all_poly[:, i].mean()
        ps = all_model[:, i].std()
        gain = (pp - pm) / pp * 100
        print(f"{'t'+str(5+i):>10}  {pp:>10.4f}  {pm:>10.4f} ±{ps:.4f}  {gain:>+11.1f}%")

    # Violin plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#111111")
    ts_labels = [f"t{5+i}" for i in range(5)]

    for ax, (data, title, color) in zip(axes, [
        (all_poly,  "Polynomial baseline", "#4C72B0"),
        (all_model, "TransolverResidual",  "#DD8452"),
    ]):
        ax.set_facecolor("#111111")
        parts = ax.violinplot(data, positions=range(5), showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[key].set_color("white")
        ax.set_xticks(range(5))
        ax.set_xticklabels(ts_labels, color="white")
        ax.set_ylabel("Relative L2", color="white")
        ax.set_title(title, color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    fig.suptitle(f"Error distribution over {len(chosen)} val samples", color="white", fontsize=12)
    plt.tight_layout()

    if save_dir:
        fig.savefig(os.path.join(save_dir, "error_distribution.png"),
                    dpi=120, bbox_inches="tight", facecolor="#111111")
        print(f"Saved: {os.path.join(save_dir, 'error_distribution.png')}")

    return fig, all_model, all_poly


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualise TransolverResidual")
    parser.add_argument("--data_dir",   default=DATA_DIR)
    parser.add_argument("--sim_id",     default=None,
                        help="Specific simulation ID to visualise (e.g. 1025_1). "
                             "If None, picks a random validation sample.")
    parser.add_argument("--window",     type=int, default=0,
                        help="Time window index within the simulation.")
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="If set, evaluate this many samples and plot distributions.")
    parser.add_argument("--component",  type=int, default=0,
                        help="Velocity component for prediction grid (0=Ux,1=Uy,2=Uz)")
    parser.add_argument("--timestep",   type=int, default=2,
                        help="Output timestep (0–4) for error map")
    parser.add_argument("--save_dir",   default=None,
                        help="Directory to save figures. If None, figures are shown interactively.")
    parser.add_argument("--no_show",    action="store_true",
                        help="Don't call plt.show() (useful when saving to disk only)")
    parser.add_argument("--slice_vis",  action="store_true",
                        help="Visualise Physics-Attention slice assignments")
    parser.add_argument("--slice_layer", type=int, default=0,
                        help="Which Transolver block to inspect for slice vis (0=first, -1=last)")

    # Model hyperparams (must match training)
    parser.add_argument("--n_layers",            type=int,   default=8)
    parser.add_argument("--hidden_dim",          type=int,   default=256)
    parser.add_argument("--n_heads",             type=int,   default=8)
    parser.add_argument("--slice_num",           type=int,   default=32)
    parser.add_argument("--mlp_ratio",           type=int,   default=1)
    parser.add_argument("--dropout",             type=float, default=0.0)
    parser.add_argument("--use_local_feats",     action="store_true",
                        help="Must be set if the model was trained with --use_local_feats")
    parser.add_argument("--use_temporal_deltas", action="store_true",
                        help="Must be set if the model was trained with --use_temporal_deltas")

    args = parser.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    if not args.no_show and args.save_dir is None:
        matplotlib.use("TkAgg")   # switch to interactive backend when displaying

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────────
    model = TransolverResidual(
        n_layers             = args.n_layers,
        hidden_dim           = args.hidden_dim,
        n_heads              = args.n_heads,
        slice_num            = args.slice_num,
        mlp_ratio            = args.mlp_ratio,
        dropout              = args.dropout,
        use_local_feats      = args.use_local_feats,
        use_temporal_deltas  = args.use_temporal_deltas,
    ).to(device)
    model.eval()
    print(f"Model loaded  ({model.num_params():,} params)")

    # ── Multi-sample evaluation ────────────────────────────────────────────────
    if args.n_samples is not None:
        all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
        if not all_files:
            print(f"No .npz files found in {args.data_dir}")
            return
        with torch.no_grad():
            fig, _, _ = evaluate_n_samples(model, all_files, args.n_samples, device, args.save_dir)
        if not args.no_show:
            plt.show()
        return

    # ── Single-sample visualisation ────────────────────────────────────────────
    if args.sim_id is not None:
        pattern = os.path.join(args.data_dir, f"{args.sim_id}-{args.window}.npz")
        matches = glob.glob(pattern)
        if not matches:
            print(f"File not found: {pattern}")
            return
        path = matches[0]
    else:
        all_files = sorted(f for f in glob.glob(os.path.join(args.data_dir, "*.npz"))
                           if ".distcache" not in f)
        if not all_files:
            print(f"No .npz files found in {args.data_dir}")
            return
        import random
        path = random.choice(all_files)
        print(f"Randomly selected: {os.path.basename(path)}")

    split = get_split_membership(path, args.data_dir)
    label = {"train": "\033[93mTRAIN SET\033[0m  ← model has seen this sample",
             "val":   "\033[92mVAL SET\033[0m    ← held-out, fair evaluation",
             "unknown": "\033[91mUNKNOWN\033[0m"}.get(split, split)
    print(f"Split membership: {label}")

    velocity_in, pos, t, velocity_out, idcs_airfoil, dist_feats, knn_feats = _load_sample(path, device)

    with torch.no_grad():
        pred = model(t, pos, idcs_airfoil, velocity_in, dist_feats, knn_feats)
        poly = poly_extrapolate(velocity_in, t, degree=2)

    pos_np    = pos[0].cpu().numpy()
    idcs_np   = idcs_airfoil[0].numpy()
    gt_np     = velocity_out[0].cpu().numpy()    # (5, N, 3)
    pred_np   = pred[0].cpu().numpy()
    poly_np   = poly[0].cpu().numpy()
    t_np      = t[0].cpu().numpy()

    model_losses = _rel_l2(pred_np, gt_np)
    poly_losses  = _rel_l2(poly_np, gt_np)

    print("\nPer-timestep relative L2:")
    print(f"{'':6}  {'poly':>8}  {'model':>8}  {'gain':>8}")
    for i, (pm, pp) in enumerate(zip(model_losses, poly_losses)):
        gain = (pp - pm) / pp * 100
        print(f"t{5+i}:    {pp:>8.4f}  {pm:>8.4f}  {gain:>+7.1f}%")

    # Always print slice entropy for first, middle, and last layer so we can
    # monitor whether Physics-Attention is routing correctly without --slice_vis.
    print("\n── Slice entropy quick-check (no figures) ──")
    n_blocks = len(model.blocks)
    for lidx in sorted({0, n_blocks // 2, n_blocks - 1}):
        sw_l, tgt_l = _extract_slice_weights(
            model, pos, velocity_in, idcs_airfoil, t, dist_feats, knn_feats, lidx
        )
        _slice_entropy_stats(sw_l, layer_label=f"layer {tgt_l}")

    sp = lambda name: os.path.join(args.save_dir, name) if args.save_dir else None

    fig1 = plot_correction_grid(
        pos_np, idcs_np, pred_np, gt_np, poly_np, t_np,
        component=args.component,
        save_path=sp(f"correction_grid_comp{args.component}.png"),
    )
    fig1b = plot_prediction_grid(
        pos_np, idcs_np, pred_np, gt_np, t_np,
        component=args.component,
        save_path=sp(f"prediction_grid_comp{args.component}.png"),
    )
    fig2 = plot_timestep_comparison(
        model_losses, poly_losses,
        save_path=sp("timestep_comparison.png"),
    )
    fig3 = plot_error_map(
        pos_np, pred_np, gt_np,
        timestep=args.timestep,
        save_path=sp(f"error_map_t{5+args.timestep}.png"),
    )

    if args.slice_vis:
        n_blocks = len(model.blocks)
        layers = ([args.slice_layer] if args.slice_layer >= 0
                  else [0, n_blocks // 2, n_blocks - 1])
        for layer_idx in layers:
            plot_slice_assignments(
                model, pos_np, velocity_in, pos, t, idcs_airfoil,
                dist_feats, knn_feats,
                layer_idx=layer_idx,
                save_path=sp("slice_vis.png"),
            )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
