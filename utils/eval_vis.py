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

    ia, dist, xsign = precompute_distance_features(
        data["pos"], data["idcs_airfoil"].astype(np.int64)
    )
    dist_feats = [(torch.from_numpy(ia), torch.from_numpy(dist), torch.from_numpy(xsign))]

    return velocity_in, pos, t, velocity_out, idcs_airfoil, dist_feats


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

@torch.no_grad()
def plot_slice_assignments(model, pos_np, velocity_in, pos, t, idcs_airfoil,
                           dist_feats, layer_idx=0, save_path=None):
    """
    Visualise what the Physics-Attention slices have learned.

    Produces two figures:
      (a) Dominant slice map — XZ scatter coloured by argmax(slice_weights),
          one panel per attention head. If the slices are physically meaningful
          you'll see spatial structure (freestream, wake, boundary layer).
          If they're mushy you'll see random salt-and-pepper noise.

      (b) Slice entropy map — XZ scatter coloured by the entropy of the slice
          weight distribution at each point. Low entropy (dark) = point
          strongly assigned to one slice. High entropy (bright) = mushy,
          point spreads mass across all slices equally.
          A well-trained model should show low entropy in the freestream
          (clean laminar regime) and moderate entropy in the wake (turbulence
          genuinely lives in multiple regimes at once).

    Args:
        layer_idx : which Transolver block to inspect (0 = first, -1 = last)
    """
    from models.transolver_residual.features import compute_features
    from models.transolver_residual.polynomial import poly_extrapolate

    device = next(model.parameters()).device
    model.eval()

    # ── Run encoder to get per-point embeddings ───────────────────────────────
    with torch.no_grad():
        feats = compute_features(
            pos, velocity_in, idcs_airfoil, t,
            poly_degree=model.poly_degree, dist_feats=dist_feats,
        )
        x = model.encoder(feats)                    # (1, N, C)
        # Run blocks up to the target layer
        n_blocks = len(model.blocks)
        target = n_blocks + layer_idx if layer_idx < 0 else layer_idx
        for i, block in enumerate(model.blocks):
            if i == target:
                break
            x = block(x)
        # Extract slice weights from the target block's attention
        block = model.blocks[target]
        attn  = block.attn
        B, N, C = x.shape
        normed = block.norm1(x)

        # Replicate PhysicsAttention slice computation
        x_mid = attn.proj_x(normed).reshape(B, N, attn.heads, attn.dim_head) \
                     .permute(0, 2, 1, 3)                   # (B, H, N, dim_head)
        temp  = attn.temperature.clamp(0.1, 5.0)
        slice_weights = torch.softmax(
            attn.proj_slice(x_mid) / temp, dim=-1
        )                                                   # (B, H, N, M)

    # slice_weights: (1, H, N, M) — bring to CPU numpy
    sw = slice_weights[0].cpu().float().numpy()             # (H, N, M)
    H, N, M = sw.shape

    pos_xz = pos_np[:, [0, 2]]    # (N, 2) — X and Z coords
    surf_mask = np.zeros(N, dtype=bool)
    surf_mask[idcs_airfoil[0].numpy()] = True

    # ── Figure (a): Dominant slice per head ──────────────────────────────────
    cols  = min(4, H)
    rows  = (H + cols - 1) // cols
    fig_a, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                                facecolor="#111111")
    axes = np.array(axes).reshape(rows, cols)
    fig_a.suptitle(
        f"Dominant slice assignment — layer {target}  "
        f"(structured = Physics-Attention working, noise = not working)",
        color="white", fontsize=11,
    )

    cmap_slices = plt.get_cmap("tab20", M)
    subsample = max(1, N // 20_000)   # show at most 20k points for speed

    for h in range(H):
        r, c   = divmod(h, cols)
        ax     = axes[r, c]
        ax.set_facecolor("#111111")
        dom    = sw[h, ::subsample].argmax(axis=-1)   # (N//sub,) dominant slice index
        px, pz = pos_xz[::subsample, 0], pos_xz[::subsample, 1]
        sc = ax.scatter(px, pz, c=dom, cmap=cmap_slices,
                        vmin=0, vmax=M - 1, s=0.3, linewidths=0)
        # Overlay airfoil surface in white
        sf_sub = surf_mask[::subsample]
        ax.scatter(px[sf_sub], pz[sf_sub], c="white", s=0.8, linewidths=0)
        ax.set_title(f"Head {h}", color="white", fontsize=9)
        ax.set_xlabel("X", color="white", fontsize=7)
        ax.set_ylabel("Z", color="white", fontsize=7)
        ax.tick_params(colors="white", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")

    # Hide unused subplots
    for h in range(H, rows * cols):
        r, c = divmod(h, cols)
        axes[r, c].set_visible(False)

    plt.tight_layout()
    if save_path:
        p = save_path.replace(".png", f"_dominant_layer{target}.png")
        fig_a.savefig(p, dpi=120, bbox_inches="tight", facecolor="#111111")
        print(f"Saved: {p}")

    # ── Figure (b): Entropy map (averaged over heads) ─────────────────────────
    # Entropy of slice distribution per point per head, averaged over heads
    eps     = 1e-9
    entropy = -(sw * np.log(sw + eps)).sum(axis=-1)   # (H, N)
    entropy = entropy.mean(axis=0)                     # (N,) mean over heads
    max_ent = np.log(M)                                # maximum possible entropy

    fig_b, axes2 = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111111")
    fig_b.suptitle(
        f"Slice assignment entropy — layer {target}  "
        f"(low=sharp assignment, high=mushy/uninformative)",
        color="white", fontsize=11,
    )

    # Scatter
    ax = axes2[0]
    ax.set_facecolor("#111111")
    px, pz = pos_xz[::subsample, 0], pos_xz[::subsample, 1]
    sc = ax.scatter(px, pz, c=entropy[::subsample],
                    cmap="plasma", vmin=0, vmax=max_ent, s=0.3, linewidths=0)
    sf_sub = surf_mask[::subsample]
    ax.scatter(px[sf_sub], pz[sf_sub], c="white", s=0.8, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, fraction=0.03)
    cb.set_label("Entropy (nats)", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    ax.axhline(y=0, color="#555555", lw=0.5, ls="--")
    ax.set_title("Per-point entropy (XZ view, Y≈0)", color="white")
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Z", color="white")
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")

    # Histogram
    ax2 = axes2[1]
    ax2.set_facecolor("#111111")
    ax2.hist(entropy, bins=60, color="#DD8452", edgecolor="none", alpha=0.85)
    ax2.axvline(x=max_ent, color="#CC5555", lw=1.5, ls="--",
                label=f"Max entropy (={max_ent:.2f})")
    ax2.axvline(x=entropy.mean(), color="#55CC55", lw=1.5, ls="--",
                label=f"Mean = {entropy.mean():.2f}")
    ax2.set_xlabel("Entropy (nats)", color="white")
    ax2.set_ylabel("Point count", color="white")
    ax2.set_title("Entropy distribution over all points", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#222222", labelcolor="white", fontsize=9)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#444444")

    mean_frac = entropy.mean() / max_ent
    print(f"\nSlice assignment analysis — layer {target}:")
    print(f"  Max possible entropy:  {max_ent:.3f} nats  (= all slices equally likely)")
    print(f"  Mean entropy:          {entropy.mean():.3f} nats  ({mean_frac*100:.1f}% of max)")
    print(f"  Median entropy:        {np.median(entropy):.3f} nats")
    print()
    if mean_frac < 0.5:
        print("  → Sharp assignments: Physics-Attention is learning distinct regimes.")
    elif mean_frac < 0.75:
        print("  → Moderate sharpness: Some structure, but not maximally informative.")
    else:
        print("  → Mushy assignments (>75% of max entropy): slices may not be contributing.")
        print("    Consider: more training, higher temperature init, or drop Transolver blocks.")

    plt.tight_layout()
    if save_path:
        p = save_path.replace(".png", f"_entropy_layer{target}.png")
        fig_b.savefig(p, dpi=120, bbox_inches="tight", facecolor="#111111")
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
        velocity_in, pos, t, velocity_out, idcs_airfoil, dist_feats = _load_sample(path, device)

        pred = model(t, pos, idcs_airfoil, velocity_in, dist_feats)
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
    parser.add_argument("--n_layers",   type=int,   default=8)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--n_heads",    type=int,   default=8)
    parser.add_argument("--slice_num",  type=int,   default=32)
    parser.add_argument("--mlp_ratio",  type=int,   default=1)
    parser.add_argument("--dropout",    type=float, default=0.0)

    args = parser.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    if not args.no_show and args.save_dir is None:
        matplotlib.use("TkAgg")   # switch to interactive backend when displaying

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────────
    model = TransolverResidual(
        n_layers    = args.n_layers,
        hidden_dim  = args.hidden_dim,
        n_heads     = args.n_heads,
        slice_num   = args.slice_num,
        mlp_ratio   = args.mlp_ratio,
        dropout     = args.dropout,
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

    velocity_in, pos, t, velocity_out, idcs_airfoil, dist_feats = _load_sample(path, device)

    with torch.no_grad():
        pred = model(t, pos, idcs_airfoil, velocity_in, dist_feats)
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
        for layer_idx in ([args.slice_layer] if args.slice_layer >= 0
                          else [0, len(model.blocks) // 2, -1]):
            plot_slice_assignments(
                model, pos_np, velocity_in, pos, t, idcs_airfoil, dist_feats,
                layer_idx=layer_idx,
                save_path=sp("slice_vis.png"),
            )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
