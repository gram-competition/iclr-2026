"""
Visualization utilities for GRaM dataset.

Coordinate frame:
    X  — streamwise (freestream left→right), range ≈ [0, 2.1]
    Y  — foil cross-section height (thin ≈ [-0.04, 0.04] at surface), range ≈ [-0.4, 0.4]
    Z  — spanwise, range ≈ [0, 1.2]

Three staggered foils live in separate Z bands:
    Foil 1  Z ≈ [0.06, 0.25]   X ≈ [0.37, 0.90]
    Foil 2  Z ≈ [0.28, 0.48]   X ≈ [1.15, 1.47]
    Foil 3  Z ≈ [0.55, 0.83]   X ≈ [1.44, 1.70]

The natural "plan view" is XZ (top-down).
The natural "profile view" is XY (side-on, ideally sliced at fixed Z).

Usage:
    from utils.visualize import plot_geometry, plot_velocity_field

    fig, ax = plot_geometry("1021_1")
    fig, ax = plot_velocity_field("1021_1", window=0, timestep=2, split="in", component="magnitude")
"""

import glob
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "gram_data")

# Axis index and labels for each 2-D projection
_PROJ = {
    "xz": (0, 2, "X (streamwise)", "Z (spanwise)"),
    "xy": (0, 1, "X (streamwise)", "Y (height)"),
    "yz": (1, 2, "Y (height)",     "Z (spanwise)"),
}


def _load(sim_id: str, window: int = 0) -> dict:
    path = os.path.join(DATA_DIR, f"{sim_id}-{window}.npz")
    if not os.path.exists(path):
        matches = sorted(glob.glob(os.path.join(DATA_DIR, f"{sim_id}-*.npz")))
        if not matches:
            raise FileNotFoundError(
                f"No files found for simulation '{sim_id}' in {DATA_DIR}"
            )
        path = matches[0]
    return dict(np.load(path))


def _subsample(idcs: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(idcs) <= n:
        return idcs
    rng = np.random.default_rng(seed)
    return rng.choice(idcs, size=n, replace=False)


# ---------------------------------------------------------------------------
# plot_geometry
# ---------------------------------------------------------------------------

def plot_geometry(
    sim_id: str,
    window: int = 0,
    projection: str = "xz",
    n_fluid: int = 8000,
    ax=None,
):
    """
    Plot the 3-D point-cloud geometry for a simulation, highlighting the
    airfoil surface.

    Args:
        sim_id:     Simulation ID string, e.g. "1021_1".
        window:     Time window (0-4); all windows share pos/idcs_airfoil.
        projection: "xz" (top-down plan view, default), "xy" (side), or "yz".
        n_fluid:    Number of background fluid points to render (subsampled).
        ax:         Optional existing Axes.

    Returns:
        fig, ax
    """
    if projection not in _PROJ:
        raise ValueError(f"projection must be one of {list(_PROJ)}")

    data = _load(sim_id, window)
    pos = data["pos"]               # (100000, 3)
    idcs_airfoil = data["idcs_airfoil"]

    is_airfoil = np.zeros(len(pos), dtype=bool)
    is_airfoil[idcs_airfoil] = True
    fluid_idcs = _subsample(np.where(~is_airfoil)[0], n_fluid)

    i, j, xlabel, ylabel = _PROJ[projection]

    fig, ax = (plt.subplots(figsize=(12, 7)) if ax is None else (ax.get_figure(), ax))

    # Background fluid points
    ax.scatter(
        pos[fluid_idcs, i], pos[fluid_idcs, j],
        s=0.4, c="#a8c4e0", alpha=0.25, linewidths=0,
        label=f"Fluid ({n_fluid:,} pts, subsampled)",
        rasterized=True,
    )

    # Airfoil surface — all points, coloured red
    ax.scatter(
        pos[idcs_airfoil, i], pos[idcs_airfoil, j],
        s=1.2, c="#c0392b", alpha=0.85, linewidths=0,
        label=f"Airfoil surface ({len(idcs_airfoil):,} pts)",
        rasterized=True,
    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_aspect("equal")
    ax.set_title(f"Geometry — sim {sim_id}  (window {window})  [{projection.upper()} view]",
                 fontsize=13)
    leg = ax.legend(loc="upper right", markerscale=6, fontsize=9)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Internal helpers: grid interpolation + foil rendering
# ---------------------------------------------------------------------------

def _to_scalar(vel: np.ndarray, component: str) -> np.ndarray:
    """(N,3) velocity array → (N,) scalar for the requested component."""
    if component == "magnitude":
        return np.linalg.norm(vel, axis=-1)
    return vel[:, {"x": 0, "y": 1, "z": 2}[component]]


def _interpolate_to_grid(
    pos: np.ndarray,
    scalar: np.ndarray,
    y_slice: float = 0.0,
    y_tol: float = 0.012,
    grid_shape: tuple = (600, 300),
) -> tuple:
    """
    Slice at Y ≈ y_slice, interpolate (X,Z) scattered points onto a regular grid.
    Returns (grid, x_range, z_range).  grid has shape (nz, nx).
    """
    mask = np.abs(pos[:, 1] - y_slice) < y_tol
    xz   = pos[mask][:, [0, 2]]
    vals = scalar[mask]

    x_range = (pos[:, 0].min(), pos[:, 0].max())
    z_range = (pos[:, 2].min(), pos[:, 2].max())
    xi = np.linspace(*x_range, grid_shape[0])
    zi = np.linspace(*z_range, grid_shape[1])
    Xi, Zi = np.meshgrid(xi, zi)
    grid = griddata(xz, vals, (Xi, Zi), method="linear")
    return grid, x_range, z_range


def _draw_field(
    ax,
    grid: np.ndarray,
    x_range: tuple,
    z_range: tuple,
    foil_patches: list,
    cmap: str,
    vmin: float,
    vmax: float,
    dark: bool = True,
):
    """
    Draw an interpolated field on ax with foil polygons on top.
    Returns the imshow object (for attaching a colorbar).
    """
    if dark:
        ax.set_facecolor("#0d0d0d")
    im = ax.imshow(
        grid,
        origin="lower",
        extent=[*x_range, *z_range],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    pc = PatchCollection(
        foil_patches, facecolor="#606060", edgecolor="white",
        linewidths=0.6, zorder=5,
    )
    ax.add_collection(pc)
    ax.set_xlim(*x_range)
    ax.set_ylim(*z_range)
    return im


# ---------------------------------------------------------------------------
# plot_velocity_field
# ---------------------------------------------------------------------------

def plot_velocity_field(
    sim_id: str,
    window: int = 0,
    timestep: int = 0,
    split: str = "in",
    component: str = "magnitude",
    y_slice: float = 0.0,
    y_tol: float = 0.012,
    grid_shape: tuple = (600, 300),
    vmin=None,
    vmax=None,
    ax=None,
):
    """
    Plot the velocity field for one timestep in CFD-style (interpolated grid,
    foils as filled grey shapes, dark background).

    Colourmap: "turbo" for magnitude  |  "RdBu_r" for signed components (x/y/z).

    Args:
        sim_id:     Simulation ID string, e.g. "1021_1".
        window:     Time window index (0-4).
        timestep:   Timestep within the window (0-4).
        split:      "in" (t0-t4) or "out" (t5-t9).
        component:  "magnitude", "x", "y", or "z".
        y_slice:    Y coordinate of the XZ slice (default 0.0 = foil midplane).
        y_tol:      Half-thickness of the Y slice.
        grid_shape: (nx, nz) interpolation grid resolution.
        vmin/vmax:  Colourmap clipping (auto = 2nd/98th percentile).
        ax:         Optional existing Axes.

    Returns:
        fig, ax
    """
    if split not in ("in", "out"):
        raise ValueError('split must be "in" or "out"')
    if component not in ("magnitude", "x", "y", "z"):
        raise ValueError('component must be "magnitude", "x", "y", or "z"')

    data         = _load(sim_id, window)
    pos          = data["pos"]
    idcs_airfoil = data["idcs_airfoil"]
    vel          = data[f"velocity_{split}"][timestep]   # (N, 3)
    scalar       = _to_scalar(vel, component)

    cmap      = "turbo" if component == "magnitude" else "RdBu_r"
    cbar_lbl  = "|v| (m/s)" if component == "magnitude" else f"v_{component} (m/s)"

    grid, x_range, z_range = _interpolate_to_grid(
        pos, scalar, y_slice, y_tol, grid_shape
    )
    flat = grid[~np.isnan(grid)]
    _vmin = vmin if vmin is not None else np.percentile(flat, 2)
    _vmax = vmax if vmax is not None else np.percentile(flat, 98)

    foil_patches = _foil_patches(pos[idcs_airfoil][:, [0, 2]])

    created = ax is None
    fig, ax = (plt.subplots(figsize=(13, 5)) if created else (ax.get_figure(), ax))
    if created:
        fig.patch.set_facecolor("#0d0d0d")

    im = _draw_field(ax, grid, x_range, z_range, foil_patches, cmap, _vmin, _vmax)

    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.022)
    cbar.set_label(cbar_lbl, color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    t_idx = timestep if split == "in" else timestep + 5
    ax.set_title(
        f"sim {sim_id}  |  window {window}  |  t{t_idx} [{split}]  |  v ({component})",
        color="white", fontsize=12,
    )
    ax.set_xlabel("X (streamwise)", color="white", fontsize=10)
    ax.set_ylabel("Z (spanwise)",   color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    if created:
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# plot_poly_comparison
# ---------------------------------------------------------------------------

def plot_poly_comparison(
    sim_id: str,
    window: int = 0,
    degree: int = 2,
    component: str = "magnitude",
    y_slice: float = 0.0,
    y_tol: float = 0.012,
    grid_shape: tuple = (600, 300),
):
    """
    Side-by-side comparison of polynomial extrapolation vs ground truth, plus
    the pointwise error, for all 5 output timesteps (t5 … t9).

    Layout: 5 rows × 3 columns
        col 0 — Ground truth velocity    (colormap: "turbo")
        col 1 — Polynomial prediction    (colormap: "turbo", same scale as GT)
        col 2 — Pointwise error |pred−gt| (colormap: "YlOrRd", own scale)

    Foils are rendered as solid grey shapes on every panel.

    Args:
        sim_id:     Simulation ID, e.g. "1021_1".
        window:     Time window (0-4).
        degree:     Polynomial degree (default 2).
        component:  "magnitude", "x", "y", or "z".
        y_slice:    Y value of the XZ slice.
        y_tol:      Half-thickness of Y slice.
        grid_shape: Interpolation grid resolution (nx, nz).

    Returns:
        fig
    """
    import torch
    from models.transolver_residual.polynomial import poly_extrapolate

    data         = _load(sim_id, window)
    pos          = data["pos"]
    idcs_airfoil = data["idcs_airfoil"]
    vel_out_np   = data["velocity_out"]   # (5, N, 3)

    vel_in = torch.from_numpy(data["velocity_in"]).unsqueeze(0)   # (1,5,N,3)
    t      = torch.from_numpy(data["t"]).unsqueeze(0)             # (1,10)
    pred_np = poly_extrapolate(vel_in, t, degree=degree)[0].numpy()  # (5,N,3)

    foil_patches_fn = lambda: _foil_patches(pos[idcs_airfoil][:, [0, 2]])

    T_OUT = 5
    cmap_vel = "turbo"
    cmap_err = "YlOrRd"

    # Pre-compute all grids so we can set shared colour limits
    gt_grids, pred_grids, err_grids = [], [], []
    for step in range(T_OUT):
        gt_s   = _to_scalar(vel_out_np[step], component)
        pr_s   = _to_scalar(pred_np[step],    component)
        err_s  = np.linalg.norm(pred_np[step] - vel_out_np[step], axis=-1)

        gt_g,   xr, zr = _interpolate_to_grid(pos, gt_s,  y_slice, y_tol, grid_shape)
        pr_g,    _,  _ = _interpolate_to_grid(pos, pr_s,  y_slice, y_tol, grid_shape)
        err_g,   _,  _ = _interpolate_to_grid(pos, err_s, y_slice, y_tol, grid_shape)

        gt_grids.append(gt_g);   pred_grids.append(pr_g);   err_grids.append(err_g)

    # Shared colour range for velocity panels (GT + pred use same scale)
    all_vel = np.concatenate([g[~np.isnan(g)] for g in gt_grids + pred_grids])
    vel_vmin, vel_vmax = np.percentile(all_vel, 2), np.percentile(all_vel, 98)

    # Error colour range: 0 → 98th percentile across all steps
    all_err = np.concatenate([g[~np.isnan(g)] for g in err_grids])
    err_vmax = np.percentile(all_err, 98)

    # ---- build figure -------------------------------------------------------
    fig, axes = plt.subplots(
        T_OUT, 3,
        figsize=(18, T_OUT * 3.2),
        gridspec_kw={"wspace": 0.05, "hspace": 0.35},
    )
    fig.patch.set_facecolor("#0d0d0d")

    col_titles = [
        f"Ground truth  (v {component})",
        f"Poly prediction  (deg {degree})",
        "Pointwise error  ||pred − gt||",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, color="white", fontsize=13, pad=8)

    for step in range(T_OUT):
        fp = foil_patches_fn()
        fp2 = foil_patches_fn()
        fp3 = foil_patches_fn()

        _draw_field(axes[step, 0], gt_grids[step],   xr, zr, fp,  cmap_vel, vel_vmin, vel_vmax)
        _draw_field(axes[step, 1], pred_grids[step], xr, zr, fp2, cmap_vel, vel_vmin, vel_vmax)
        im_err = _draw_field(axes[step, 2], err_grids[step], xr, zr, fp3, cmap_err, 0, err_vmax)

        axes[step, 0].set_ylabel(f"t{step + 5}", color="white", fontsize=12, labelpad=4)

        for col in range(3):
            axes[step, col].tick_params(colors="white", labelsize=7)
            axes[step, col].set_xlabel("")
            for spine in axes[step, col].spines.values():
                spine.set_edgecolor("#333333")

    # Shared colourbars — one for velocity, one for error
    cbar_ax_vel = fig.add_axes([0.35, 0.02, 0.28, 0.012])
    cbar_ax_err = fig.add_axes([0.68, 0.02, 0.28, 0.012])

    sm_vel = plt.cm.ScalarMappable(cmap=cmap_vel, norm=Normalize(vel_vmin, vel_vmax))
    sm_err = plt.cm.ScalarMappable(cmap=cmap_err, norm=Normalize(0, err_vmax))

    cb_vel = fig.colorbar(sm_vel, cax=cbar_ax_vel, orientation="horizontal")
    cb_err = fig.colorbar(sm_err, cax=cbar_ax_err, orientation="horizontal")

    lbl = "|v| (m/s)" if component == "magnitude" else f"v_{component} (m/s)"
    cb_vel.set_label(lbl,               color="white", fontsize=10)
    cb_err.set_label("||pred − gt||",   color="white", fontsize=10)
    for cb in (cb_vel, cb_err):
        cb.ax.xaxis.set_tick_params(color="white")
        plt.setp(cb.ax.xaxis.get_ticklabels(), color="white")

    fig.suptitle(
        f"Polynomial extrapolation (deg {degree}) vs ground truth  —  "
        f"sim {sim_id}  window {window}",
        color="white", fontsize=14, y=0.995,
    )
    return fig


# ---------------------------------------------------------------------------
# make_velocity_video
# ---------------------------------------------------------------------------

def _foil_patches(surf_xz: np.ndarray) -> list:
    """Return one matplotlib Polygon per foil (detected by Z gaps), XZ plane."""
    z = surf_xz[:, 1]
    z_sorted_unique = np.sort(np.unique(np.round(z, 3)))
    gaps = np.where(np.diff(z_sorted_unique) > 0.02)[0]
    boundaries = [z.min()] + list(z_sorted_unique[gaps + 1]) + [z.max() + 1e-6]

    patches = []
    for z0, z1 in zip(boundaries[:-1], boundaries[1:]):
        mask = (z >= z0) & (z < z1)
        pts = surf_xz[mask]
        if len(pts) < 4:
            continue
        hull = ConvexHull(pts)
        poly = Polygon(pts[hull.vertices], closed=True)
        patches.append(poly)
    return patches


def make_velocity_video(
    sim_id: str,
    window: int = 0,
    component: str = "magnitude",
    y_slice: float = 0.0,
    y_tol: float = 0.012,
    grid_shape: tuple = (500, 250),
    fps: int = 3,
    cmap: str = "jet",
    output_path: str | None = None,
):
    """
    Render a GIF video of the velocity field over all 10 timesteps (5 in + 5 out)
    for one simulation window. Uses a top-down XZ plan view.

    The scattered 3D points are sliced at Y ≈ y_slice ± y_tol and interpolated
    onto a regular XZ grid for smooth rendering (similar to CFD post-processing
    software output).

    Args:
        sim_id:      Simulation ID string, e.g. "1021_1".
        window:      Time window index (0-4).
        component:   "magnitude", "x", "y", or "z".
        y_slice:     Y value to slice at (foil midplane = 0.0).
        y_tol:       Half-thickness of Y slice (default ±0.012 ≈ 1 mesh layer).
        grid_shape:  (nx, nz) resolution of the interpolated grid.
        fps:         Frames per second in the output GIF.
        cmap:        Matplotlib colormap name (default "jet" matches CFD style).
        output_path: Where to save the GIF. Defaults to
                     ./visualizations/{sim_id}_w{window}_{component}.gif

    Returns:
        Path to the saved GIF.
    """
    data = _load(sim_id, window)
    pos = data["pos"]               # (100000, 3)
    idcs_airfoil = data["idcs_airfoil"]

    # Y-slice mask — applied once, shared across all frames
    y_mask = np.abs(pos[:, 1] - y_slice) < y_tol
    pos_slice = pos[y_mask]         # (M, 3)
    xz = pos_slice[:, [0, 2]]      # (M, 2)

    # Regular XZ grid
    x_range = (pos[:, 0].min(), pos[:, 0].max())
    z_range = (pos[:, 2].min(), pos[:, 2].max())
    xi = np.linspace(*x_range, grid_shape[0])
    zi = np.linspace(*z_range, grid_shape[1])
    Xi, Zi = np.meshgrid(xi, zi)   # (nz, nx)

    # Stack all 10 timesteps
    vel_in  = data["velocity_in"]   # (5, 100000, 3)
    vel_out = data["velocity_out"]  # (5, 100000, 3)
    all_vel = np.concatenate([vel_in, vel_out], axis=0)  # (10, 100000, 3)

    ci = {"magnitude": None, "x": 0, "y": 1, "z": 2}[component]

    print(f"Interpolating {len(all_vel)} frames onto {grid_shape} grid …")
    frames = []
    for t, vel_t in enumerate(all_vel):
        v = vel_t[y_mask]
        scalar = np.linalg.norm(v, axis=-1) if ci is None else v[:, ci]
        grid = griddata(xz, scalar, (Xi, Zi), method="linear")
        frames.append(grid)
        print(f"  frame {t+1}/10", end="\r")
    print()

    # Global colormap range across all frames
    all_vals = np.concatenate([f[~np.isnan(f)] for f in frames])
    vmin, vmax = np.percentile(all_vals, 2), np.percentile(all_vals, 98)

    # Airfoil planform patches (convex hull per foil in XZ)
    surf_xz = pos[idcs_airfoil][:, [0, 2]]
    foil_patches = _foil_patches(surf_xz)

    # Build figure
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    img = ax.imshow(
        frames[0],
        origin="lower",
        extent=[*x_range, *z_range],
        aspect="auto",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation="bilinear",
    )

    # Foils as filled gray polygons with white outline
    pc = PatchCollection(foil_patches, facecolor="#555555", edgecolor="white",
                         linewidths=0.5, zorder=5)
    ax.add_collection(pc)

    cbar = fig.colorbar(img, ax=ax, pad=0.01, fraction=0.02)
    cbar_label = "|v| (m/s)" if component == "magnitude" else f"v_{component} (m/s)"
    cbar.set_label(cbar_label, color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("X (streamwise)", color="white", fontsize=11)
    ax.set_ylabel("Z (spanwise)",   color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    title = ax.set_title("", color="white", fontsize=12)
    fig.tight_layout()

    split_labels = ["in"] * 5 + ["out"] * 5

    def update(frame_idx):
        img.set_data(frames[frame_idx])
        split = split_labels[frame_idx]
        title.set_text(
            f"sim {sim_id}  |  window {window}  |  t{frame_idx} [{split}]  "
            f"|  velocity ({component})"
        )
        return img, title

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=False
    )

    if output_path is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "visualizations")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{sim_id}_w{window}_{component}.gif")

    ani.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# plot_poly_error_3d  — PyVista interactive 3-D error visualisation
# ---------------------------------------------------------------------------

def plot_poly_error_3d(
    sim_id: str,
    window: int = 0,
    degree: int = 2,
    subsample: int = 40_000,
    screenshot: str | None = None,
):
    """
    Show a 5-panel interactive 3-D point cloud where each point is coloured by
    the pointwise velocity error  ||poly_pred - gt||  at each of the 5 output
    timesteps (t5 … t9).

    The airfoil surface is overlaid in solid grey.  You can rotate/zoom each
    panel independently.

    Args:
        sim_id:     Simulation ID, e.g. "1021_1".
        window:     Time window index (0-4).
        degree:     Polynomial degree used for extrapolation (default 2).
        subsample:  Fluid points to render per panel (100k is fine but slower).
        screenshot: If given, save a PNG to this path instead of opening the
                    interactive window (useful on headless servers).

    Requirements:
        pip install pyvista   ← already in requirements.txt / venv
    """
    import pyvista as pv
    import torch
    from models.transolver_residual.polynomial import poly_extrapolate

    # ---- load data --------------------------------------------------------
    data = _load(sim_id, window)
    pos          = data["pos"]              # (N, 3)
    idcs_airfoil = data["idcs_airfoil"]
    vel_in       = torch.from_numpy(data["velocity_in"]).unsqueeze(0)   # (1,5,N,3)
    vel_out      = torch.from_numpy(data["velocity_out"])               # (5,N,3)
    t            = torch.from_numpy(data["t"]).unsqueeze(0)             # (1,10)

    # polynomial prediction at output times
    pred = poly_extrapolate(vel_in, t, degree=degree)[0]  # (5,N,3)

    # per-point, per-step error magnitude
    error = (pred - vel_out).norm(dim=-1).numpy()         # (5,N)

    # ---- subsampling ------------------------------------------------------
    rng = np.random.default_rng(0)
    is_airfoil = np.zeros(len(pos), dtype=bool)
    is_airfoil[idcs_airfoil] = True
    fluid_idcs = np.where(~is_airfoil)[0]
    if len(fluid_idcs) > subsample:
        fluid_idcs = rng.choice(fluid_idcs, size=subsample, replace=False)

    fluid_pos  = pos[fluid_idcs]          # (M, 3)
    airfoil_pos = pos[idcs_airfoil]       # (K, 3)

    # global error range for a consistent colormap across all panels
    err_vmin = 0.0
    err_vmax = float(np.percentile(error[:, fluid_idcs], 98))

    # ---- plotter ----------------------------------------------------------
    off_screen = screenshot is not None
    pl = pv.Plotter(
        shape=(1, 5),
        window_size=[1800, 420],
        off_screen=off_screen,
        border=False,
    )
    pl.set_background("black")

    surf_cloud = pv.PolyData(airfoil_pos)

    for step in range(5):
        pl.subplot(0, step)

        fluid_cloud = pv.PolyData(fluid_pos)
        fluid_cloud["error"] = error[step, fluid_idcs]

        pl.add_mesh(
            fluid_cloud,
            scalars="error",
            cmap="hot_r",
            clim=[err_vmin, err_vmax],
            point_size=2.0,
            render_points_as_spheres=False,
            show_scalar_bar=(step == 4),
            scalar_bar_args=dict(
                title="||v_pred - v_gt||",
                color="white",
                title_font_size=12,
                label_font_size=10,
            ),
        )
        pl.add_mesh(surf_cloud, color="#888888", point_size=2.5,
                    render_points_as_spheres=False, opacity=0.9)
        pl.add_title(f"t{step + 5}  (poly deg={degree})", font_size=10, color="white")
        pl.camera_position = "xz"   # top-down plan view by default

    pl.add_text(
        f"Polynomial extrapolation error — sim {sim_id}  window {window}",
        position="upper_edge", font_size=11, color="white",
    )

    if screenshot:
        pl.screenshot(screenshot)
        print(f"Screenshot saved → {screenshot}")
    else:
        pl.show(title=f"Poly error — {sim_id}")

    pl.close()
