from utils.visualize import (
    plot_geometry,
    plot_velocity_field,
    plot_poly_comparison,
    plot_poly_error_3d,
    make_velocity_video,
)
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Save geometry plot for every unique geometry family ─────────────────────
# Naming: {prefix}_{variant}-{window}.npz  — the geometry (pos, idcs_airfoil)
# is the same for all variants sharing the same prefix (e.g. "1021").
# We pick the lexicographically first simulation in each family and save
# an XZ plan-view to geometries/{prefix}.png

DATA_DIR = "gram_data"
OUT_DIR  = "geometries"
os.makedirs(OUT_DIR, exist_ok=True)

# Collect one representative sim_id per prefix
all_npz   = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
real_npz  = [f for f in all_npz if ".distcache." not in f and ".knncache." not in f]
sim_ids   = sorted({os.path.basename(f).rsplit("-", 1)[0] for f in real_npz})

# Group by prefix (everything before the first underscore)
prefix_to_sim = {}
for sid in sim_ids:
    prefix = sid.split("-")[0]
    if prefix not in prefix_to_sim:
        prefix_to_sim[prefix] = sid   # first one wins (alphabetically sorted)

print(f"Found {len(prefix_to_sim)} unique geometry families: {sorted(prefix_to_sim)}")

for prefix, sim_id in sorted(prefix_to_sim.items()):
    out_path = os.path.join(OUT_DIR, f"{prefix}.png")
    print(f"  {prefix}  →  sim {sim_id}  →  {out_path}")
    fig, ax = plot_geometry(sim_id, projection="xz")
    ax.set_title(f"Geometry family {prefix}  (sim {sim_id})  [XZ view]", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"\nDone. {len(prefix_to_sim)} images saved to ./{OUT_DIR}/")


# ── Interactive / one-off plots below (comment/uncomment as needed) ──────────
# matplotlib.use("TkAgg")   # switch back to interactive backend if you need plt.show()

# sim_id = "1025_1"

# fig, ax = plot_geometry(sim_id, projection="xz")
# plt.show()

# fig, ax = plot_velocity_field(sim_id, window=0, timestep=2, split="in",  component="magnitude")
# fig, ax = plot_velocity_field(sim_id, window=0, timestep=0, split="out", component="x")
# plt.show()

# fig = plot_poly_comparison(sim_id, window=0, degree=2, component="magnitude")
# plt.show()

# plot_poly_error_3d(sim_id, window=0, degree=2)

# make_velocity_video(sim_id, window=0)
