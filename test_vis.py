from utils.visualize import (
    plot_geometry,
    plot_velocity_field,
    plot_poly_comparison,
    plot_poly_error_3d,
    make_velocity_video,
)
import matplotlib.pyplot as plt

sim_id = "1025_1"

# ── Geometry (point-cloud, XZ plan view) ────────────────────────────────────
fig, ax = plot_geometry(sim_id, projection="xz")
plt.show()

# ── Single velocity frame  (CFD-style interpolated, dark background) ────────
fig, ax = plot_velocity_field(sim_id, window=0, timestep=2, split="in",  component="magnitude")
fig, ax = plot_velocity_field(sim_id, window=0, timestep=0, split="out", component="x")
plt.show()

# ── Polynomial comparison: GT | prediction | error  (5 output timesteps) ────
fig = plot_poly_comparison(sim_id, window=0, degree=2, component="magnitude")
plt.show()

# ── 3-D PyVista error visualisation (5 panels, interactive) ─────────────────
# On headless / SSH: pass screenshot="out.png"
plot_poly_error_3d(sim_id, window=0, degree=2)

# ── GIF video (uncomment to generate) ───────────────────────────────────────
# make_velocity_video(sim_id, window=0)
