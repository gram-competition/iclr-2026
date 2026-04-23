"""
Local inference check (4-tensor competition signature).

    from models.fmgreco_stnn import SpatioTemporalGNN
    model = SpatioTemporalGNN()
    velocity_out = model(t, pos, idcs_airfoil, velocity_in)

Evaluates RL2 on ``1021_1-0.npz`` if present in the working directory.
Optional weights: ``models/fmgreco_stnn/state_dict.pt`` (loaded in the model
constructor when the file exists).
"""

import os

import numpy as np
import torch

from models.fmgreco_stnn import SpatioTemporalGNN


def rl2(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return ((pred - gt).norm() / (gt.norm() + 1e-8)).item()


# ── Load sample ──────────────────────────────────────────────────────────────
d = np.load("1021_1-0.npz", allow_pickle=True)

t = torch.from_numpy(d["t"].astype(np.float32)).unsqueeze(0)
pos = torch.from_numpy(d["pos"].astype(np.float32)).unsqueeze(0)
velocity_in = torch.from_numpy(d["velocity_in"].astype(np.float32)).unsqueeze(0)
velocity_out_gt = torch.from_numpy(d["velocity_out"].astype(np.float32))
idcs_airfoil = [torch.from_numpy(d["idcs_airfoil"].astype(np.int64))]

model = SpatioTemporalGNN()
model.eval()
_w = os.path.join(os.path.dirname(__file__), "models", "fmgreco_stnn", "state_dict.pt")
print(
    f"Optional checkpoint: {_w} (exists={os.path.isfile(_w)}; loaded in __init__ if present)\n"
)

# ── Run inference with the bare 4-argument signature ─────────────────────────
with torch.no_grad():
    velocity_out = model(t, pos, idcs_airfoil, velocity_in)

velocity_out = velocity_out.squeeze(0)

# ── Metrics ─────────────────────────────────────────────────────────────────
print("=== RL2 per timestep ===")
for i in range(5):
    print(f"  t+{i + 1}: {rl2(velocity_out[i], velocity_out_gt[i]):.4f}")

overall = rl2(velocity_out, velocity_out_gt)
persistence = rl2(
    velocity_in.squeeze(0)[-1].unsqueeze(0).expand(5, -1, -1),
    velocity_out_gt,
)
print(f"\nOverall RL2:          {overall:.4f}")
print(f"Persistence RL2:      {persistence:.4f}")
print(f"Improvement:          {(persistence - overall) / persistence * 100:.1f}%")
