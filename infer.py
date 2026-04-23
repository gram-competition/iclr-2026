"""
Local inference check (4-tensor competition signature).

    from models.fmgreco_stnn import SpatioTemporalGNN
    model = SpatioTemporalGNN()
    velocity_out = model(t, pos, idcs_airfoil, velocity_in)

Evaluates RL2 on ``1021_1-0.npz`` if present in the working directory.

Weights: ``models/fmgreco_stnn/state_dict.pt`` — use
``scripts/export_best_for_submission.py`` on ``best.pt`` so the file includes
``model_state_dict`` and ``args`` (architecture must match training, e.g. ``k``).
Forward applies hard no-slip in ``SpatioTemporalGNN`` (``apply_hard_no_slip``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from models.fmgreco_stnn import SpatioTemporalGNN
from src.evaluation.inference import (
    _build_spatio_temporal_gnn,
    _merge_stnn_args_from_checkpoint,
)
from src.data import (
    compute_velocity_standardization,
    load_pressure_inputs_from_npz,
    scale_velocity,
    unscale_velocity_batch,
)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rl2(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return ((pred - gt).norm() / (gt.norm() + 1e-8)).item()


# ── Load sample ──────────────────────────────────────────────────────────────
d = np.load("1021_1-0.npz", allow_pickle=True)

t = torch.from_numpy(d["t"].astype(np.float32)).unsqueeze(0)
pos = torch.from_numpy(d["pos"].astype(np.float32)).unsqueeze(0)
velocity_in_raw = torch.from_numpy(d["velocity_in"].astype(np.float32))
velocity_out_gt = torch.from_numpy(d["velocity_out"].astype(np.float32))
idcs_airfoil = [torch.from_numpy(d["idcs_airfoil"].astype(np.int64))]

velocity_mean, velocity_std = compute_velocity_standardization(
    velocity_in_raw,
    eps=1e-6,
)
velocity_in = scale_velocity(velocity_in_raw, velocity_mean, velocity_std).unsqueeze(0)

pressure_raw = load_pressure_inputs_from_npz(
    d,
    t_in=int(velocity_in_raw.shape[0]),
    num_points=int(pos.shape[1]),
)
pm = pressure_raw.mean()
ps = pressure_raw.std(unbiased=False).clamp_min(1e-6)
pressure_in = ((pressure_raw - pm) / ps).unsqueeze(0)

t = t.to(_device)
pos = pos.to(_device)
velocity_in = velocity_in.to(_device)
idcs_airfoil = [idcs_airfoil[0].to(_device)]
pressure_in = pressure_in.to(_device)

_repo_root = Path(__file__).resolve().parent
_w = _repo_root / "models" / "fmgreco_stnn" / "state_dict.pt"


def _infer_base_args() -> argparse.Namespace:
    """Defaults aligned with ``train_config.yaml`` when checkpoint has no ``args``."""
    return argparse.Namespace(
        latent_dim=128,
        num_blocks=12,
        num_heads=4,
        num_temporal_layers=2,
        k=16,
        use_so3_backbone=False,
        use_steerable_backbone=False,
        steerable_lmax=2,
        use_pressure=True,
        use_relative_velocity_inputs=True,
    )


if _w.is_file():
    _ck = torch.load(_w, map_location=_device, weights_only=False)
    if isinstance(_ck, dict) and "model_state_dict" in _ck:
        _merged = _merge_stnn_args_from_checkpoint(_infer_base_args(), _ck)
        model = _build_spatio_temporal_gnn(
            _merged, _device, load_bundled_weights=False
        )
        model.load_state_dict(_ck["model_state_dict"], strict=True)
        print(f"Loaded submission checkpoint: {_w} (architecture from checkpoint['args'])\n")
    else:
        model = SpatioTemporalGNN(load_bundled_weights=False)
        model.load_state_dict(_ck, strict=True)
        print(f"Loaded raw state_dict: {_w} (using default architecture — may mismatch)\n")
else:
    model = SpatioTemporalGNN()
    print(f"No {_w}; using random init + optional bundled weights.\n")

model.eval()
print(f"Inference device: {_device}\n")

# ── Run inference with the bare 4-argument signature ─────────────────────────
with torch.no_grad():
    velocity_out = model(
        t,
        pos,
        idcs_airfoil,
        velocity_in,
        pressure_in=pressure_in,
    )

velocity_out = velocity_out.squeeze(0).cpu()

vm = velocity_mean.reshape(1, -1)
vs = velocity_std.reshape(1, -1)
pred_phys = unscale_velocity_batch(velocity_out.unsqueeze(0), vm, vs).squeeze(0)
# unscale can broadcast to (1, 1, N, 3); collapse to (N, 3) for the persistence baseline.
# Force CPU for all operands (model may run on CUDA; vm/vs must match the velocity slice).
last_phys = unscale_velocity_batch(
    velocity_in.squeeze(0)[-1].unsqueeze(0).cpu(),
    vm.cpu(),
    vs.cpu(),
).reshape(-1, 3)

# ── Metrics (physical space) ────────────────────────────────────────────────
print("=== RL2 per timestep ===")
for i in range(5):
    print(f"  t+{i + 1}: {rl2(pred_phys[i], velocity_out_gt[i]):.4f}")

overall = rl2(pred_phys, velocity_out_gt)
persistence = rl2(
    last_phys.unsqueeze(0).expand(5, -1, -1),
    velocity_out_gt,
)
print(f"\nOverall RL2:          {overall:.4f}")
print(f"Persistence RL2:      {persistence:.4f}")
print(f"Improvement:          {(persistence - overall) / persistence * 100:.1f}%")
