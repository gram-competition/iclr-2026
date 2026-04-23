from __future__ import annotations

import json
import os
import sys

import torch

# Repo layout: run from repository root with `python smoke_test_submission.py`
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.fmgreco_stnn import SpatioTemporalGNN


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpatioTemporalGNN().to(device)
    model.eval()
    b, n = 1, 100000
    t = torch.linspace(0.0, 0.9, 10, device=device).view(1, 10)
    pos = torch.rand(b, n, 3, device=device)
    velocity_in = torch.randn(b, 5, n, 3, device=device)
    idcs_airfoil = [torch.arange(0, 1000, device=device)]
    with torch.no_grad():
        pred = model(t, pos, idcs_airfoil, velocity_in)
    print(
        json.dumps(
            {
                "pred_shape": list(pred.shape),
                "pred_abs_mean": float(pred.abs().mean().item()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
