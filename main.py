from __future__ import annotations

import os
import sys
from typing import Any

# Training stack and dataset helpers (imported for API re-exports, not for smoke)
from src.data import (
    WarpedIFWDataset,
    build_loader as _build_loader,
    compute_velocity_standardization as _compute_velocity_standardization,
    resolve_overfit_file,
    scale_velocity as _scale_velocity,
    split_train_val_test,
    unscale_velocity_batch as _unscale_velocity_batch,
)
from src.training import (
    NUM_POS,
    NUM_T_IN,
    NUM_T_OUT,
    evaluate,
    hint_metric,
    parse_args as _parse_args,
    run_full_test_inference,
    set_seed as _set_seed,
)
from src.training import trainer as _trainer
from models import SpatioTemporalGNN, get_model_class

Model = SpatioTemporalGNN


def set_seed(seed: int) -> None:
    _set_seed(seed)


def parse_args(argv: list[str] | None = None):
    return _parse_args(argv)


def train(args: Any) -> None:
    # Notebook compatibility: if callers rebind `main.Model`, use that class.
    _trainer.Model = Model
    _trainer.train(args)


def run_smoke() -> None:
    """Quick `GRAM_MODEL` / registry sanity check (default when no CLI args)."""
    import torch

    from models.registry import get_model_class, list_models

    # Default aligns with Agrover112 / PR #10 (gated EGNO) submission; override with
    # GRAM_SMOKE_MODEL=mlp or GRAM_MODEL=...
    default = os.environ.get("GRAM_SMOKE_MODEL", "spatio_temporal_gnn")
    which = os.environ.get("GRAM_MODEL", default)
    try:
        ModelCls = get_model_class(which)
    except KeyError:
        print(
            f"Warning: unknown GRAM_MODEL={which!r}; falling back to {default!r}. "
            f"Known: {', '.join(list_models())}"
        )
        ModelCls = get_model_class(default)
    model = ModelCls()
    model.eval()

    batch_size = 95
    num_t_in, num_t_out, num_pos = 5, 5, 100000

    t = torch.rand((batch_size, num_t_in + num_t_out))
    pos = torch.rand((batch_size, num_pos, 3))
    idcs_airfoil = [
        torch.randint(num_pos, size=(n,))
        for n in torch.randint(3142, 24198, size=(batch_size,))
    ]
    velocity_in = torch.rand((batch_size, num_t_in, num_pos, 3))
    ground_truth = torch.rand((batch_size, num_t_out, num_pos, 3))

    velocity_out = model(t, pos, idcs_airfoil, velocity_in)
    assert velocity_out.shape == (batch_size, num_t_out, num_pos, 3)

    metric = (velocity_out - ground_truth).norm(dim=3).mean(dim=(1, 2))
    print(
        f"Model={ModelCls.__name__} (GRAM_MODEL={os.environ.get('GRAM_MODEL', default)!r}) | "
        f"Metric: {metric.mean():.4f} +- {metric.std():.4f}"
    )


__all__ = [
    "Model",
    "SpatioTemporalGNN",
    "NUM_POS",
    "NUM_T_IN",
    "NUM_T_OUT",
    "WarpedIFWDataset",
    "_build_loader",
    "_compute_velocity_standardization",
    "_scale_velocity",
    "_unscale_velocity_batch",
    "evaluate",
    "get_model_class",
    "hint_metric",
    "parse_args",
    "resolve_overfit_file",
    "run_full_test_inference",
    "run_smoke",
    "set_seed",
    "split_train_val_test",
    "train",
]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_smoke()
    else:
        train(parse_args())
