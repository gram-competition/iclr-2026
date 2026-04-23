"""Single submission model: fmgreco_stnn (SpatioTemporalGNN).

Loss / optim setup lives under ``src/training/pointcloud_losses.py`` (vendored from
the original MLP training helpers). ``models/`` only contains this operator.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SpatioTemporalGNN",
    "CANONICAL_MODEL_REGISTRY",
    "MODEL_ALIASES",
    "MODEL_REGISTRY",
    "get_model_class",
    "normalise_model_name",
]

from .fmgreco_stnn import SpatioTemporalGNN

CANONICAL_MODEL_REGISTRY: dict[str, type] = {
    "spatio_temporal_gnn": SpatioTemporalGNN,
}

MODEL_ALIASES: dict[str, str] = {
    "stnn": "spatio_temporal_gnn",
    "fmgreco_stnn": "spatio_temporal_gnn",
    "fmgreco-stnn": "spatio_temporal_gnn",
    "mlp": "spatio_temporal_gnn",
    "gated_egno_mean_res": "spatio_temporal_gnn",
    "gated_egno": "spatio_temporal_gnn",
    "gegno": "spatio_temporal_gnn",
}

MODEL_REGISTRY: dict[str, type] = {
    **CANONICAL_MODEL_REGISTRY,
    **{
        alias: CANONICAL_MODEL_REGISTRY[target]
        for alias, target in MODEL_ALIASES.items()
        if target in CANONICAL_MODEL_REGISTRY
    },
}


def normalise_model_name(model_name: str) -> str:
    key = model_name.strip().lower().replace("-", "_")
    return MODEL_ALIASES.get(key, key)


def get_model_class(model_name: str) -> type:
    normalised = normalise_model_name(model_name)
    if normalised not in CANONICAL_MODEL_REGISTRY:
        available = ", ".join(sorted(CANONICAL_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model_name={model_name!r} (normalised={normalised!r}). "
            f"Available: {available}."
        ) from None
    return CANONICAL_MODEL_REGISTRY[normalised]
