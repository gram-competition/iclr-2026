"""Wide name → class map for ``GRAM_MODEL`` / ``main.py`` smoke (STNN-only branch)."""

from __future__ import annotations

from typing import Type

from torch.nn import Module

_ALIASES: dict[str, str] = {
    "stnn": "spatio_temporal_gnn",
    "fmgreco_stnn": "spatio_temporal_gnn",
    "fmgreco-stnn": "spatio_temporal_gnn",
    # allow common smoke defaults to resolve without errors on this single-model branch
    "mlp": "spatio_temporal_gnn",
    "gated_egno_mean_res": "spatio_temporal_gnn",
    "gated_egno": "spatio_temporal_gnn",
    "gegno": "spatio_temporal_gnn",
    "pr10": "spatio_temporal_gnn",
}


def _normalise_key(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def get_model_class(name: str) -> Type[Module]:
    key = _normalise_key(name)
    key = _ALIASES.get(key, key)
    if key != "spatio_temporal_gnn":
        raise KeyError(
            f"Unknown model {name!r}. This branch only ships SpatioTemporalGNN. "
            f"Try: GRAM_MODEL=spatio_temporal_gnn (aliases: stnn, fmgreco_stnn, mlp for smoke)."
        )
    from models.fmgreco_stnn import SpatioTemporalGNN

    return SpatioTemporalGNN


def list_models() -> list[str]:
    return sorted(_ALIASES)
