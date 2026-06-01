from __future__ import annotations

from .ab_upt import ABUPT
from .aero_chrono_mixer import AeroChronoMixer
from .airformer import AirFormer
from .cdf_2grid import CDFDoubleGridNet
from .delta_graph import DeltaGraph
from .ensemble_spatiotemporal_models import EnsembleSpatioTemporalModels
from .finite_graph_v4 import FiniteGraphV4
from .fno_dse_time import FNO_DSE_TIME
from .gated_egno import GatedEGNOMeanResModel
from .harshitsinghsnu import ImprovedMLP
from .kagent import Kagent
from .levers_tail_submission import LeversTailV2Submission
from .mlp import MLP
from .transolver_ar import TransolverAR
from .perceiver_flow import PerceiverFlow
from .smoothsplatnet import SmoothSplatNet
from .submission_model import SubmissionModel
from .transolver_corrector import TransolverCorrector
from .transolver_residual import TransolverResidual
from .vrt_ensemble import VRTEnsemble
from .wavelet_latent_operator import WaveletLatentOperator
from .zonal_moe.wrapper import Model as ZonalMoe

from .spatiotemporal_mno import SpatiotemporalMNO

# Registry used by this team's training/evaluation tooling (src/). It is scoped
# to our own models; the flat imports above expose every competition submission.
CANONICAL_MODEL_REGISTRY = {
    "mlp": MLP,
    "spatiotemporal_mno": SpatiotemporalMNO,
}

MODEL_ALIASES = {
    "stmno": "spatiotemporal_mno",
    "st_mno": "spatiotemporal_mno",
    "spatiotemporal-mno": "spatiotemporal_mno",
}

MODEL_REGISTRY = {
    **CANONICAL_MODEL_REGISTRY,
    **{alias: CANONICAL_MODEL_REGISTRY[target] for alias, target in MODEL_ALIASES.items()},
}


def normalise_model_name(model_name: str) -> str:
    key = model_name.strip().lower()
    return MODEL_ALIASES.get(key, key)


def get_model_class(model_name: str):
    normalised = normalise_model_name(model_name)
    try:
        return CANONICAL_MODEL_REGISTRY[normalised]
    except KeyError as exc:
        available = ", ".join(sorted(CANONICAL_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model_name='{model_name}'. Available models: {available}."
        ) from exc


__all__ = [
    "CANONICAL_MODEL_REGISTRY",
    "MLP",
    "MODEL_ALIASES",
    "MODEL_REGISTRY",
    "SpatiotemporalMNO",
    "get_model_class",
    "normalise_model_name",
]
