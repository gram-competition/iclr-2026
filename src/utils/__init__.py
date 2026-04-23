"""Common utility modules for the refactored training stack."""

from .config import (
	apply_parser_defaults_from_config,
	load_yaml_config,
	read_config_defaults_from_cli,
)
from .training_history import TrainingEpochRecord, persist_training_artifacts
from .vrt_style_inference import (
	apply_vrt_style_inference,
	forward_with_y_reflection_tta,
	persistence_prediction,
	reflect_y_inputs,
	should_use_persistence_fallback,
	unreflect_y_prediction,
)

__all__ = [
	"apply_parser_defaults_from_config",
	"load_yaml_config",
	"read_config_defaults_from_cli",
	"TrainingEpochRecord",
	"persist_training_artifacts",
	"apply_vrt_style_inference",
	"forward_with_y_reflection_tta",
	"persistence_prediction",
	"reflect_y_inputs",
	"should_use_persistence_fallback",
	"unreflect_y_prediction",
]
