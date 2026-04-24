from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence


def load_yaml_config(config_path: str | Path) -> dict[str, object]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML config loading requires PyYAML. Install it with 'pip install pyyaml'."
        ) from exc

    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj)

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a top-level mapping of key/value pairs.")
    return {str(key): value for key, value in payload.items()}


def read_config_defaults_from_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", type=str, default="")
    bootstrap_args, _ = bootstrap_parser.parse_known_args(argv)
    if not bootstrap_args.config:
        return {}
    return load_yaml_config(bootstrap_args.config)


def apply_parser_defaults_from_config(
    parser: argparse.ArgumentParser,
    config_defaults: Mapping[str, object],
    *,
    strict_unknown_keys: bool,
) -> None:
    if not config_defaults:
        return

    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(config_defaults).difference(valid_keys))
    if unknown_keys and strict_unknown_keys:
        parser.error("Unknown key(s) in config file: " + ", ".join(unknown_keys))

    filtered_defaults = {
        key: value for key, value in config_defaults.items() if key in valid_keys
    }
    if filtered_defaults:
        parser.set_defaults(**filtered_defaults)
