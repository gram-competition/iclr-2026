from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelInputs:
    """Normalized tensors passed through the submission model."""

    t: torch.Tensor
    pos: torch.Tensor
    idcs_airfoil: list[torch.Tensor]
    velocity_in: torch.Tensor
    domain_min: torch.Tensor
    domain_max: torch.Tensor


@dataclass(frozen=True)
class VoxelGridSpec:
    """Specification for point-to-voxel rasterization."""

    grid_size: int = 48
    padding: float = 0.05
    smooth_sigma: float = 0.8
    normalize_mass: bool = True


@dataclass(frozen=True)
class ScatteringFeatureConfig:
    """Configuration for runtime wavelet conditioning features."""

    grid_size: int = 48
    padding: float = 0.05
    smooth_sigma: float = 0.8
    j: int = 3
    l: int = 3
    sigma_0: float = 2.0
    max_order: int = 2
    integral_powers: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    map_pooled_size: int = 12
    use_canonical_bounds: bool = True
    velocity_channel_mixing: tuple[str, ...] = ("vx", "vy", "vz", "speed")
