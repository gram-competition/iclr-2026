from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F

from .types import ModelInputs

TIME_FEATURE_DIM = 10


def _empty_airfoil_indices(batch_size: int, device: torch.device) -> list[torch.Tensor]:
    return [torch.empty(0, dtype=torch.long, device=device) for _ in range(batch_size)]


def normalize_airfoil_indices(
    idcs_airfoil: Sequence[torch.Tensor] | torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Return one long index tensor per batch element."""

    if idcs_airfoil is None:
        return _empty_airfoil_indices(batch_size, device)

    if isinstance(idcs_airfoil, torch.Tensor):
        if idcs_airfoil.ndim == 1:
            return [idcs_airfoil.to(device=device, dtype=torch.long) for _ in range(batch_size)]
        if idcs_airfoil.ndim == 2:
            if idcs_airfoil.shape[0] != batch_size:
                raise ValueError(f"Expected {batch_size} airfoil rows, got {idcs_airfoil.shape[0]}")
            return [row.to(device=device, dtype=torch.long) for row in idcs_airfoil]
        raise ValueError(f"Unsupported idcs_airfoil shape: {tuple(idcs_airfoil.shape)}")

    if len(idcs_airfoil) != batch_size:
        raise ValueError(f"Expected {batch_size} airfoil tensors, got {len(idcs_airfoil)}")
    return [torch.as_tensor(indices, device=device, dtype=torch.long) for indices in idcs_airfoil]


def normalize_submission_inputs(
    t: torch.Tensor,
    pos: torch.Tensor,
    idcs_airfoil: Sequence[torch.Tensor] | torch.Tensor | None,
    velocity_in: torch.Tensor,
) -> ModelInputs:
    """Normalize tensors from the GRaM submission forward signature."""

    pos = torch.as_tensor(pos)
    velocity_in = torch.as_tensor(velocity_in, device=pos.device)
    t = torch.as_tensor(t, device=pos.device, dtype=pos.dtype)
    idcs_list = normalize_airfoil_indices(idcs_airfoil, pos.shape[0], pos.device)

    if t.ndim != 2:
        raise ValueError(f"Expected t to have shape (B, T), got {tuple(t.shape)}")
    if t.shape[1] != TIME_FEATURE_DIM:
        raise ValueError(f"Expected t to have {TIME_FEATURE_DIM} features, got {t.shape[1]}")
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError(f"Expected pos to have shape (B, N, 3), got {tuple(pos.shape)}")
    if velocity_in.ndim != 4 or velocity_in.shape[1:] != (5, pos.shape[1], 3):
        raise ValueError(f"Expected velocity_in to have shape (B, 5, N, 3), got {tuple(velocity_in.shape)}")

    domain_min = pos.amin(dim=1)
    domain_max = pos.amax(dim=1)
    return ModelInputs(
        t=t,
        pos=pos,
        idcs_airfoil=idcs_list,
        velocity_in=velocity_in.to(dtype=pos.dtype),
        domain_min=domain_min,
        domain_max=domain_max,
    )


def flatten_velocity_history(velocity_in: torch.Tensor) -> torch.Tensor:
    """Flatten (B, 5, N, 3) to (B, N, 15)."""

    batch_size, time_steps, num_points, num_channels = velocity_in.shape
    return velocity_in.permute(0, 2, 1, 3).reshape(batch_size, num_points, time_steps * num_channels)


def make_coordinate_grid(batch_size: int, grid_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a batch of [-1, 1]^3 coordinate grids."""

    axis = torch.linspace(-1.0, 1.0, grid_size, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([xx, yy, zz], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)


def normalize_points_to_unit_cube(
    pos: torch.Tensor,
    domain_min: torch.Tensor,
    domain_max: torch.Tensor,
    padding: float,
) -> torch.Tensor:
    """Map each batch element into a padded unit cube."""

    center = 0.5 * (domain_min + domain_max)
    scale = (domain_max - domain_min).amax(dim=-1, keepdim=True).clamp_min(1e-6)
    coords = (pos - center[:, None, :]) / scale[:, None, :] + 0.5
    if padding > 0.0:
        coords = coords * (1.0 - 2.0 * padding) + padding
    return coords.clamp(0.0, 1.0)


def sample_volume_features(
    volume: torch.Tensor,
    pos: torch.Tensor,
    *,
    domain_min: torch.Tensor,
    domain_max: torch.Tensor,
    padding: float,
) -> torch.Tensor:
    """Sample dense voxel features at point locations."""

    coords = normalize_points_to_unit_cube(pos, domain_min, domain_max, padding=padding)
    grid = coords.mul(2.0).sub(1.0)
    grid = grid[:, :, None, None, [2, 1, 0]]
    sampled = F.grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[:, :, :, 0, 0].transpose(1, 2)


def zero_airfoil_predictions(prediction: torch.Tensor, idcs_airfoil: Iterable[torch.Tensor]) -> torch.Tensor:
    """Enforce zero velocity on airfoil surface points."""

    num_points = prediction.shape[2]
    for batch_index, raw_indices in enumerate(idcs_airfoil):
        valid = raw_indices[(raw_indices >= 0) & (raw_indices < num_points)].unique()
        if valid.numel() > 0:
            prediction[batch_index, :, valid, :] = 0.0
    return prediction
