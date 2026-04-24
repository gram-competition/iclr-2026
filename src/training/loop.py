from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from src.data import WarpedIFWDataset, build_loader, unscale_velocity_batch
from src.training.pointcloud_losses import compute_wall_distance_weights


def max_abs_boundary_error(
    velocity: torch.Tensor,
    idcs_airfoil: list[torch.Tensor],
    *,
    velocity_mean: torch.Tensor | None = None,
    velocity_std: torch.Tensor | None = None,
) -> float:
    max_abs = 0.0
    for batch_idx, boundary_indices in enumerate(idcs_airfoil):
        if boundary_indices.numel() == 0:
            continue
        boundary_values = velocity[batch_idx, :, boundary_indices.long(), :]
        if boundary_values.numel() == 0:
            continue

        if velocity_mean is not None and velocity_std is not None:
            scaled_zero = ((0.0 - velocity_mean[batch_idx]) / velocity_std[batch_idx]).view(
                1,
                1,
                -1,
            )
            boundary_error = boundary_values - scaled_zero.to(
                device=boundary_values.device,
                dtype=boundary_values.dtype,
            )
        else:
            boundary_error = boundary_values

        batch_max = float(boundary_error.abs().max().item())
        if batch_max > max_abs:
            max_abs = batch_max
    return max_abs


def assert_no_slip_boundary(
    pred_scaled: torch.Tensor,
    idcs_airfoil: list[torch.Tensor],
    *,
    atol: float,
    context: str,
    velocity_mean: torch.Tensor | None = None,
    velocity_std: torch.Tensor | None = None,
) -> None:
    with torch.no_grad():
        max_abs = max_abs_boundary_error(
            pred_scaled,
            idcs_airfoil,
            velocity_mean=velocity_mean,
            velocity_std=velocity_std,
        )
    if max_abs > atol:
        if velocity_mean is not None and velocity_std is not None:
            raise RuntimeError(
                f"{context}: no-slip boundary check failed. "
                f"max(|pred_scaled - scaled_zero| on airfoil)={max_abs:.3e} > atol={atol:.3e}."
            )
        raise RuntimeError(
            f"{context}: no-slip boundary check failed. "
            f"max(|pred_scaled| on airfoil)={max_abs:.3e} > atol={atol:.3e}."
        )


def hint_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).norm(dim=3).mean(dim=(1, 2)).mean()


def move_batch_to_device(batch, device: torch.device):
    (
        t,
        pos,
        idcs_airfoil,
        velocity_in,
        velocity_out,
        pressure_in,
        velocity_mean,
        velocity_std,
        wall_distance,
        surface_frame,
        knn_indices,
    ) = batch
    return (
        t.to(device, non_blocking=True),
        pos.to(device, non_blocking=True),
        [indices.to(device, non_blocking=True) for indices in idcs_airfoil],
        velocity_in.to(device, non_blocking=True),
        velocity_out.to(device, non_blocking=True),
        pressure_in.to(device, non_blocking=True),
        velocity_mean.to(device, non_blocking=True),
        velocity_std.to(device, non_blocking=True),
        wall_distance.to(device, non_blocking=True),
        surface_frame.to(device, non_blocking=True),
        knn_indices.to(device, non_blocking=True) if knn_indices is not None else None,
    )


def autocast_context(device: torch.device, *, enabled: bool):
    if enabled and device.type == "cuda":
        # bfloat16: native on Ampere+, wider dynamic range than fp16, no GradScaler needed.
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def forward_model(
    model: torch.nn.Module,
    t: torch.Tensor,
    pos: torch.Tensor,
    idcs_airfoil: list[torch.Tensor],
    velocity_in: torch.Tensor,
    velocity_mean: torch.Tensor,
    velocity_std: torch.Tensor,
    *,
    return_knn_indices: bool,
    pressure_in: torch.Tensor | None = None,
    wall_distance: torch.Tensor | None = None,
    surface_frame: torch.Tensor | None = None,
    knn_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if return_knn_indices:
        try:
            pred_scaled, knn_indices = model(
                t,
                pos,
                idcs_airfoil,
                velocity_in,
                velocity_mean,
                velocity_std,
                return_knn_indices=True,
                pressure_in=pressure_in,
                wall_distance=wall_distance,
                surface_frame=surface_frame,
                knn_indices=knn_indices,
            )
        except TypeError as exc:
            raise TypeError(
                "Model.forward must support return_knn_indices=True when Sobolev loss is enabled."
            ) from exc
        return pred_scaled, knn_indices

    pred_scaled = model(
        t,
        pos,
        idcs_airfoil,
        velocity_in,
        velocity_mean,
        velocity_std,
        pressure_in=pressure_in,
        wall_distance=wall_distance,
        surface_frame=surface_frame,
        knn_indices=knn_indices,
    )
    return pred_scaled, None


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    no_slip_atol: float = 0.0,
    use_amp: bool = False,
    assert_no_slip: bool = False,
    use_sobolev: bool = False,
    wall_distance_loss_alpha: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """Returns (mean RL2 scaled, mean HINT unscaled, 0.0 placeholder, persistence RL2, persistence HINT).

    Persistence = repeat last input velocity frame for all output timesteps (same spaces as model).
    """
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    total_persist_loss = 0.0
    total_persist_metric = 0.0
    with torch.no_grad():
        for batch in loader:
            (
                t,
                pos,
                idcs_airfoil,
                velocity_in,
                velocity_out,
                pressure_in,
                velocity_mean,
                velocity_std,
                wall_distance,
                surface_frame,
                batch_knn_indices,
            ) = move_batch_to_device(batch, device)
            with autocast_context(device, enabled=use_amp):
                pred_scaled, knn_indices = forward_model(
                    model,
                    t,
                    pos,
                    idcs_airfoil,
                    velocity_in,
                    velocity_mean,
                    velocity_std,
                    return_knn_indices=use_sobolev,
                    pressure_in=pressure_in,
                    wall_distance=wall_distance,
                    surface_frame=surface_frame,
                    knn_indices=batch_knn_indices,
                )
                wall_weights = None
                if wall_distance_loss_alpha > 0.0:
                    wall_weights = compute_wall_distance_weights(
                        pos, idcs_airfoil, alpha=wall_distance_loss_alpha,
                        wall_distance=wall_distance,
                    )
                if use_sobolev:
                    loss = loss_fn(pred_scaled, velocity_out, knn_indices=knn_indices, pos=pos, wall_weights=wall_weights)
                else:
                    loss = loss_fn(pred_scaled, velocity_out, wall_weights=wall_weights)

            persist_pred = velocity_in[:, -1:, :, :].expand_as(velocity_out)
            if use_sobolev:
                p_loss = loss_fn(
                    persist_pred,
                    velocity_out,
                    knn_indices=knn_indices,
                    pos=pos,
                    wall_weights=wall_weights,
                )
            else:
                p_loss = loss_fn(
                    persist_pred, velocity_out, wall_weights=wall_weights
                )
            total_persist_loss += p_loss.item()

            if assert_no_slip:
                assert_no_slip_boundary(
                    pred_scaled,
                    idcs_airfoil,
                    atol=no_slip_atol,
                    context="evaluate",
                    velocity_mean=velocity_mean,
                    velocity_std=velocity_std,
                )

            total_loss += loss.item()

            pred_unscaled = unscale_velocity_batch(
                pred_scaled.float(),
                velocity_mean,
                velocity_std,
            )
            target_unscaled = unscale_velocity_batch(
                velocity_out.float(),
                velocity_mean,
                velocity_std,
            )
            total_metric += hint_metric(pred_unscaled, target_unscaled).item()

            persist_unscaled = unscale_velocity_batch(
                persist_pred.float(),
                velocity_mean,
                velocity_std,
            )
            total_persist_metric += hint_metric(
                persist_unscaled, target_unscaled
            ).item()
    denom = max(1, len(loader))
    return (
        total_loss / denom,
        total_metric / denom,
        0.0,
        total_persist_loss / denom,
        total_persist_metric / denom,
    )


def run_full_test_inference(
    model: torch.nn.Module,
    test_files: Sequence[Path],
    loss_fn: torch.nn.Module,
    device: torch.device,
    *,
    num_points: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    scaler_eps: float,
    no_slip_atol: float,
    prefetch_factor: int,
    persistent_workers: bool,
    use_amp: bool,
    assert_no_slip: bool,
    use_sobolev: bool = False,
    wall_distance_loss_alpha: float = 0.0,
    precompute_knn: bool = False,
    knn_cache_dir: str | Path | None = None,
    knn_k: int = 16,
) -> tuple[float, float, float, float, float]:
    test_dataset = WarpedIFWDataset(
        test_files,
        num_points=num_points,
        random_crop=False,
        seed=seed,
        scaler_eps=scaler_eps,
        precompute_knn=precompute_knn,
        knn_cache_dir=knn_cache_dir,
        knn_k=knn_k,
    )
    test_loader = build_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    print(
        f"Running full test inference on {len(test_dataset)} samples with "
        f"{num_points} points/sample."
    )
    return evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        no_slip_atol=no_slip_atol,
        use_amp=use_amp,
        assert_no_slip=assert_no_slip,
        use_sobolev=use_sobolev,
        wall_distance_loss_alpha=wall_distance_loss_alpha,
    )
