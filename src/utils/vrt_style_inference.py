"""
Inference-time patterns from the Volumetric Routing / VRT-ensemble line (e.g. PR #19).

These utilities are **model-agnostic**: any module with the competition signature
``(t, pos, idcs_airfoil, velocity_in) -> (B, 5, N, 3)`` can use reflection TTA or
persistence fallback without depending on VRT or ensemble checkpoints.
"""

from __future__ import annotations

from typing import List

import torch


def reflect_y_inputs(
    pos: torch.Tensor, velocity_in: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror about the x–z plane: ``y -> -y`` and ``v_y -> -v_y``.

    Same convention as v3ctr0id VRT ensemble: reflected prediction is mapped back
    with :func:`unreflect_y_prediction`.
    """
    pos_ref = pos.clone()
    pos_ref[:, :, 1] *= -1.0
    vel_ref = velocity_in.clone()
    vel_ref[..., 1] *= -1.0
    return pos_ref, vel_ref


def unreflect_y_prediction(pred: torch.Tensor) -> torch.Tensor:
    """Map a velocity field from reflected coordinates back to the original frame."""
    out = pred.clone()
    out[..., 1] *= -1.0
    return out


@torch.inference_mode()
def forward_with_y_reflection_tta(
    model: torch.nn.Module,
    t: torch.Tensor,
    pos: torch.Tensor,
    idcs_airfoil: List[torch.Tensor],
    velocity_in: torch.Tensor,
) -> torch.Tensor:
    """Run the model in the original and y-reflected frames; average the two outputs.

    This is a **single-model** version of the reflection branch from VRT ensemble
    (one forward + one reflected forward, then mean), keeping cost at 2×.
    """
    p0 = model(t, pos, idcs_airfoil, velocity_in)
    pos_r, vel_r = reflect_y_inputs(pos, velocity_in)
    p1 = unreflect_y_prediction(model(t, pos_r, idcs_airfoil, vel_r))
    return 0.5 * (p0 + p1)


@torch.inference_mode()
def persistence_prediction(
    velocity_in: torch.Tensor,
    idcs_airfoil: List[torch.Tensor],
    *,
    t_out: int = 5,
) -> torch.Tensor:
    """Repeat the last input frame and enforce a zero (no-slip) airfoil mask."""
    last = velocity_in[:, -1:, :, :]
    pred = last.repeat(1, t_out, 1, 1).contiguous()
    for b, idx in enumerate(idcs_airfoil):
        if idx.numel() == 0:
            continue
        pred[b, :, idx.long().to(device=pred.device, dtype=torch.long), :] = 0.0
    return pred


@torch.inference_mode()
def should_use_persistence_fallback(
    velocity_in: torch.Tensor,
    *,
    in_norm_threshold: float = 33_000.0,
    in_step_mean_threshold: float = 1.0,
    batch_index: int = 0,
) -> bool:
    """Heuristic: very large ||v|| but tiny frame-to-frame changes (batch ``batch_index``)."""
    v = velocity_in[batch_index]
    in_norm = float(torch.linalg.norm(v.reshape(-1)).item())
    if v.shape[0] < 2:
        return False
    step_norms = [
        float(torch.linalg.norm((v[k] - v[k - 1]).reshape(-1)).item())
        for k in range(1, v.shape[0])
    ]
    in_step_mean = sum(step_norms) / max(1, len(step_norms))
    return in_norm >= in_norm_threshold and in_step_mean <= in_step_mean_threshold


@torch.inference_mode()
def apply_vrt_style_inference(
    model: torch.nn.Module,
    t: torch.Tensor,
    pos: torch.Tensor,
    idcs_airfoil: List[torch.Tensor],
    velocity_in: torch.Tensor,
    *,
    reflection_tta: bool = False,
    persistence_fallback: bool = False,
    in_norm_threshold: float = 33_000.0,
    in_step_mean_threshold: float = 1.0,
) -> torch.Tensor:
    """Run optional persistence override, else optional reflection TTA, else plain ``model``."""
    if persistence_fallback and should_use_persistence_fallback(
        velocity_in,
        in_norm_threshold=in_norm_threshold,
        in_step_mean_threshold=in_step_mean_threshold,
    ):
        t_out = model.num_t_out if hasattr(model, "num_t_out") else 5
        return persistence_prediction(velocity_in, idcs_airfoil, t_out=t_out)
    if reflection_tta:
        return forward_with_y_reflection_tta(model, t, pos, idcs_airfoil, velocity_in)
    return model(t, pos, idcs_airfoil, velocity_in)
