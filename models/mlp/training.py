from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR


def compute_wall_distance_weights(
    pos: torch.Tensor,
    idcs_airfoil: list[torch.Tensor],
    *,
    alpha: float = 1.0,
    eps: float = 1e-4,
    wall_distance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-point loss weights that up-weight near-wall errors.

    Returns a ``(B, 1, N, 1)`` weight tensor broadcastable to ``(B, T, N, C)``
    prediction shapes.  Weight for point *i*:

        ``w_i = 1 / (d_i + eps)^alpha``

    The weights are normalised per sample so that their mean equals 1 –
    this keeps the effective loss scale the same regardless of ``alpha``.

    Parameters
    ----------
    pos : (B, N, 3)
    idcs_airfoil : list of per-sample airfoil index tensors
    alpha : exponent controlling how sharply near-wall points dominate
    eps : prevents division by zero for on-wall points
    wall_distance : optional precomputed ``(B, N)`` raw wall distances
    """
    batch_size, num_pos, _ = pos.shape
    weights = torch.ones(
        (batch_size, 1, num_pos, 1), device=pos.device, dtype=pos.dtype
    )
    for b, indices in enumerate(idcs_airfoil):
        if indices.numel() == 0:
            continue
        if wall_distance is not None:
            min_dist = wall_distance[b]  # (N,)
        else:
            airfoil_pos = pos[b, indices.to(device=pos.device, dtype=torch.long), :]
            dists = torch.cdist(
                pos[b].unsqueeze(0), airfoil_pos.unsqueeze(0)
            ).squeeze(0)  # (N, A)
            min_dist = dists.min(dim=-1).values  # (N,)
        w = 1.0 / (min_dist + eps).pow(alpha)
        w = w / w.mean().clamp_min(1e-8)  # normalise so mean == 1
        weights[b, 0, :, 0] = w
    return weights


class RelativeL2Loss(nn.Module):
    """Relative L2 error: ||pred - target||_2 / (||target||_2 + eps)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        wall_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have identical shapes, got {pred.shape} and {target.shape}."
            )

        if wall_weights is not None:
            # Weighted RL2: weight each point's squared error then sum.
            # pred/target: (B, T, N, C);  wall_weights: (B, 1, N, 1)
            sq_error = (pred - target).pow(2) * wall_weights  # broadcast
            batch_size = pred.shape[0]
            numerator = sq_error.reshape(batch_size, -1).sum(dim=1).sqrt()
            denominator = target.reshape(batch_size, -1).norm(dim=1).clamp_min(self.eps)
            return (numerator / denominator).mean()

        batch_size = pred.shape[0]
        pred_flat = pred.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)

        numerator = torch.linalg.norm(pred_flat - target_flat, dim=1)
        denominator = torch.linalg.norm(target_flat, dim=1).clamp_min(self.eps)
        return (numerator / denominator).mean()


class SobolevLoss(nn.Module):
    """Relative L2 + relative spatial gradient penalty (Sobolev-like norm).

    The spatial gradient term penalises smoothed predictions by comparing
    kNN-estimated spatial gradients of *pred* and *target*. The gradient
    discrepancy is normalised by the target gradient statistic so its scale
    stays comparable to ``RelativeL2Loss``. When ``grad_weight`` is 0 this
    reduces to plain ``RelativeL2Loss``.

    Parameters
    ----------
    grad_weight : float
        Multiplier for the spatial-gradient penalty relative to RL2.
    eps : float
        Denominator clamp for relative norms.
    """

    def __init__(self, grad_weight: float = 0.1, eps: float = 1e-8):
        super().__init__()
        if grad_weight < 0.0:
            raise ValueError("grad_weight must be >= 0.")
        self.grad_weight = grad_weight
        self.eps = eps
        self._rl2 = RelativeL2Loss(eps=eps)

    @staticmethod
    def _knn_gradient_norm(
        field: torch.Tensor,
        pos: torch.Tensor,
        knn_indices: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Mean L2 norm of kNN-estimated spatial gradients per sample.

        Parameters
        ----------
        field : (B, T, N, C)
        pos : (B, N, 3)
        knn_indices : (B, N, K)  – long indices into the N dimension.

        Returns
        -------
        (B,) – mean gradient norm per sample.
        """
        batch_size, num_times, num_points, num_channels = field.shape
        k = knn_indices.shape[2]

        # Gather neighbour positions: (B, N, K, 3)
        offsets = (
            torch.arange(batch_size, device=pos.device, dtype=torch.long)
            .view(batch_size, 1, 1)
            .mul(num_points)
        )
        flat_idx = (knn_indices + offsets).reshape(-1)
        neigh_pos = pos.reshape(-1, 3).index_select(0, flat_idx).view(
            batch_size, num_points, k, 3
        )
        # Keep spatial differences in float32 so AMP does not reduce the
        # neighbour-distance denominator precision.
        rel_pos = (neigh_pos - pos.unsqueeze(2)).to(dtype=torch.float32)  # (B, N, K, 3)
        dist_sq = rel_pos.pow(2).sum(dim=-1).clamp_min(eps)  # (B, N, K)

        # Gather neighbour field values: (B, T, N, K, C)
        flat_idx_t = flat_idx.unsqueeze(0).expand(num_times, -1)
        field_flat = field.reshape(num_times, batch_size * num_points, num_channels)
        neigh_field = (
            field_flat.gather(1, flat_idx_t.unsqueeze(-1).expand(-1, -1, num_channels))
            .view(num_times, batch_size, num_points, k, num_channels)
            .permute(1, 0, 2, 3, 4)
        )  # (B, T, N, K, C)
        rel_field = neigh_field.to(dtype=torch.float32) - field.unsqueeze(3).to(
            dtype=torch.float32
        )  # (B, T, N, K, C)

        # Approximate gradient magnitude per channel:
        #   |df/dx| ≈ |f_j - f_i| / |x_j - x_i|  averaged over neighbours.
        grad_mag = rel_field.abs() / dist_sq.sqrt().unsqueeze(1).unsqueeze(-1).clamp_min(eps)
        return grad_mag.mean(dim=(1, 2, 3, 4))  # (B,)

    def decompose(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        knn_indices: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        wall_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the RL2 term and the normalised gradient term separately."""
        rl2 = self._rl2(pred, target, wall_weights=wall_weights)
        if knn_indices is None or pos is None:
            return rl2, rl2.new_zeros(())

        pred_grad = self._knn_gradient_norm(pred, pos, knn_indices, self.eps)
        target_grad = self._knn_gradient_norm(target, pos, knn_indices, self.eps)
        grad_error = (
            (pred_grad - target_grad).abs()
            / target_grad.clamp_min(self.eps)
        ).mean()
        return rl2, grad_error

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        knn_indices: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        wall_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rl2, grad_error = self.decompose(
            pred,
            target,
            knn_indices=knn_indices,
            pos=pos,
            wall_weights=wall_weights,
        )
        if self.grad_weight == 0.0 or knn_indices is None or pos is None:
            return rl2
        return rl2 + self.grad_weight * grad_error


LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "rl2": RelativeL2Loss,
    "sobolev": SobolevLoss,
}


@dataclass
class TrainingComponents:
    """Container for the standard optimizer/scheduler/loss setup."""

    loss_fn: nn.Module
    optimizer: AdamW
    scheduler: OneCycleLR | LambdaLR
    epochs: int


def build_training_components(
    model: nn.Module,
    *,
    steps_per_epoch: int,
    max_lr: float = 1e-3,
    epochs: int = 500,
    weight_decay: float = 1e-4,
    lr_scheduler_name: str = "one-cycle",
    loss_fn_name: str = "rl2",
    sobolev_grad_weight: float = 0.1,
) -> TrainingComponents:
    """
    Build the requested training setup.

    Parameters
    ----------
    lr_scheduler_name : str
        Learning-rate schedule to use. Supported values are ``"one-cycle"``
        and ``"fixed"``.
    loss_fn_name : str
        Key in ``LOSS_REGISTRY``.  ``"rl2"`` (default) or ``"sobolev"``.
    sobolev_grad_weight : float
        Spatial-gradient penalty weight when ``loss_fn_name="sobolev"``.
    """
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive.")

    loss_cls = LOSS_REGISTRY.get(loss_fn_name)
    if loss_cls is None:
        raise ValueError(
            f"Unknown loss function '{loss_fn_name}'. "
            f"Available: {sorted(LOSS_REGISTRY.keys())}."
        )
    if loss_fn_name == "sobolev":
        loss_fn: nn.Module = SobolevLoss(grad_weight=sobolev_grad_weight)
    else:
        loss_fn = loss_cls()

    normalized_scheduler_name = lr_scheduler_name.strip().lower()
    if normalized_scheduler_name not in {"one-cycle", "fixed"}:
        raise ValueError(
            f"Unknown lr_scheduler_name '{lr_scheduler_name}'. "
            "Available: ['fixed', 'one-cycle']."
        )

    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    if normalized_scheduler_name == "one-cycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            div_factor=10.0,
            final_div_factor=100.0,
        )
    else:
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda _: 1.0,
        )

    return TrainingComponents(
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
    )
