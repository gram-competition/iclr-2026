"""Mean ensemble of two SpatioTemporalGNN checkpoints (competition entry)."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .gnn_base import SpatioTemporalGNN
from .gnn_physfeat import SpatioTemporalGNNPhysFeat

# Weights on Hugging Face — override with env GRAM_ENSEMBLE_HF_REPO if you use a different space
HF_REPO_ID = os.environ.get("GRAM_ENSEMBLE_HF_REPO", "vinayedula/spatiotemporal_models")
_HF_PATH_PREFIX = "ensemble_spatiotemporal_models"
HF_FILE_BASE = f"{_HF_PATH_PREFIX}/checkpoints_SpatioTemporalGNN_300epochs/best.pt"
HF_FILE_PHYSFEAT = f"{_HF_PATH_PREFIX}/checkpoints_SpatioTemporalGNNPhysFeat_100epochs/best.pt"


def _sanitize_airfoil_idx(idx: torch.Tensor, num_points: int, device: torch.device) -> torch.Tensor:
    if idx.numel() == 0:
        return idx.to(device=device, dtype=torch.long)
    return idx.to(device=device, dtype=torch.long).clamp_(0, num_points - 1)


class MeanOutputEnsemble(nn.Module):
    """Average outputs from two competition-compatible forecasters."""

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        if not models:
            raise ValueError("MeanOutputEnsemble requires at least one model.")
        self.models = nn.ModuleList(models)
        for m in self.models:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        preds = [m(t=t, pos=pos, idcs_airfoil=idcs_airfoil, velocity_in=velocity_in) for m in self.models]
        return torch.stack(preds, dim=0).mean(dim=0)


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_spatiotemporal_gnn_from_checkpoint(checkpoint_path: Path) -> nn.Module:
    ckpt = _load_checkpoint(checkpoint_path)
    args = ckpt["args"]
    model = SpatioTemporalGNN(
        backbone=args["backbone"],
        hidden_dim=args["hidden_dim"],
        num_layers=args["num_layers"],
        heads=args["heads"],
        k=args["k"],
        num_sub=args["num_sub"],
        use_fourier=args.get("use_fourier", True),
        use_hierarchical=args.get("use_hierarchical", False),
        dropout=0.0,
        interp_k=args.get("interp_k", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _build_spatiotemporal_gnn_physfeat_from_checkpoint(checkpoint_path: Path) -> nn.Module:
    ckpt = _load_checkpoint(checkpoint_path)
    args = ckpt["args"]
    model = SpatioTemporalGNNPhysFeat(
        backbone=args["backbone"],
        hidden_dim=args["hidden_dim"],
        num_layers=args["num_layers"],
        heads=args["heads"],
        k=args["k"],
        num_sub=args["num_sub"],
        use_fourier=args.get("use_fourier", True),
        use_hierarchical=args.get("use_hierarchical", False),
        dropout=0.0,
        interp_k=args.get("interp_k", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _hf_download_checkpoints(cache_dir: Path) -> tuple[Path, Path]:
    """Download both checkpoints; cache under package dir (same idea as vrt-ensemble)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    p_base = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILE_BASE,
        cache_dir=str(cache_dir),
    )
    p_phys = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILE_PHYSFEAT,
        cache_dir=str(cache_dir),
    )
    return Path(p_base), Path(p_phys)


def _load_spatiotemporal_pair_mean_ensemble(package_dir: Path, device: torch.device) -> nn.Module:
    """Load base + physfeat; inference is unweighted mean of the two forwards."""
    cache_dir = package_dir / ".hf_cache"
    ckpt_base, ckpt_physfeat = _hf_download_checkpoints(cache_dir)
    base = _build_spatiotemporal_gnn_from_checkpoint(ckpt_base)
    physfeat = _build_spatiotemporal_gnn_physfeat_from_checkpoint(ckpt_physfeat)
    return MeanOutputEnsemble([base, physfeat]).to(device).eval()


class EnsembleSpatioTemporalModels(nn.Module):
    """Mean of two GNNs; optional VRT-style persistence fallback on pathological inputs."""

    def __init__(
        self,
        enable_hard_fallback: bool = True,
        in_norm_threshold: float = 33000.0,
        in_step_mean_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.enable_hard_fallback = enable_hard_fallback
        self.in_norm_threshold = in_norm_threshold
        self.in_step_mean_threshold = in_step_mean_threshold

        package_dir = Path(__file__).parent.resolve()
        # Keep construction device-agnostic. The evaluator may later call
        # model.to(cuda:N), and forward should follow that runtime device.
        self.device = torch.device("cpu")
        self.ensemble = _load_spatiotemporal_pair_mean_ensemble(package_dir, self.device)

    def _runtime_device(self, fallback: torch.device) -> torch.device:
        """Resolve the actual device where ensemble params/buffers currently live."""
        for p in self.ensemble.parameters():
            return p.device
        for b in self.ensemble.buffers():
            return b.device
        return fallback

    @torch.no_grad()
    def _should_use_persistence_fallback(self, velocity_in: torch.Tensor) -> bool:
        """High input L2 norm but small frame-to-frame change → persistence (uses batch item 0)."""
        v = velocity_in[0]
        in_norm = float(torch.linalg.norm(v.reshape(-1)).item())
        step_norms = [
            float(torch.linalg.norm((v[k] - v[k - 1]).reshape(-1)).item()) for k in range(1, v.shape[0])
        ]
        in_step_mean = float(sum(step_norms) / len(step_norms))
        return in_norm >= self.in_norm_threshold and in_step_mean <= self.in_step_mean_threshold

    @torch.no_grad()
    def _persistence_prediction(
        self,
        velocity_in: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        t_out: int,
    ) -> torch.Tensor:
        last = velocity_in[:, -1:, :, :]
        pred = last.repeat(1, t_out, 1, 1).contiguous()
        for b, idx in enumerate(idcs_airfoil):
            idx = _sanitize_airfoil_idx(idx, pred.shape[2], pred.device)
            if idx.numel() > 0:
                pred[b, :, idx, :] = 0.0
        return pred

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        input_device = pos.device
        runtime_device = self._runtime_device(input_device)
        self.device = runtime_device
        t = t.to(runtime_device)
        pos = pos.to(runtime_device)
        velocity_in = velocity_in.to(runtime_device)
        n_pts = pos.shape[1]
        idcs_airfoil = [_sanitize_airfoil_idx(idx, n_pts, runtime_device) for idx in idcs_airfoil]

        with torch.inference_mode():
            if self.enable_hard_fallback and self._should_use_persistence_fallback(velocity_in):
                t_out = int(t.shape[1] - velocity_in.shape[1])
                out = self._persistence_prediction(
                    velocity_in, idcs_airfoil, t_out=max(t_out, 1)
                )
            else:
                out = self.ensemble(
                    t=t, pos=pos, idcs_airfoil=idcs_airfoil, velocity_in=velocity_in
                )
        return out.to(input_device)
