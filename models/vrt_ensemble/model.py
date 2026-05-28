"""
VRT Ensemble Model for GRaM Competition.

This is a 4-member ensemble of Volumetric Routing Transformer (VRT) models,
averaging predictions with optional reflection test-time augmentation (TTA).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .volumetric_routing_transformer import VolumetricRoutingTransformer

# HuggingFace model repo
HF_REPO_ID = "vinayedula/vrt-ensemble"


class VRTEnsemble(nn.Module):
    """4-member VRT ensemble with reflection TTA and persistence fallback."""

    def __init__(
        self,
        enable_reflection_tta: bool = True,
        enable_hard_fallback: bool = True,
        in_norm_threshold: float = 33000.0,
        in_step_mean_threshold: float = 1.0,
    ):
        """Initialize ensemble by loading 4 pre-trained VRT models.
        
        Args:
            enable_reflection_tta: Use y-reflection test-time augmentation (8-way mean)
            enable_hard_fallback: Use persistence fallback for high-magnitude low-dynamics inputs
            in_norm_threshold: Input L2 norm threshold for fallback activation
            in_step_mean_threshold: Mean temporal step norm threshold for fallback
        """
        super().__init__()
        self.enable_reflection_tta = enable_reflection_tta
        self.enable_hard_fallback = enable_hard_fallback
        self.in_norm_threshold = in_norm_threshold
        self.in_step_mean_threshold = in_step_mean_threshold

        # Get the checkpoint directory relative to this file
        this_dir = Path(__file__).parent
        ckpt_base = this_dir / "checkpoints"

        # Load 4 ensemble members with shared hyperparameters
        self.models = nn.ModuleList()
        for member_idx, member_dir in enumerate(
            sorted([ckpt_base / f"member{i+1}" for i in range(4)])
        ):
            model = self._load_member(member_dir, member_idx)
            self.models.append(model)

        # Keep constructor device-agnostic. The evaluator decides device via model.to(...).
        self.device = torch.device("cpu")

    def _load_member(self, member_dir: Path, member_idx: int) -> VolumetricRoutingTransformer:
        """Load a single ensemble member model from local or HuggingFace."""
        member_name = member_dir.name
        
        # Cache directory in the submission folder
        cache_dir = str(member_dir.parent / "checkpoints")
        
        # Try to load from HuggingFace if available, fallback to local
        try:
            ckpt_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"checkpoints/{member_name}/best.pt",
                cache_dir=cache_dir,
            )
            stats_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"checkpoints/{member_name}/vrt_flow_stats.pt",
                cache_dir=cache_dir,
            )
        except Exception:
            # Fallback to local checkpoints if HF download fails
            ckpt_path = member_dir / "best.pt"
            stats_path = member_dir / "vrt_flow_stats.pt"
            
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Member {member_idx} missing checkpoint: {ckpt_path}")
            if not stats_path.is_file():
                raise FileNotFoundError(f"Member {member_idx} missing stats: {stats_path}")

        # Load checkpoint to extract hyperparameters
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        args = ckpt.get("args", {})

        # Reconstruct model with same hyperparameters
        hidden_dimension = int(args.get("hidden_dim", 256))
        base_channels = int(args.get("vrt_base_channels", 64))
        grid_str = str(args.get("vrt_grid", "64,32,32"))
        grid_vals = [int(x.strip()) for x in grid_str.split(",")]
        if len(grid_vals) != 3:
            raise ValueError(f"Invalid vrt_grid in checkpoint: {grid_str}")
        grid_resolution = (grid_vals[0], grid_vals[1], grid_vals[2])
        use_fourier = bool(args.get("use_fourier", True))

        model = VolumetricRoutingTransformer(
            hidden_dimension=hidden_dimension,
            num_pre_blocks=3,
            num_post_blocks=6,
            grid_resolution=grid_resolution,
            base_channels=base_channels,
            temporal_dim=max(32, hidden_dimension // 4),
            num_pos_freqs=10 if use_fourier else 0,
            num_vel_freqs=4,
            num_dist_freqs=8,
            t_out=5,
        )

        # Load state dict and flow statistics
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.load_flow_stats(str(stats_path))
        model.eval()

        return model

    @torch.no_grad()
    def _should_use_persistence_fallback(self, velocity_in: torch.Tensor) -> bool:
        """Detect high-magnitude, low-dynamics inputs from velocity_in only."""
        # velocity_in shape: (batch_size, 5, N, 3)
        v = velocity_in[0]
        in_norm = float(torch.linalg.norm(v.reshape(-1)).item())
        step_norms = [
            torch.linalg.norm((v[k] - v[k - 1]).reshape(-1)).item() for k in range(1, v.shape[0])
        ]
        in_step_mean = float(sum(step_norms) / len(step_norms))
        return in_norm >= self.in_norm_threshold and in_step_mean <= self.in_step_mean_threshold

    @torch.no_grad()
    def _persistence_prediction(
        self,
        velocity_in: torch.Tensor,
        idcs_airfoil: List[torch.Tensor],
        t_out: int = 5,
    ) -> torch.Tensor:
        """Repeat last input frame and enforce no-slip boundary."""
        last = velocity_in[:, -1:, :, :]  # (B,1,N,3)
        pred = last.repeat(1, t_out, 1, 1).contiguous()
        for b, idx in enumerate(idcs_airfoil):
            if idx.numel() > 0:
                pred[b, :, idx.long().to(pred.device), :] = 0.0
        return pred

    @torch.no_grad()
    def _reflect_y_inputs(
        self,
        pos: torch.Tensor,
        velocity_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mirror about x-z plane: y -> -y and v_y -> -v_y."""
        pos_ref = pos.clone()
        pos_ref[:, :, 1] *= -1.0
        vel_in_ref = velocity_in.clone()
        vel_in_ref[..., 1] *= -1.0
        return pos_ref, vel_in_ref

    @torch.no_grad()
    def _unreflect_y_prediction(self, pred_reflected: torch.Tensor) -> torch.Tensor:
        """Map prediction from reflected coordinates back to original."""
        pred = pred_reflected.clone()
        pred[..., 1] *= -1.0
        return pred

    def _runtime_device(self, fallback: torch.device) -> torch.device:
        """Resolve the actual device where model parameters/buffers currently live."""
        for p in self.models.parameters():
            return p.device
        for b in self.models.buffers():
            return b.device
        return fallback

    def __call__(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: List[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        """Ensemble forward pass with optional reflection TTA and fallback.
        
        Args:
            t: Time parameters (batch_size, 10)
            pos: Point positions (batch_size, 100k, 3)
            idcs_airfoil: List of airfoil point indices (variable-length per sample)
            velocity_in: Input velocity history (batch_size, 5, 100k, 3)
            
        Returns:
            Predicted velocity field (batch_size, 5, 100k, 3)
        """
        # Auto-detect input device from input tensors
        input_device = pos.device
        runtime_device = self._runtime_device(input_device)
        self.device = runtime_device

        # Move inputs to the actual runtime device.
        t = t.to(runtime_device)
        pos = pos.to(runtime_device)
        velocity_in = velocity_in.to(runtime_device)
        idcs_airfoil = [idx.to(runtime_device) for idx in idcs_airfoil]

        with torch.inference_mode():
            if self.enable_reflection_tta:
                # Apply reflection TTA: original + reflected for each of 4 models
                pos_ref, vel_in_ref = self._reflect_y_inputs(pos, velocity_in)
                preds = []
                for model in self.models:
                    # Original orientation
                    pred_orig = model(t=t, pos=pos, idcs_airfoil=idcs_airfoil, velocity_in=velocity_in).float()
                    preds.append(pred_orig)
                    
                    # Reflected orientation
                    pred_ref = model(t=t, pos=pos_ref, idcs_airfoil=idcs_airfoil, velocity_in=vel_in_ref).float()
                    preds.append(self._unreflect_y_prediction(pred_ref))
                
                # Average all 8 predictions (4 models × 2 orientations)
                pred = torch.stack(preds, dim=0).mean(dim=0)
            else:
                # Standard ensemble: average predictions from 4 models
                preds = [
                    model(t=t, pos=pos, idcs_airfoil=idcs_airfoil, velocity_in=velocity_in).float()
                    for model in self.models
                ]
                pred = torch.stack(preds, dim=0).mean(dim=0)

            # Apply hard fallback for high-magnitude low-dynamics cases
            if self.enable_hard_fallback and self._should_use_persistence_fallback(velocity_in):
                pred = self._persistence_prediction(velocity_in, idcs_airfoil, t_out=pred.shape[1])

        # Move output back to input device
        return pred.to(input_device)


# Export the model class
__all__ = ["VRTEnsemble"]
