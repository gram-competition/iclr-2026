"""GRaM competition entry point for the FiniteGraphV4 model.

Usage (as required by the competition):

    from models import FiniteGraphV4

    model = FiniteGraphV4()    # loads weights during construction
    velocity_out = model(t, pos, idcs_airfoil, velocity_in)

Signature:
    t:            (B, 10)
    pos:          (B, 100_000, 3)
    idcs_airfoil: list[Tensor], variable-length indices into pos
    velocity_in:  (B, 5, 100_000, 3)
    return:       (B, 5, 100_000, 3)

The model predicts residuals r_k relative to v_t4 (the last input
timestep) for k = 0..4, adds them back to v_t4 and zeros the velocity on
the airfoil to enforce the no-slip boundary condition exactly.

The trained checkpoint is expected alongside this file as ``weights.pt``.
It is a standard PyTorch save dict with keys
``{state_dict, model_config, stats, ...}``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .features import build_features
from .net import FiniteGraphModelV4, FiniteGraphInferenceWrapperV4


WEIGHTS_FILENAME = "weights.pt"

MODEL_CONFIG = {
    "in_ch": 26,
    "hidden": 128,
    "latent": 128,
    "k1": 16,
    "k2": 4,
    "n_attn_heads": 4,
    "out_heads": 5,
    "out_ch_per_head": 3,
    "shared_weights": False,
    "dropout": 0.05,
    "temporal_hidden": 48,
}


class FiniteGraphV4(nn.Module):
    """Two-hop directional finite-graph forecaster for GRaM ICLR 2026 (v4)."""

    def __init__(self):
        super().__init__()

        ckpt_path = Path(__file__).resolve().parent / WEIGHTS_FILENAME
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"FiniteGraphV4: expected weights file at {ckpt_path}. "
                f"Drop the trained checkpoint into "
                f"'{ckpt_path.parent}' as '{WEIGHTS_FILENAME}' "
                f"before instantiating the model."
            )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.inner = FiniteGraphModelV4(**MODEL_CONFIG)
        self.inner.load_state_dict(ckpt["state_dict"])
        self.inner.eval()

        s = ckpt["stats"]
        self.register_buffer("feat_mean", s["feat_mean"])
        self.register_buffer("feat_std", s["feat_std"])
        self.register_buffer("resid_mean", s["resid_mean"])
        self.register_buffer("resid_std", s["resid_std"])

        self._wrapper = FiniteGraphInferenceWrapperV4(
            self.inner, k_pool=128, inference_batch=768,
        )
        self._wrapper.set_stats(self.feat_mean, self.feat_std)

    @torch.no_grad()
    def forward(
        self,
        t: torch.Tensor,                      # (B, 10)  — unused
        pos: torch.Tensor,                    # (B, N, 3)
        idcs_airfoil: list[torch.Tensor],     # len B, each (M_b,)
        velocity_in: torch.Tensor,            # (B, 5, N, 3)
    ) -> torch.Tensor:
        del t  # timestamps are not needed — deltas use the fixed dt = 1 ms
        B = pos.shape[0]
        outs = [
            self._predict_single(pos[b], idcs_airfoil[b], velocity_in[b])
            for b in range(B)
        ]
        return torch.stack(outs, dim=0)

    def _predict_single(
        self,
        pos: torch.Tensor,                # (N, 3)
        idcs_airfoil: torch.Tensor,       # (M,)
        velocity_in: torch.Tensor,        # (5, N, 3)
    ) -> torch.Tensor:
        device = pos.device
        N = pos.shape[0]

        pos_np = pos.detach().cpu().numpy().astype(np.float32)
        vin_np = velocity_in.detach().cpu().numpy().astype(np.float32)
        idcs_np = idcs_airfoil.detach().cpu().numpy()

        features_np = build_features(pos_np, vin_np, idcs_np)
        features = torch.from_numpy(features_np).to(device)

        fm = self.feat_mean.to(device)
        fs = self.feat_std.to(device)
        rm = self.resid_mean.to(device)
        rs = self.resid_std.to(device)

        self._wrapper.set_stats(fm, fs)
        self._wrapper.to(device)
        self.inner.to(device)

        features_norm = (features - fm) / fs
        out_norm = self._wrapper(features_norm, pos.to(device))  # (5, N, 3)

        residuals = out_norm * rs + rm
        v_t4 = velocity_in[4].to(device)
        v_out = residuals + v_t4.unsqueeze(0)

        airfoil_mask = torch.zeros(N, dtype=torch.bool, device=device)
        airfoil_mask[idcs_airfoil.to(device)] = True
        v_out[:, airfoil_mask, :] = 0.0

        return v_out
