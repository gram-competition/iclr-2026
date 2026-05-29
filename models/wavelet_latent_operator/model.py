from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .features import build_runtime_wavelet_conditioning, create_scattering_modules, local_scattering_channel_count
from .layers import WaveletResidualBlock
from .ops import (
    flatten_velocity_history,
    make_coordinate_grid,
    normalize_submission_inputs,
    sample_volume_features,
    zero_airfoil_predictions,
)
from .types import ScatteringFeatureConfig


class WaveletLatentOperator(nn.Module):
    """Inference-only GRaM submission model."""

    def __init__(self) -> None:
        super().__init__()
        config = self._load_config()
        scattering_config = ScatteringFeatureConfig(**config["scattering_config"])

        self.scattering_config = scattering_config
        self.output_steps = 5
        self.local_basis_dim = 7
        self.map_size = scattering_config.map_pooled_size
        self.geom_scattering_dim = int(config["geom_scattering_dim"])
        self.use_repeat_last_baseline = bool(config.get("use_repeat_last_baseline", True))
        self.zero_airfoil = bool(config.get("zero_airfoil", True))
        latent_dim = int(config.get("latent_dim", 64))
        depth = int(config.get("depth", 4))
        point_hidden_dim = int(config.get("point_hidden_dim", 160))
        decoder_hidden_dim = int(config.get("decoder_hidden_dim", 160))

        self.integral_scattering, self.local_scattering = create_scattering_modules(scattering_config, device="cpu")
        self.maps_per_channel = local_scattering_channel_count(
            scattering_config.j,
            scattering_config.l,
            scattering_config.max_order,
        )
        self.map_channels = len(scattering_config.velocity_channel_mixing) * self.maps_per_channel

        self.geometry_context = self._build_geometry_context(latent_dim)
        self.geometry_gate = self._build_geometry_gate(latent_dim)
        self.lift = self._build_latent_lift(latent_dim)
        self.blocks = nn.ModuleList([WaveletResidualBlock(latent_dim, latent_dim) for _ in range(depth)])
        self.future_head = nn.Sequential(
            nn.Conv3d(latent_dim, latent_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(latent_dim, self.output_steps * latent_dim, kernel_size=1),
        )
        self.coefficient_decoder = self._build_coefficient_decoder(point_hidden_dim, decoder_hidden_dim, latent_dim)
        self.coefficient_scale = 0.35

        self._load_weights(config)

    @staticmethod
    def _module_dir() -> Path:
        return Path(__file__).resolve().parent

    @classmethod
    def _load_config(cls) -> dict[str, Any]:
        with (cls._module_dir() / "config.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _build_geometry_context(self, latent_dim: int) -> nn.Sequential:
        hidden_dim = max(64, self.geom_scattering_dim // 2)
        return nn.Sequential(
            nn.LayerNorm(self.geom_scattering_dim),
            nn.Linear(self.geom_scattering_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _build_geometry_gate(self, latent_dim: int) -> nn.Sequential:
        hidden_dim = max(64, self.geom_scattering_dim // 2)
        return nn.Sequential(
            nn.LayerNorm(self.geom_scattering_dim),
            nn.Linear(self.geom_scattering_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.map_channels),
        )

    def _build_latent_lift(self, latent_dim: int) -> nn.Sequential:
        input_channels = self.output_steps * self.map_channels + 1 + 3
        return nn.Sequential(
            nn.Conv3d(input_channels, latent_dim, kernel_size=1),
            nn.GroupNorm(1, latent_dim),
            nn.GELU(),
        )

    def _build_coefficient_decoder(
        self, point_hidden_dim: int, decoder_hidden_dim: int, latent_dim: int
    ) -> nn.Sequential:
        decoder_input_dim = 3 + 15 + 5 + 1 + latent_dim
        return nn.Sequential(
            nn.Linear(decoder_input_dim, point_hidden_dim),
            nn.LayerNorm(point_hidden_dim),
            nn.GELU(),
            nn.Linear(point_hidden_dim, decoder_hidden_dim),
            nn.LayerNorm(decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, self.local_basis_dim),
        )

    def _load_weights(self, config: dict[str, Any]) -> None:
        weights_path = self._module_dir() / config.get("weights_path", "state_dict.pt")
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing submission weights: {weights_path}")

        state = torch.load(weights_path, map_location="cpu")

        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]

        missing, unexpected = self.load_state_dict(state, strict=False)

        allowed_missing_prefixes = (
            "integral_scattering.integral_scattering.tensor",
            "integral_scattering.integral_scattering.tensor_gaussian_filter",
            "local_scattering.integral_scattering.tensor",
            "local_scattering.integral_scattering.tensor_gaussian_filter",
        )

        bad_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]

        if bad_missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch. Missing keys: {bad_missing}. Unexpected keys: {unexpected}")

    def _downsample_geometry_volume(self, geom_volume: torch.Tensor) -> torch.Tensor:
        geom_volume = geom_volume.unsqueeze(1)
        if geom_volume.shape[-1] == self.map_size:
            return geom_volume
        factor = geom_volume.shape[-1] // self.map_size
        return F.avg_pool3d(geom_volume, kernel_size=factor, stride=factor)

    def _build_latent_input(
        self,
        history_wavelet_maps: torch.Tensor,
        geom_scattering: torch.Tensor,
        geom_volume: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.geometry_context(geom_scattering)
        gates = torch.sigmoid(self.geometry_gate(geom_scattering))[:, None, :, None, None, None]
        modulated_maps = history_wavelet_maps * gates
        latent_input = modulated_maps.reshape(
            history_wavelet_maps.shape[0],
            self.output_steps * self.map_channels,
            self.map_size,
            self.map_size,
            self.map_size,
        )
        geom_volume = self._downsample_geometry_volume(geom_volume)
        return latent_input, geom_volume, context

    def _decode_future_steps(
        self,
        future_latent: torch.Tensor,
        geom_volume: torch.Tensor,
        pos: torch.Tensor,
        domain_min: torch.Tensor,
        domain_max: torch.Tensor,
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, num_points, _ = velocity_in.shape
        velocity_basis = velocity_in.permute(0, 2, 1, 3)
        mean_basis = velocity_basis.mean(dim=2, keepdim=True)
        delta_basis = (velocity_basis[:, :, -1] - velocity_basis[:, :, -2]).unsqueeze(2)
        local_basis = torch.cat([velocity_basis, mean_basis, delta_basis], dim=2)

        history_flat = flatten_velocity_history(velocity_in)
        speed_history = velocity_in.norm(dim=-1).permute(0, 2, 1)
        local_geometry = sample_volume_features(
            geom_volume,
            pos,
            domain_min=domain_min,
            domain_max=domain_max,
            padding=self.scattering_config.padding,
        )

        predictions = []
        for step_index in range(self.output_steps):
            sampled_latent = sample_volume_features(
                future_latent[:, step_index],
                pos,
                domain_min=domain_min,
                domain_max=domain_max,
                padding=self.scattering_config.padding,
            )
            decoder_inputs = torch.cat([pos, history_flat, speed_history, local_geometry, sampled_latent], dim=-1)
            coefficients = torch.tanh(self.coefficient_decoder(decoder_inputs)) * self.coefficient_scale
            predictions.append(torch.einsum("bnk,bnkc->bnc", coefficients, local_basis))

        prediction = torch.stack(predictions, dim=1)
        if self.use_repeat_last_baseline:
            baseline = velocity_in[:, -1:, :, :].expand(batch_size, self.output_steps, num_points, 3)
            prediction = baseline + prediction
        return prediction

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        inputs = normalize_submission_inputs(t, pos, idcs_airfoil, velocity_in)

        history_wavelet_maps, geom_scattering, geom_volume = build_runtime_wavelet_conditioning(
            pos=inputs.pos,
            idcs_airfoil=inputs.idcs_airfoil,
            velocity_in=inputs.velocity_in,
            config=self.scattering_config,
            domain_min=inputs.domain_min,
            domain_max=inputs.domain_max,
            integral_scattering=self.integral_scattering,
            local_scattering=self.local_scattering,
        )
        history_wavelet_maps = history_wavelet_maps.to(device=inputs.pos.device, dtype=inputs.pos.dtype)
        geom_scattering = geom_scattering.to(device=inputs.pos.device, dtype=inputs.pos.dtype)
        geom_volume = geom_volume.to(device=inputs.pos.device, dtype=inputs.pos.dtype)

        latent_input, geom_volume, context = self._build_latent_input(
            history_wavelet_maps, geom_scattering, geom_volume
        )
        coords = make_coordinate_grid(latent_input.shape[0], self.map_size, latent_input.device, latent_input.dtype)
        hidden = self.lift(torch.cat([latent_input, geom_volume, coords], dim=1))
        for block in self.blocks:
            hidden = block(hidden, context)

        future_latent = self.future_head(hidden).view(
            latent_input.shape[0],
            self.output_steps,
            -1,
            self.map_size,
            self.map_size,
            self.map_size,
        )
        prediction = self._decode_future_steps(
            future_latent,
            geom_volume,
            inputs.pos,
            inputs.domain_min,
            inputs.domain_max,
            inputs.velocity_in,
        )
        return zero_airfoil_predictions(prediction, inputs.idcs_airfoil) if self.zero_airfoil else prediction
