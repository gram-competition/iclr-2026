"""
Volumetric Routing Transformer (VRT) for GRaM.

"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sanitize_boundary_idx(idx: torch.Tensor, num_points: int, device: torch.device) -> torch.Tensor:
    """Clamp boundary indices into valid point range and ensure long dtype."""
    if idx.numel() == 0:
        return idx.to(device=device, dtype=torch.long)
    return idx.to(device=device, dtype=torch.long).clamp_(0, num_points - 1)


def generate_fourier_features(x: torch.Tensor, num_frequencies: int) -> torch.Tensor:
    """Expand coordinates with sin/cos Fourier bands."""
    if num_frequencies <= 0:
        return x
    frequency_bands = (
        2.0 ** torch.arange(num_frequencies, device=x.device, dtype=x.dtype) * math.pi
    )
    scaled_x = x.unsqueeze(-1) * frequency_bands
    spectral_encoding = torch.cat([scaled_x.sin(), scaled_x.cos()], dim=-1).reshape(
        *x.shape[:-1], -1
    )
    return torch.cat([x, spectral_encoding], dim=-1)


@torch.no_grad()
def calculate_distance_to_boundary(
    positions: torch.Tensor, boundary_indices: list[torch.Tensor], chunk_size: int = 8192
) -> torch.Tensor:
    """Shortest Euclidean distance from each point to nearest boundary point."""
    batch_size, num_points, _ = positions.shape
    distances = torch.empty(
        batch_size, num_points, device=positions.device, dtype=positions.dtype
    )
    for b in range(batch_size):
        idx = _sanitize_boundary_idx(boundary_indices[b], num_points, positions.device)
        if idx.numel() == 0:
            distances[b].zero_()
            continue
        surface_points = positions[b].index_select(0, idx)
        parts = []
        for chunk in positions[b].split(chunk_size):
            parts.append(torch.cdist(chunk, surface_points).min(dim=1).values)
        distances[b] = torch.cat(parts, dim=0)
    return distances


@torch.no_grad()
def nearest_boundary_geometry(
    positions: torch.Tensor,
    boundary_indices: list[torch.Tensor],
    chunk_size: int = 8192,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:

    bsz, npts, _ = positions.shape
    distances = torch.empty(bsz, npts, device=positions.device, dtype=positions.dtype)
    directions = torch.zeros(bsz, npts, 3, device=positions.device, dtype=positions.dtype)
    for b in range(bsz):
        idx = _sanitize_boundary_idx(boundary_indices[b], npts, positions.device)
        if idx.numel() == 0:
            distances[b].zero_()
            continue
        surface_points = positions[b].index_select(0, idx)
        d_parts: list[torch.Tensor] = []
        j_parts: list[torch.Tensor] = []
        for chunk in positions[b].split(chunk_size):
            d = torch.cdist(chunk, surface_points)
            mn, j = d.min(dim=1)
            d_parts.append(mn)
            j_parts.append(j)
        dist_b = torch.cat(d_parts, dim=0)
        nearest_idx = torch.cat(j_parts, dim=0)
        nearest = surface_points[nearest_idx]
        vec = nearest - positions[b]
        unit = vec / dist_b.unsqueeze(-1).clamp(min=eps)
        unit = torch.where(dist_b.unsqueeze(-1) < eps, torch.zeros_like(unit), unit)
        distances[b] = dist_b
        directions[b] = unit
    return distances, directions


class PointwiseSwiGLUBlock(nn.Module):
    def __init__(self, dimension: int, expansion_multiplier: float = 8 / 3):
        super().__init__()
        hidden_dim = int(dimension * expansion_multiplier)
        hidden_dim = (hidden_dim + 1) // 2 * 2
        self.layer_norm = nn.LayerNorm(dimension)
        self.gate_proj = nn.Linear(dimension, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dimension, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dimension, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.layer_norm(x)
        g = F.silu(self.gate_proj(n)) * self.up_proj(n)
        return x + self.down_proj(g)


class TemporalVelocityEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_dimension: int = 64):
        super().__init__()
        self.out_dimension = out_dimension
        self.temporal_convs = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, out_dimension, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(out_dimension, out_dimension, kernel_size=1),
        )
        self.feature_fusion = nn.Linear(out_dimension * 2, out_dimension, bias=False)

    def forward(self, normalized_velocity: torch.Tensor) -> torch.Tensor:
        bsz, time_steps, npts, channels = normalized_velocity.shape
        sequence = normalized_velocity.permute(0, 2, 3, 1).reshape(
            bsz * npts, channels, time_steps
        )
        feat = self.temporal_convs(sequence)
        avg = feat.mean(dim=-1)
        latest = feat[:, :, -1]
        fused = self.feature_fusion(torch.cat([avg, latest], dim=-1))
        return fused.reshape(bsz, npts, self.out_dimension)


# =========================================================================
# 3D ConvNeXt V2 
# =========================================================================

class GRN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, D, H, W, C)
        gx = x.norm(p=2, dim=(1, 2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_res = x
        x = self.dwconv(x)
        # Permute to (B, D, H, W, C) for LayerNorm and Linear layers
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # Permute back to (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        return input_res + x


class SOTAConvNeXtV2Lattice(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        grid_resolution: tuple[int, int, int] = (64, 32, 32),
        base_channels: int = 64,
    ):
        super().__init__()
        self.grid_resolution = tuple(grid_resolution)
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.proj_in = nn.Conv3d(hidden_dim, c1, kernel_size=3, padding=1)
        
        # Stacking multiple blocks safely increases capacity without breaking FLOP budget
        self.encoder_level1 = nn.Sequential(ConvNeXtV2Block3D(c1), ConvNeXtV2Block3D(c1))
        self.downsample1 = nn.Conv3d(c1, c2, kernel_size=2, stride=2)

        self.encoder_level2 = nn.Sequential(ConvNeXtV2Block3D(c2), ConvNeXtV2Block3D(c2))
        self.downsample2 = nn.Conv3d(c2, c3, kernel_size=2, stride=2)

        self.encoder_level3 = nn.Sequential(ConvNeXtV2Block3D(c3), ConvNeXtV2Block3D(c3))
        self.downsample3 = nn.Conv3d(c3, c4, kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            ConvNeXtV2Block3D(c4),
            ConvNeXtV2Block3D(c4),
            ConvNeXtV2Block3D(c4)
        )

        # Replaced torch.cat with Additive skips for memory/FLOP optimization
        self.upsample3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.decoder_level3 = nn.Sequential(ConvNeXtV2Block3D(c3), ConvNeXtV2Block3D(c3))

        self.upsample2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.decoder_level2 = nn.Sequential(ConvNeXtV2Block3D(c2), ConvNeXtV2Block3D(c2))

        self.upsample1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.decoder_level1 = nn.Sequential(ConvNeXtV2Block3D(c1), ConvNeXtV2Block3D(c1))

        self.final_projection = nn.Conv3d(c1, hidden_dim, kernel_size=1)

    def scatter_points_to_lattice(
        self, features: torch.Tensor, normalized_coords: torch.Tensor
    ) -> torch.Tensor:
        bsz, npts, channels = features.shape
        gx, gy, gz = self.grid_resolution

        coords = normalized_coords.clamp(0, 1 - 1e-6)
        idx_x = (coords[..., 0] * gx).floor().long()
        idx_y = (coords[..., 1] * gy).floor().long()
        idx_z = (coords[..., 2] * gz).floor().long()
        flat = idx_x * (gy * gz) + idx_y * gz + idx_z

        lattice_state = torch.zeros(
            bsz, channels, gx * gy * gz, device=features.device, dtype=features.dtype
        )
        lattice_count = torch.zeros(
            bsz, 1, gx * gy * gz, device=features.device, dtype=features.dtype
        )

        lattice_state.scatter_add_(2, flat.unsqueeze(1).expand(-1, channels, -1), features.transpose(1, 2))
        lattice_count.scatter_add_(2, flat.unsqueeze(1), torch.ones(bsz, 1, npts, device=features.device, dtype=features.dtype))
        avg = lattice_state / lattice_count.clamp(min=1.0)
        return avg.view(bsz, channels, gx, gy, gz)

    def sample_lattice_to_points(
        self, lattice_features: torch.Tensor, normalized_coords: torch.Tensor
    ) -> torch.Tensor:
        bsz, npts = lattice_features.shape[0], normalized_coords.shape[1]
        grid = (normalized_coords * 2.0 - 1.0)[..., [2, 1, 0]].view(bsz, 1, 1, npts, 3)
        sampled = F.grid_sample(
            lattice_features, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        return sampled.squeeze(2).squeeze(2).transpose(1, 2)

    def forward(self, features: torch.Tensor, normalized_coords: torch.Tensor) -> torch.Tensor:
        lattice = self.scatter_points_to_lattice(features, normalized_coords)
        
        enc1 = self.encoder_level1(self.proj_in(lattice))
        enc2 = self.encoder_level2(self.downsample1(enc1))
        enc3 = self.encoder_level3(self.downsample2(enc2))
        
        bottleneck = self.bottleneck(self.downsample3(enc3))
        
        dec3 = self.upsample3(bottleneck) + enc3
        dec3 = self.decoder_level3(dec3)
        
        dec2 = self.upsample2(dec3) + enc2
        dec2 = self.decoder_level2(dec2)
        
        dec1 = self.upsample1(dec2) + enc1
        dec1 = self.decoder_level1(dec1)
        
        return self.sample_lattice_to_points(self.final_projection(dec1), normalized_coords)

# =========================================================================

class VolumetricRoutingTransformer(nn.Module):
    def __init__(
        self,
        hidden_dimension: int = 256,
        num_pre_blocks: int = 3,
        num_post_blocks: int = 6,
        grid_resolution: tuple[int, int, int] = (64, 32, 32),
        base_channels: int = 64,
        temporal_dim: int = 64,
        num_pos_freqs: int = 10,
        num_vel_freqs: int = 4,
        num_dist_freqs: int = 8,
        t_out: int = 5,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.t_out = t_out
        self.num_pos_freqs = num_pos_freqs
        self.num_vel_freqs = num_vel_freqs
        self.num_dist_freqs = num_dist_freqs
        self.enable_timing = False
        self._last_timers: dict[str, float] = {}

        pos_dim = 3 * (1 + 2 * self.num_pos_freqs)
        vin_flat_dim = 15
        vel_fourier_dim = 3 * 2 * self.num_vel_freqs
        dist_dim = 4 + 2 * self.num_dist_freqs
        geom_dir_dim = 3
        flow_stat_dim = 2
        total_input_dim = (
            pos_dim
            + vin_flat_dim
            + vel_fourier_dim
            + dist_dim
            + geom_dir_dim
            + flow_stat_dim
            + 1
            + temporal_dim
        )

        self.temporal_encoder = TemporalVelocityEncoder(in_channels=3, out_dimension=temporal_dim)
        self.initial_feature_projection = nn.Linear(total_input_dim, hidden_dimension)
        self.pre_routing_blocks = nn.ModuleList(
            [PointwiseSwiGLUBlock(hidden_dimension) for _ in range(num_pre_blocks)]
        )
        
        self.spatial_lattice_solver = SOTAConvNeXtV2Lattice(
            hidden_dimension, grid_resolution=grid_resolution, base_channels=base_channels
        )
        
        self.post_routing_blocks = nn.ModuleList(
            [PointwiseSwiGLUBlock(hidden_dimension) for _ in range(num_post_blocks)]
        )
        self.final_normalization = nn.LayerNorm(hidden_dimension)
        self.velocity_delta_projection = nn.Linear(hidden_dimension, 3 * t_out)
        nn.init.zeros_(self.velocity_delta_projection.weight)
        nn.init.zeros_(self.velocity_delta_projection.bias)
        self.register_buffer("flow_channel_mean", torch.zeros(1, 1, 1, 3), persistent=False)
        self.register_buffer("flow_channel_scale", torch.ones(1, 1, 1, 3), persistent=False)
        self.register_buffer("spatial_bounds_lo", torch.zeros(1, 1, 3), persistent=False)
        self.register_buffer("spatial_bounds_hi", torch.ones(1, 1, 3), persistent=False)
        if pretrained_path is not None and len(pretrained_path) > 0 and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state_dict, strict=False)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for key in (
            prefix + "flow_channel_mean",
            prefix + "flow_channel_scale",
            prefix + "spatial_bounds_lo",
            prefix + "spatial_bounds_hi",
        ):
            state_dict.pop(key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.no_grad()
    def set_flow_stats(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        self.flow_channel_mean.copy_(mean.reshape(1, 1, 1, 3).to(self.flow_channel_mean))
        self.flow_channel_scale.copy_(scale.reshape(1, 1, 1, 3).to(self.flow_channel_scale))

    @torch.no_grad()
    def set_spatial_bounds(self, lo: torch.Tensor, hi: torch.Tensor) -> None:
        self.spatial_bounds_lo.copy_(lo.reshape(1, 1, 3).to(self.spatial_bounds_lo))
        self.spatial_bounds_hi.copy_(hi.reshape(1, 1, 3).to(self.spatial_bounds_hi))

    @torch.no_grad()
    def load_flow_stats(self, stats_path: str) -> None:
        payload = torch.load(stats_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid flow-stats file (expected dict): {stats_path}")
        if "flow_channel_mean" in payload and "flow_channel_scale" in payload:
            mean = torch.as_tensor(payload["flow_channel_mean"], dtype=torch.float32)
            scale = torch.as_tensor(payload["flow_channel_scale"], dtype=torch.float32)
        else:
            raise KeyError(
                f"Flow-stats file missing required keys in {stats_path}. "
                "Need flow_channel_mean/flow_channel_scale."
            )
        self.set_flow_stats(mean, scale)

        if "spatial_bounds_lo" in payload and "spatial_bounds_hi" in payload:
            lo = torch.as_tensor(payload["spatial_bounds_lo"], dtype=torch.float32)
            hi = torch.as_tensor(payload["spatial_bounds_hi"], dtype=torch.float32)
            self.set_spatial_bounds(lo, hi)
        else:
            raise KeyError(
                f"Flow-stats file missing required keys in {stats_path}. "
                "Need spatial_bounds_lo/spatial_bounds_hi."
            )

    @torch.no_grad()
    def save_flow_stats(self, stats_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)
        payload = {
            "flow_channel_mean": self.flow_channel_mean.detach().cpu().view(3),
            "flow_channel_scale": self.flow_channel_scale.detach().cpu().view(3),
            "spatial_bounds_lo": self.spatial_bounds_lo.detach().cpu().view(3),
            "spatial_bounds_hi": self.spatial_bounds_hi.detach().cpu().view(3),
        }
        torch.save(payload, stats_path)

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        del t
        timings: dict[str, float] = {}
        if self.enable_timing and pos.device.type == "cuda":
            torch.cuda.synchronize(pos.device)
        t0 = torch.cuda.Event(enable_timing=True) if (self.enable_timing and pos.device.type == "cuda") else None
        t1 = torch.cuda.Event(enable_timing=True) if (self.enable_timing and pos.device.type == "cuda") else None

        if t0 is not None:
            t0.record()
        surface_distances, boundary_dirs = nearest_boundary_geometry(pos, idcs_airfoil)
        if t1 is not None:
            t1.record()
            torch.cuda.synchronize(pos.device)
            timings["distance"] = t0.elapsed_time(t1) / 1000.0

        bsz, _, npts, _ = velocity_in.shape
        last_velocity_frame = velocity_in[:, -1]
        normalized_velocity = (velocity_in - self.flow_channel_mean) / self.flow_channel_scale

        position_features = generate_fourier_features(pos, self.num_pos_freqs)
        flattened_velocity_history = normalized_velocity.permute(0, 2, 1, 3).reshape(bsz, npts, 15)

        vel_frequencies = (
            2.0 ** torch.arange(self.num_vel_freqs, device=pos.device, dtype=pos.dtype) * math.pi
        )
        scaled_last_vel = normalized_velocity[:, -1].unsqueeze(-1) * vel_frequencies
        last_velocity_spectral = torch.cat([scaled_last_vel.sin(), scaled_last_vel.cos()], dim=-1).reshape(bsz, npts, -1)

        temporal_features = self.temporal_encoder(normalized_velocity)
        speed = torch.linalg.norm(normalized_velocity, dim=-1)  # (B, T, N)
        flow_mean_speed = speed.mean(dim=1, keepdim=False).unsqueeze(-1)  # (B, N, 1)
        flow_std_speed = speed.std(dim=1, unbiased=False, keepdim=False).unsqueeze(-1)  # (B, N, 1)

        boundary_mask = torch.zeros(bsz, npts, 1, device=pos.device, dtype=pos.dtype)
        for b, idx in enumerate(idcs_airfoil):
            idx = _sanitize_boundary_idx(idx, npts, pos.device)
            if idx.numel() > 0:
                boundary_mask[b, idx, 0] = 1.0

        dist_expanded = surface_distances.unsqueeze(-1)
        dist_log = torch.log1p(dist_expanded)
        # Per-sample normalization over points (B, 1, 1) keeps rank aligned with (B, N, 1).
        dist_norm = dist_expanded / dist_expanded.amax(dim=1, keepdim=True).clamp_min(1e-6)
        dist_inv = 1.0 / (1.0 + dist_expanded)
        dist_frequencies = (
            2.0 ** torch.arange(self.num_dist_freqs, device=pos.device, dtype=pos.dtype) * math.pi
        )
        scaled_dist = dist_log * dist_frequencies
        distance_features = torch.cat(
            [dist_expanded, dist_log, dist_norm, dist_inv, scaled_dist.sin(), scaled_dist.cos()],
            dim=-1,
        )

        features = torch.cat(
            [
                position_features,
                flattened_velocity_history,
                last_velocity_spectral,
                temporal_features,
                flow_mean_speed,
                flow_std_speed,
                distance_features,
                boundary_dirs,
                boundary_mask,
            ],
            dim=-1,
        )
        hidden = self.initial_feature_projection(features)

        for block in self.pre_routing_blocks:
            hidden = block(hidden)

        # Guard against degenerate/invalid bounds that can produce NaNs/Infs and
        # later invalid lattice indices on CUDA.
        bounds_span = (self.spatial_bounds_hi - self.spatial_bounds_lo).clamp_min(1e-6)
        normalized_coords = (pos - self.spatial_bounds_lo) / bounds_span
        normalized_coords = torch.nan_to_num(normalized_coords, nan=0.5, posinf=1.0, neginf=0.0)
        normalized_coords = normalized_coords.clamp(0.0, 1.0 - 1e-6)
        hidden = hidden + self.spatial_lattice_solver(hidden, normalized_coords)

        for block in self.post_routing_blocks:
            hidden = block(hidden)

        delta_norm = self.velocity_delta_projection(self.final_normalization(hidden))
        delta_norm = delta_norm.reshape(bsz, npts, self.t_out, 3).permute(0, 2, 1, 3)
        pred = last_velocity_frame.unsqueeze(1) + (delta_norm * self.flow_channel_scale)

        for b, idx in enumerate(idcs_airfoil):
            idx = _sanitize_boundary_idx(idx, npts, pred.device)
            if idx.numel() > 0:
                pred[b, :, idx, :] = 0.0

        self._last_timers = timings if self.enable_timing else {}
        return pred
