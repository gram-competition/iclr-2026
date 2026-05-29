from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from kymatio.scattering3d.filter_bank import solid_harmonic_filter_bank
from kymatio.torch import HarmonicScattering3D
from torch import nn

from .ops import normalize_points_to_unit_cube
from .types import ScatteringFeatureConfig, VoxelGridSpec


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _gaussian_kernel1d(sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if sigma <= 0.0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    radius = max(int(round(3.0 * sigma)), 1)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_smooth_3d_channels(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable Gaussian smoothing to each channel independently."""

    if sigma <= 0.0:
        return x

    channels = x.shape[1]
    kernel = _gaussian_kernel1d(sigma, device=x.device, dtype=x.dtype)
    radius = kernel.numel() // 2

    kernels = [
        kernel.view(1, 1, -1, 1, 1).expand(channels, 1, -1, 1, 1),
        kernel.view(1, 1, 1, -1, 1).expand(channels, 1, 1, -1, 1),
        kernel.view(1, 1, 1, 1, -1).expand(channels, 1, 1, 1, -1),
    ]
    paddings = [
        [0, 0, 0, 0, radius, radius],
        [0, 0, radius, radius, 0, 0],
        [radius, radius, 0, 0, 0, 0],
    ]

    for kernel_3d, padding in zip(kernels, paddings):
        x = F.pad(x, padding, mode="replicate")
        x = F.conv3d(x, kernel_3d, groups=channels)
    return x


def anti_aliased_downsample_channels(
    x: torch.Tensor,
    output_size: int,
    sigma_by_channel: Sequence[float],
) -> torch.Tensor:
    """Smooth channels before average pooling to a smaller cubic grid."""

    if x.shape[-1] == output_size:
        return x

    factor = x.shape[-1] // output_size
    out = torch.empty(
        x.shape[0],
        x.shape[1],
        output_size,
        output_size,
        output_size,
        device=x.device,
        dtype=x.dtype,
    )

    channel_groups: dict[float, list[int]] = {}
    for channel_index, sigma in enumerate(float(value) for value in sigma_by_channel):
        channel_groups.setdefault(sigma, []).append(channel_index)

    for sigma, channel_indices in channel_groups.items():
        index_tensor = torch.tensor(channel_indices, device=x.device, dtype=torch.long)
        channel_block = x.index_select(1, index_tensor)
        channel_block = gaussian_smooth_3d_channels(channel_block, sigma)
        channel_block = F.avg_pool3d(channel_block, kernel_size=factor, stride=factor)
        out.index_copy_(1, index_tensor, channel_block)

    return out


def batched_trilinear_splat(
    points: torch.Tensor,
    features: torch.Tensor | None,
    *,
    spec: VoxelGridSpec,
    mins: torch.Tensor,
    maxs: torch.Tensor,
) -> torch.Tensor:
    """Rasterize point features onto a dense voxel grid with trilinear splatting."""

    batch_size, num_points, _ = points.shape
    grid_size = spec.grid_size

    if features is None:
        features = torch.ones(batch_size, num_points, 1, device=points.device, dtype=points.dtype)

    channels = features.shape[-1]
    coords = normalize_points_to_unit_cube(points, domain_min=mins, domain_max=maxs, padding=spec.padding) * float(
        grid_size - 1
    )
    base = torch.floor(coords).long()
    frac = coords - base.float()

    flat_size = grid_size * grid_size * grid_size
    volume = torch.zeros(batch_size * flat_size, channels, device=points.device, dtype=points.dtype)
    mass = torch.zeros(batch_size * flat_size, 1, device=points.device, dtype=points.dtype)
    batch_offset = (torch.arange(batch_size, device=points.device) * flat_size)[:, None]

    for dx in (0, 1):
        wx = (1.0 - frac[..., 0]) if dx == 0 else frac[..., 0]
        ix = (base[..., 0] + dx).clamp(0, grid_size - 1)
        for dy in (0, 1):
            wy = (1.0 - frac[..., 1]) if dy == 0 else frac[..., 1]
            iy = (base[..., 1] + dy).clamp(0, grid_size - 1)
            for dz in (0, 1):
                wz = (1.0 - frac[..., 2]) if dz == 0 else frac[..., 2]
                iz = (base[..., 2] + dz).clamp(0, grid_size - 1)

                weights = (wx * wy * wz).reshape(batch_size, num_points, 1)
                linear_index = ix * (grid_size * grid_size) + iy * grid_size + iz
                linear_index = (linear_index + batch_offset).reshape(-1)

                mass.index_add_(0, linear_index, weights.reshape(-1, 1))
                volume.index_add_(0, linear_index, (features * weights).reshape(-1, channels))

    volume = volume / mass.clamp_min(1e-6)
    volume = volume.view(batch_size, grid_size, grid_size, grid_size, channels).permute(0, 4, 1, 2, 3).contiguous()

    if spec.smooth_sigma > 0.0:
        volume = gaussian_smooth_3d_channels(volume, spec.smooth_sigma)
    if spec.normalize_mass:
        total = volume.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        volume = volume / total
    return volume


def local_scattering_channel_count(j: int, l: int, max_order: int) -> int:
    """Number of local scattering maps emitted for one scalar input channel."""

    first_order = (l + 1) * (j + 1)
    if max_order == 1:
        return first_order
    second_order = (l + 1) * (j * (j + 1) // 2)
    return first_order + second_order


class CustomHarmonicScattering3D(nn.Module):
    """Small Kymatio wrapper for pooled or localized 3D scattering features."""

    def __init__(
        self,
        J: int,
        shape: tuple[int, int, int],
        *,
        L: int = 3,
        sigma_0: float = 1.0,
        max_order: int = 2,
        integral_powers: Sequence[float] = (0.5, 1.0, 2.0),
        method: str = "integral",
        output_size: int | None = None,
    ) -> None:
        super().__init__()
        self.J = J
        self.L = L
        self.sigma_0 = sigma_0
        self.max_order = max_order
        self.integral_powers = tuple(float(power) for power in integral_powers)
        self.method = method
        self.output_size = output_size

        filters_np = solid_harmonic_filter_bank(
            shape[0],
            shape[1],
            shape[2],
            J=J,
            L=L,
            sigma_0=sigma_0,
            fourier=True,
        )
        self.filters: list[list[list[torch.Tensor]]] = []
        for l_index in range(L + 1):
            level_filters: list[list[torch.Tensor]] = []
            for j_index in range(J + 1):
                order_filters = [torch.tensor(filt, dtype=torch.complex64) for filt in filters_np[l_index][j_index]]
                level_filters.append(order_filters)
            self.filters.append(level_filters)

        self.integral_scattering = HarmonicScattering3D(
            J=J,
            shape=shape,
            L=L,
            sigma_0=sigma_0,
            max_order=max_order,
            integral_powers=list(self.integral_powers),
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.integral_scattering = self.integral_scattering.to(*args, **kwargs)
        for l_index in range(len(self.filters)):
            for j_index in range(len(self.filters[l_index])):
                for m_index in range(len(self.filters[l_index][j_index])):
                    self.filters[l_index][j_index][m_index] = self.filters[l_index][j_index][m_index].to(
                        *args, **kwargs
                    )
        return self

    @staticmethod
    def _modulus_rotation(coefficients: list[torch.Tensor]) -> torch.Tensor:
        squared = []
        for coefficient in coefficients:
            if coefficient.is_complex():
                squared.append(coefficient.real.square() + coefficient.imag.square())
            else:
                squared.append(coefficient.square())
        return torch.sqrt(torch.stack(squared, dim=0).sum(dim=0).clamp_min(1e-12))

    def _sigma_for_channel(self, scale_j: int) -> float:
        return max(0.5, 0.5 * float(self.sigma_0) * (2.0**scale_j))

    def _local_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
        x_ft = torch.fft.fftn(x, dim=(-3, -2, -1))
        maps: list[torch.Tensor] = []
        sigmas: list[float] = []

        for l_index in range(self.L + 1):
            for j1 in range(self.J + 1):
                first_order_coeffs = []
                for filt in self.filters[l_index][j1]:
                    conv = torch.fft.ifftn(x_ft * filt[None], dim=(-3, -2, -1))
                    first_order_coeffs.append(conv)
                u1 = self._modulus_rotation(first_order_coeffs)
                maps.append(u1)
                sigmas.append(self._sigma_for_channel(j1))

                if self.max_order > 1:
                    u1_ft = torch.fft.fftn(u1, dim=(-3, -2, -1))
                    for j2 in range(j1 + 1, self.J + 1):
                        second_order_coeffs = []
                        for filt in self.filters[l_index][j2]:
                            conv2 = torch.fft.ifftn(u1_ft * filt[None], dim=(-3, -2, -1))
                            second_order_coeffs.append(conv2)
                        u2 = self._modulus_rotation(second_order_coeffs)
                        maps.append(u2)
                        sigmas.append(self._sigma_for_channel(max(j1, j2)))

        return torch.stack(maps, dim=1), sigmas

    def forward(self, x: torch.Tensor, method: str | None = None) -> torch.Tensor:
        method = self.method if method is None else method
        if method == "integral":
            flat = x.abs().reshape(x.shape[0], -1)
            order0 = torch.stack([flat.pow(power).sum(dim=1) for power in self.integral_powers], dim=1)
            coeffs = self.integral_scattering(x).float().reshape(x.shape[0], -1)
            return torch.cat([order0, coeffs], dim=1)
        if method == "local":
            return self._local_maps(x)[0]
        if method == "local_downsampled":
            maps, sigmas = self._local_maps(x)
            if self.output_size is not None:
                maps = anti_aliased_downsample_channels(maps, self.output_size, sigmas)
            return maps
        raise ValueError(f"Unknown scattering method: {method}")


def create_scattering_modules(
    config: ScatteringFeatureConfig,
    *,
    device: str | torch.device | None = None,
) -> tuple[CustomHarmonicScattering3D, CustomHarmonicScattering3D]:
    """Create pooled and local scattering modules for runtime inference."""

    resolved_device = _resolve_device(device)
    integral = CustomHarmonicScattering3D(
        J=config.j,
        shape=(config.grid_size, config.grid_size, config.grid_size),
        L=config.l,
        sigma_0=config.sigma_0,
        max_order=config.max_order,
        integral_powers=config.integral_powers,
        method="integral",
    ).to(resolved_device)
    local = CustomHarmonicScattering3D(
        J=config.j,
        shape=(config.grid_size, config.grid_size, config.grid_size),
        L=config.l,
        sigma_0=config.sigma_0,
        max_order=config.max_order,
        integral_powers=config.integral_powers,
        method="local_downsampled",
        output_size=config.map_pooled_size,
    ).to(resolved_device)
    return integral, local


def _voxel_spec(config: ScatteringFeatureConfig, *, smooth_sigma: float, normalize_mass: bool) -> VoxelGridSpec:
    return VoxelGridSpec(
        grid_size=config.grid_size,
        padding=config.padding,
        smooth_sigma=smooth_sigma,
        normalize_mass=normalize_mass,
    )


def build_runtime_wavelet_conditioning(
    *,
    pos: torch.Tensor,
    idcs_airfoil: list[torch.Tensor],
    velocity_in: torch.Tensor,
    config: ScatteringFeatureConfig,
    domain_min: torch.Tensor,
    domain_max: torch.Tensor,
    integral_scattering: CustomHarmonicScattering3D | None = None,
    local_scattering: CustomHarmonicScattering3D | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the runtime conditioning tensors used by the submission model.

    Returns:
        history_wavelet_maps: (B, T, C, P, P, P)
        geom_scattering: (B, G)
        geom_volume: (B, grid, grid, grid)
    """

    batch_size, time_steps, num_points, _ = velocity_in.shape
    device = pos.device
    dtype = pos.dtype

    geometry_spec = _voxel_spec(config, smooth_sigma=config.smooth_sigma, normalize_mass=True)
    velocity_spec = _voxel_spec(config, smooth_sigma=config.smooth_sigma, normalize_mass=False)

    if integral_scattering is None or local_scattering is None:
        integral_scattering, local_scattering = create_scattering_modules(config, device=device)
    else:
        integral_scattering = integral_scattering.to(device)
        local_scattering = local_scattering.to(device)

    geometry_volumes: list[torch.Tensor] = []
    geometry_features: list[torch.Tensor] = []
    history_wavelet_maps: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_index in range(batch_size):
            mins = domain_min[batch_index : batch_index + 1].to(device=device, dtype=dtype)
            maxs = domain_max[batch_index : batch_index + 1].to(device=device, dtype=dtype)

            raw_indices = torch.as_tensor(idcs_airfoil[batch_index], device=device, dtype=torch.long)
            valid_indices = raw_indices[(raw_indices >= 0) & (raw_indices < num_points)].unique()
            surface_points = pos[batch_index, valid_indices] if valid_indices.numel() > 0 else pos[batch_index]

            geom_volume = batched_trilinear_splat(
                surface_points.unsqueeze(0),
                None,
                spec=geometry_spec,
                mins=mins,
                maxs=maxs,
            )[0, 0]
            geometry_volumes.append(geom_volume)
            geometry_features.append(integral_scattering(geom_volume.unsqueeze(0), method="integral")[0])

            velocity_history = velocity_in[batch_index]
            mean_field = velocity_history.mean(dim=0, keepdim=True)
            residual_history = velocity_history - mean_field

            repeated_points = pos[batch_index].unsqueeze(0).expand(time_steps, -1, -1)
            repeated_mins = mins.expand(time_steps, -1)
            repeated_maxs = maxs.expand(time_steps, -1)
            speed = residual_history.norm(dim=-1, keepdim=True)
            scalar_channels = {
                "vx": residual_history[..., 0:1],
                "vy": residual_history[..., 1:2],
                "vz": residual_history[..., 2:3],
                "speed": speed,
            }

            per_channel_maps = []
            for channel_name in config.velocity_channel_mixing:
                voxel_history = batched_trilinear_splat(
                    repeated_points,
                    scalar_channels[channel_name],
                    spec=velocity_spec,
                    mins=repeated_mins,
                    maxs=repeated_maxs,
                )[:, 0]
                per_channel_maps.append(local_scattering(voxel_history, method="local_downsampled"))

            history_wavelet_maps.append(torch.cat(per_channel_maps, dim=1))

    return (
        torch.stack(history_wavelet_maps, dim=0),
        torch.stack(geometry_features, dim=0),
        torch.stack(geometry_volumes, dim=0),
    )
