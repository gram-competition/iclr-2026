"""Standalone SmoothSplatNet backbone derived from the V13 single model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


T_IN = 5
T_OUT = 5

DOMAIN_MIN = torch.tensor([0.0, -0.41, 0.0])
DOMAIN_MAX = torch.tensor([2.10, 0.41, 1.22])


def _fourier(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype) * math.pi
    xf = x.unsqueeze(-1) * freqs
    enc = torch.cat([xf.sin(), xf.cos()], dim=-1).reshape(*x.shape[:-1], -1)
    return torch.cat([x, enc], dim=-1)


def _compute_dist(pos: torch.Tensor, idcs_airfoil: list[torch.Tensor]) -> torch.Tensor:
    batch_size, num_pos, _ = pos.shape
    out = torch.empty(batch_size, num_pos, device=pos.device, dtype=pos.dtype)
    for batch_idx in range(batch_size):
        idx = idcs_airfoil[batch_idx].to(pos.device).long()
        surf = pos[batch_idx].index_select(0, idx)
        chunks = [
            torch.cdist(chunk, surf).min(dim=1).values
            for chunk in pos[batch_idx].split(8192)
        ]
        out[batch_idx] = torch.cat(chunks, dim=0)
    return out


class AdaptiveCoordinateWarp(nn.Module):
    """Learned per-axis monotonic coordinate transform."""

    def __init__(self, n_knots: int = 16):
        super().__init__()
        assert n_knots % 2 == 0
        self.x_logits = nn.Parameter(torch.zeros(n_knots))
        self.y_logits = nn.Parameter(torch.zeros(n_knots // 2))
        self.z_logits = nn.Parameter(torch.zeros(n_knots))

    @staticmethod
    def _warp_axis(x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(logits, dim=0)
        num_bins = weights.shape[0]
        cdf = torch.zeros(num_bins + 1, device=x.device, dtype=x.dtype)
        cdf[1:] = torch.cumsum(weights, dim=0)
        scaled = (x * num_bins).clamp(0, num_bins - 1e-6)
        bin_idx = scaled.long()
        frac = scaled - bin_idx.float()
        return cdf[bin_idx] + frac * weights[bin_idx]

    def forward(self, pos01: torch.Tensor) -> torch.Tensor:
        y_logits_full = torch.cat([self.y_logits, self.y_logits.flip(0)])
        return torch.stack(
            [
                self._warp_axis(pos01[..., 0], self.x_logits),
                self._warp_axis(pos01[..., 1], y_logits_full),
                self._warp_axis(pos01[..., 2], self.z_logits),
            ],
            dim=-1,
        )


class SE3D(nn.Module):
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels = x.shape[:2]
        weights = self.pool(x).view(batch_size, channels)
        return x * self.fc(weights).view(batch_size, channels, 1, 1, 1)


class _DoubleConvSE(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )
        self.se = SE3D(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv(x))


class _VoxelUNetSmooth(nn.Module):
    def __init__(self, hidden: int, grid: tuple[int, int, int] = (64, 32, 32), ch_base: int = 64):
        super().__init__()
        self.grid = tuple(grid)
        ch = ch_base
        self.enc1 = _DoubleConvSE(hidden, ch)
        self.enc2 = _DoubleConvSE(ch, ch * 2)
        self.enc3 = _DoubleConvSE(ch * 2, ch * 4)
        self.dec2 = _DoubleConvSE(ch * 4 + ch * 2, ch * 2)
        self.dec1 = _DoubleConvSE(ch * 2 + ch, ch)
        self.out = nn.Conv3d(ch, hidden, 1)
        self.pool = nn.MaxPool3d(2)

    def _trilinear_splat(self, x: torch.Tensor, pos_w: torch.Tensor) -> torch.Tensor:
        batch_size, num_pos, channels = x.shape
        grid_x, grid_y, grid_z = self.grid

        dtype = x.dtype
        cx = (pos_w[..., 0].float() * grid_x).clamp(0, grid_x - 1e-6)
        cy = (pos_w[..., 1].float() * grid_y).clamp(0, grid_y - 1e-6)
        cz = (pos_w[..., 2].float() * grid_z).clamp(0, grid_z - 1e-6)

        x0 = cx.floor().long().clamp(0, grid_x - 1)
        y0 = cy.floor().long().clamp(0, grid_y - 1)
        z0 = cz.floor().long().clamp(0, grid_z - 1)

        xd = (cx - x0.float()).to(dtype).unsqueeze(1)
        yd = (cy - y0.float()).to(dtype).unsqueeze(1)
        zd = (cz - z0.float()).to(dtype).unsqueeze(1)

        x1 = (x0 + 1).clamp(max=grid_x - 1)
        y1 = (y0 + 1).clamp(max=grid_y - 1)
        z1 = (z0 + 1).clamp(max=grid_z - 1)

        voxel = torch.zeros(batch_size, channels, grid_x * grid_y * grid_z, device=x.device, dtype=dtype)
        count = torch.zeros(batch_size, 1, grid_x * grid_y * grid_z, device=x.device, dtype=dtype)
        feat_t = x.transpose(1, 2)

        for xi, xw in ((x0, 1 - xd), (x1, xd)):
            for yi, yw in ((y0, 1 - yd), (y1, yd)):
                for zi, zw in ((z0, 1 - zd), (z1, zd)):
                    weights = xw * yw * zw
                    flat = (xi * (grid_y * grid_z) + yi * grid_z + zi).unsqueeze(1)
                    voxel.scatter_add_(2, flat.expand(-1, channels, -1), feat_t * weights)
                    count.scatter_add_(2, flat, weights.to(dtype))

        voxel = voxel / count.clamp(min=1.0)
        return voxel.view(batch_size, channels, grid_x, grid_y, grid_z)

    @staticmethod
    def _devoxelize(voxel: torch.Tensor, pos_w: torch.Tensor) -> torch.Tensor:
        num_pos = pos_w.shape[1]
        grid = (pos_w * 2.0 - 1.0)[..., [2, 1, 0]].view(voxel.shape[0], 1, 1, num_pos, 3)
        out = F.grid_sample(
            voxel,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return out.squeeze(2).squeeze(2).transpose(1, 2)

    def forward(self, x: torch.Tensor, pos_w: torch.Tensor) -> torch.Tensor:
        v0 = self._trilinear_splat(x, pos_w)
        e1 = self.enc1(v0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        u2 = F.interpolate(e3, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self._devoxelize(self.out(d1), pos_w)


class _ResMLP(nn.Module):
    def __init__(self, d: int, mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * mult),
            nn.GELU(),
            nn.Linear(d * mult, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BoundaryRefinement(nn.Module):
    def __init__(self, hidden: int, feat_dim: int):
        super().__init__()
        self.correction = nn.Sequential(
            nn.Linear(hidden + feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, h: torch.Tensor, raw_feat: torch.Tensor) -> torch.Tensor:
        corr = self.correction(torch.cat([h, raw_feat], dim=-1))
        gate = self.gate(raw_feat)
        return h + gate * corr


class SmoothSplatBackbone(nn.Module):
    """Single SmoothSplatNet member, architecture-compatible with V13 checkpoints."""

    def __init__(self):
        super().__init__()

        hidden = 256
        ch_base = 64
        grid = (64, 32, 32)
        num_pos_freqs = 10
        num_vel_freqs = 3
        num_dist_freqs = 6
        num_pre = 2
        num_post = 4

        pos_dim = 3 * (1 + 2 * num_pos_freqs)
        vin_dim = T_IN * 3
        vin_fourier_dim = 3 * 2 * num_vel_freqs
        dist_dim = 2 + 2 * num_dist_freqs
        in_dim = pos_dim + vin_dim + vin_fourier_dim + dist_dim + 1

        self.num_pos_freqs = num_pos_freqs
        self.num_vel_freqs = num_vel_freqs
        self.num_dist_freqs = num_dist_freqs

        self.warp = AdaptiveCoordinateWarp(n_knots=16)
        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks_pre = nn.ModuleList([_ResMLP(hidden) for _ in range(num_pre)])
        self.unet = _VoxelUNetSmooth(hidden, grid=grid, ch_base=ch_base)
        self.blocks_post = nn.ModuleList([_ResMLP(hidden) for _ in range(num_post)])
        self.norm_out = nn.LayerNorm(hidden)
        self.bar = BoundaryRefinement(hidden, in_dim)
        self.proj_out = nn.Linear(hidden, T_OUT * 3)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
        self.register_buffer("vel_std", torch.ones(1, 1, 1, 3))
        self.register_buffer("domain_min", DOMAIN_MIN.view(1, 1, 3))
        self.register_buffer("domain_max", DOMAIN_MAX.view(1, 1, 3))

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        del t
        batch_size, _, num_pos, _ = velocity_in.shape
        last_v = velocity_in[:, -1]
        v_norm = (velocity_in - self.vel_mean) / self.vel_std

        pos_feat = _fourier(pos, self.num_pos_freqs)
        vin_flat = v_norm.permute(0, 2, 1, 3).reshape(batch_size, num_pos, T_IN * 3)
        last_norm = v_norm[:, -1]

        vel_freqs = (
            2.0 ** torch.arange(self.num_vel_freqs, device=pos.device, dtype=pos.dtype)
            * math.pi
        )
        vel_fourier = last_norm.unsqueeze(-1) * vel_freqs
        vin_fourier = torch.cat([vel_fourier.sin(), vel_fourier.cos()], dim=-1).reshape(batch_size, num_pos, -1)

        airfoil_ind = torch.zeros(batch_size, num_pos, 1, device=pos.device, dtype=pos.dtype)
        for batch_idx, idx in enumerate(idcs_airfoil):
            airfoil_ind[batch_idx, idx.to(pos.device).long(), 0] = 1.0

        dist = _compute_dist(pos, idcs_airfoil)
        d = dist.unsqueeze(-1)
        d_log = torch.log1p(d)
        dist_freqs = (
            2.0 ** torch.arange(self.num_dist_freqs, device=pos.device, dtype=pos.dtype)
            * math.pi
        )
        dist_fourier = d_log * dist_freqs
        d_feat = torch.cat([d, d_log, dist_fourier.sin(), dist_fourier.cos()], dim=-1)

        feat = torch.cat([pos_feat, vin_flat, vin_fourier, d_feat, airfoil_ind], dim=-1)

        h = self.proj_in(feat)
        for block in self.blocks_pre:
            h = block(h)

        pos01 = (pos - self.domain_min) / (self.domain_max - self.domain_min)
        pos_warped = self.warp(pos01)
        h = h + self.unet(h, pos_warped)

        for block in self.blocks_post:
            h = block(h)
        h = self.norm_out(h)
        h = self.bar(h, feat)

        delta_norm = self.proj_out(h).reshape(batch_size, num_pos, T_OUT, 3)
        delta_norm = delta_norm.permute(0, 2, 1, 3)
        delta = delta_norm * self.vel_std
        pred = last_v.unsqueeze(1) + delta

        for batch_idx, idx in enumerate(idcs_airfoil):
            pred[batch_idx, :, idx.to(pred.device).long(), :] = 0.0

        return pred
