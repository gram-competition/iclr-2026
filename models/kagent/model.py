"""kagent submission — voxel-UNet residual model with SDF wall-distance feature.

Trained autonomously by a coding agent ("alphonse") on the public warped-ifw
dataset. Best checkpoint at val/l2_error = 0.8707.

Architecture:
  - Normalise velocities per-component (buffers from stats.json)
  - Per-point features: [5 past velocities, pos, airfoil mask, sdf, log1p(sdf)]
  - ResMLP pre-blocks, 3D voxel-UNet spatial mix, ResMLP post-blocks
  - Predict residual from last input frame in normalised space
  - Enforce no-slip by zeroing velocity at idcs_airfoil
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


T_IN = 5
T_OUT = 5


def _compute_sdf(pos: torch.Tensor, airfoil_idcs: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Per-point Euclidean distance to the nearest airfoil point (single sample)."""
    a = pos[airfoil_idcs.to(pos.device)]
    sdf = torch.full((pos.shape[0],), float("inf"), device=pos.device, dtype=pos.dtype)
    for s in range(0, a.shape[0], chunk):
        d = torch.cdist(pos, a[s:s + chunk]).min(dim=-1).values
        sdf = torch.minimum(sdf, d)
    return sdf


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ConvBlock3D(nn.Module):
    def __init__(self, c_in, c_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(groups, c_out),
            nn.GELU(),
            nn.Conv3d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(groups, c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, c_in, c_mid=64, c_out=None, groups=8):
        super().__init__()
        c_out = c_out or c_in
        self.enc1 = ConvBlock3D(c_in, c_mid, groups)
        self.enc2 = ConvBlock3D(c_mid, c_mid * 2, groups)
        self.enc3 = ConvBlock3D(c_mid * 2, c_mid * 4, groups)
        self.dec2 = ConvBlock3D(c_mid * 2 + c_mid * 4, c_mid * 2, groups)
        self.dec1 = ConvBlock3D(c_mid + c_mid * 2, c_mid, groups)
        self.out = nn.Conv3d(c_mid, c_out, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool3d(e1, 2))
        e3 = self.enc3(F.avg_pool3d(e2, 2))
        d2 = self.dec2(torch.cat(
            [F.interpolate(e3, scale_factor=2, mode="trilinear", align_corners=False), e2], dim=1
        ))
        d1 = self.dec1(torch.cat(
            [F.interpolate(d2, scale_factor=2, mode="trilinear", align_corners=False), e1], dim=1
        ))
        return self.out(d1)


class VoxelSpatial(nn.Module):
    def __init__(self, dim, res=64, unet_mid=64, pad=0.05):
        super().__init__()
        self.res = res
        self.pad = pad
        self.unet = UNet3D(c_in=dim, c_mid=unet_mid, c_out=dim)
        nn.init.zeros_(self.unet.out.weight)
        nn.init.zeros_(self.unet.out.bias)

    def forward(self, feats, pos):
        B, N, D = feats.shape
        R = self.res
        lo = pos.amin(dim=1, keepdim=True) - self.pad
        hi = pos.amax(dim=1, keepdim=True) + self.pad
        p01 = (pos - lo) / (hi - lo).clamp(min=1e-6)
        idx = (p01 * R).long().clamp(0, R - 1)
        flat = idx[..., 0] * R * R + idx[..., 1] * R + idx[..., 2]

        vox = feats.new_zeros(B, D, R * R * R)
        cnt = feats.new_zeros(B, 1, R * R * R)
        vox.scatter_add_(2, flat.unsqueeze(1).expand(-1, D, -1), feats.transpose(1, 2))
        cnt.scatter_add_(2, flat.unsqueeze(1),
                         torch.ones_like(flat, dtype=feats.dtype).unsqueeze(1))
        vox = vox / cnt.clamp(min=1.0)
        vox = vox.view(B, D, R, R, R)

        vox = self.unet(vox)

        grid = (p01 * 2 - 1)[:, None, None, :, [2, 1, 0]]
        sampled = F.grid_sample(vox, grid, mode="bilinear",
                                align_corners=False, padding_mode="border")
        sampled = sampled.squeeze(2).squeeze(2).transpose(1, 2)
        return feats + sampled


class VoxelResidualModel(nn.Module):
    def __init__(self, hidden=256, voxel_res=64, voxel_mid=64,
                 n_blocks_pre=2, n_blocks_post=4):
        super().__init__()
        in_dim = T_IN * 3 + 3 + 1 + 2
        out_dim = T_OUT * 3
        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks_pre = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks_pre)])
        self.spatial = VoxelSpatial(dim=hidden, res=voxel_res, unet_mid=voxel_mid)
        self.blocks_post = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks_post)])
        self.norm_out = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
        self.register_buffer("vel_std", torch.ones(1, 1, 1, 3))

    def forward(self, velocity_in, pos, idcs_airfoil, sdf):
        B, T, N, C = velocity_in.shape
        v_norm = (velocity_in - self.vel_mean) / self.vel_std
        v_feat = v_norm.permute(0, 2, 1, 3).reshape(B, N, T * C)

        mask = torch.zeros(B, N, 1, device=velocity_in.device, dtype=velocity_in.dtype)
        for b, idcs in enumerate(idcs_airfoil):
            mask[b, idcs.to(mask.device), 0] = 1.0

        sdf_raw = (sdf / 5.0).unsqueeze(-1)
        sdf_log = torch.log1p(sdf).unsqueeze(-1)

        x = torch.cat([v_feat, pos, mask, sdf_raw, sdf_log], dim=-1)
        x = self.proj_in(x)
        x = self.blocks_pre(x)
        x = self.spatial(x, pos)
        x = self.blocks_post(x)
        x = self.norm_out(x)
        delta_norm = self.proj_out(x).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)
        delta = delta_norm * self.vel_std

        last_frame = velocity_in[:, -1:].expand(-1, T_OUT, -1, -1)
        pred = last_frame + delta

        no_slip = torch.ones(B, 1, N, 1, device=pred.device, dtype=pred.dtype)
        for b, idcs in enumerate(idcs_airfoil):
            no_slip[b, 0, idcs.to(no_slip.device), 0] = 0.0
        return pred * no_slip


class Model(nn.Module):
    """Entry point with the competition signature.

    Computes the per-sample SDF (wall distance) on the fly since the signature
    does not receive it, then delegates to the voxel-UNet residual model.
    """

    def __init__(self):
        super().__init__()
        self.net = VoxelResidualModel(
            hidden=256, voxel_res=64, voxel_mid=64,
            n_blocks_pre=2, n_blocks_post=4,
        )
        state_dict = torch.load(
            os.path.join("models", "kagent", "state_dict.pt"),
            map_location="cpu",
        )
        self.net.load_state_dict(state_dict)

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        B = pos.shape[0]
        sdf = torch.stack([_compute_sdf(pos[b], idcs_airfoil[b]) for b in range(B)])
        return self.net(velocity_in, pos, idcs_airfoil, sdf)
