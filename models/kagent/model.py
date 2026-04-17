"""kagent submission."""

import math
import os

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


def _compute_dist(pos: torch.Tensor, idcs_airfoil: list) -> torch.Tensor:
    B, N, _ = pos.shape
    out = torch.empty(B, N, device=pos.device, dtype=pos.dtype)
    for b in range(B):
        idx = idcs_airfoil[b].to(pos.device).long()
        af = pos[b].index_select(0, idx)
        chunks = [torch.cdist(chunk, af).min(dim=1).values for chunk in pos[b].split(8192)]
        out[b] = torch.cat(chunks, dim=0)
    return out


class _DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _VoxelUNet(nn.Module):
    def __init__(self, hidden, grid=(64, 32, 32), ch_base=64):
        super().__init__()
        self.grid = tuple(grid)
        self.enc1 = _DoubleConv(hidden, ch_base)
        self.enc2 = _DoubleConv(ch_base, ch_base * 2)
        self.enc3 = _DoubleConv(ch_base * 2, ch_base * 4)
        self.dec2 = _DoubleConv(ch_base * 4 + ch_base * 2, ch_base * 2)
        self.dec1 = _DoubleConv(ch_base * 2 + ch_base, ch_base)
        self.out = nn.Conv3d(ch_base, hidden, 1)
        self.pool = nn.MaxPool3d(2)

    def _voxelize(self, x, pos01):
        B, N, C = x.shape
        Gx, Gy, Gz = self.grid
        idx = pos01.clamp(0, 1 - 1e-6)
        idx = torch.stack([
            (idx[..., 0] * Gx).floor().long(),
            (idx[..., 1] * Gy).floor().long(),
            (idx[..., 2] * Gz).floor().long(),
        ], dim=-1)
        flat = idx[..., 0] * (Gy * Gz) + idx[..., 1] * Gz + idx[..., 2]
        voxel = torch.zeros(B, C, Gx * Gy * Gz, device=x.device, dtype=x.dtype)
        count = torch.zeros(B, 1, Gx * Gy * Gz, device=x.device, dtype=x.dtype)
        voxel.scatter_add_(2, flat.unsqueeze(1).expand(-1, C, -1), x.transpose(1, 2))
        ones = torch.ones(B, 1, N, device=x.device, dtype=x.dtype)
        count.scatter_add_(2, flat.unsqueeze(1), ones)
        voxel = voxel / count.clamp(min=1.0)
        return voxel.view(B, C, Gx, Gy, Gz)

    def _devoxelize(self, voxel, pos01):
        B, C = voxel.shape[:2]
        N = pos01.shape[1]
        p = pos01 * 2.0 - 1.0
        grid = p[..., [2, 1, 0]].view(B, 1, 1, N, 3)
        out = F.grid_sample(voxel, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return out.squeeze(2).squeeze(2).transpose(1, 2)

    def forward(self, x, pos01):
        v0 = self._voxelize(x, pos01)
        e1 = self.enc1(v0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        u2 = F.interpolate(e3, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self._devoxelize(self.out(d1), pos01)


class _ResMLP(nn.Module):
    def __init__(self, d, mult=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * mult),
            nn.GELU(),
            nn.Linear(d * mult, d),
        )

    def forward(self, x):
        return x + self.net(x)


class _VoxelUNetModel(nn.Module):
    def __init__(self, hidden=256, num_pre=2, num_post=4,
                 grid=(64, 32, 32), ch_base=64,
                 num_pos_freqs=10, num_vel_freqs=3, num_dist_freqs=6):
        super().__init__()
        pos_dim = 3 * (1 + 2 * num_pos_freqs)
        vin_dim = T_IN * 3
        vin_fourier_dim = 3 * 2 * num_vel_freqs
        dist_dim = 2 + 2 * num_dist_freqs
        in_dim = pos_dim + vin_dim + vin_fourier_dim + dist_dim + 1

        self.num_pos_freqs = num_pos_freqs
        self.num_vel_freqs = num_vel_freqs
        self.num_dist_freqs = num_dist_freqs

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks_pre = nn.ModuleList([_ResMLP(hidden) for _ in range(num_pre)])
        self.unet = _VoxelUNet(hidden, grid=grid, ch_base=ch_base)
        self.blocks_post = nn.ModuleList([_ResMLP(hidden) for _ in range(num_post)])
        self.norm_out = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, T_OUT * 3)

        self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
        self.register_buffer("vel_std", torch.ones(1, 1, 1, 3))
        self.register_buffer("domain_min", DOMAIN_MIN.view(1, 1, 3))
        self.register_buffer("domain_max", DOMAIN_MAX.view(1, 1, 3))

    def forward(self, velocity_in, pos, idcs_airfoil, dist_airfoil):
        B, T, N, C = velocity_in.shape
        last_v = velocity_in[:, -1]
        v_norm = (velocity_in - self.vel_mean) / self.vel_std

        pos_feat = _fourier(pos, self.num_pos_freqs)
        vin_flat = v_norm.permute(0, 2, 1, 3).reshape(B, N, T_IN * 3)
        last_norm = v_norm[:, -1]
        vfreqs = 2.0 ** torch.arange(self.num_vel_freqs, device=pos.device, dtype=pos.dtype) * math.pi
        vf = last_norm.unsqueeze(-1) * vfreqs
        vin_fourier = torch.cat([vf.sin(), vf.cos()], dim=-1).reshape(B, N, -1)

        airfoil_ind = torch.zeros(B, N, 1, device=pos.device, dtype=pos.dtype)
        for b, idx in enumerate(idcs_airfoil):
            airfoil_ind[b, idx.to(pos.device).long(), 0] = 1.0

        d = dist_airfoil.unsqueeze(-1)
        d_log = torch.log1p(d)
        dfreqs = 2.0 ** torch.arange(self.num_dist_freqs, device=pos.device, dtype=pos.dtype) * math.pi
        df = d_log * dfreqs
        d_feat = torch.cat([d, d_log, df.sin(), df.cos()], dim=-1)

        feat = torch.cat([pos_feat, vin_flat, vin_fourier, d_feat, airfoil_ind], dim=-1)
        h = self.proj_in(feat)
        for blk in self.blocks_pre:
            h = blk(h)

        pos01 = (pos - self.domain_min) / (self.domain_max - self.domain_min)
        h = h + self.unet(h, pos01)

        for blk in self.blocks_post:
            h = blk(h)

        h = self.norm_out(h)
        delta_norm = self.proj_out(h).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)
        delta = delta_norm * self.vel_std
        pred = last_v.unsqueeze(1) + delta

        for b, idx in enumerate(idcs_airfoil):
            pred[b, :, idx.to(pred.device).long(), :] = 0.0

        return pred


def _build_and_load(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ch_base = sd["unet.enc1.block.0.weight"].shape[0]
    m = _VoxelUNetModel(hidden=256, num_pre=2, num_post=4,
                        grid=(64, 32, 32), ch_base=ch_base)
    m.load_state_dict(sd)
    return m


class Model(nn.Module):
    """Signature: (t, pos, idcs_airfoil, velocity_in) -> velocity_out."""

    _WEIGHTS = (0.325 / 0.675, 0.35 / 0.675)

    def __init__(self):
        super().__init__()
        here = os.path.join("models", "kagent")
        self.members = nn.ModuleList([
            _build_and_load(os.path.join(here, "state_dict_1.pt")),
            _build_and_load(os.path.join(here, "state_dict_2.pt")),
        ])

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list,
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        dist = _compute_dist(pos, idcs_airfoil)

        pos_f = pos.clone()
        pos_f[..., 1] = -pos_f[..., 1]
        v_f = velocity_in.clone()
        v_f[..., 1] = -v_f[..., 1]
        dist_f = _compute_dist(pos_f, idcs_airfoil)

        agg = None
        for m, w in zip(self.members, self._WEIGHTS):
            out_d = m(velocity_in, pos, idcs_airfoil, dist)
            out_f = m(v_f, pos_f, idcs_airfoil, dist_f)
            out_f = out_f.clone()
            out_f[..., 1] = -out_f[..., 1]
            avg = 0.5 * (out_d + out_f)
            agg = avg * w if agg is None else agg + avg * w
        return agg
