from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download


TI = 5
TO = 5
W = (0.325 / 0.675, 0.35 / 0.675)
A = torch.tensor([0.0, -0.41, 0.0], dtype=torch.float32)
B = torch.tensor([2.10, 0.41, 1.22], dtype=torch.float32)


def _f(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype) * math.pi
    xf = x.unsqueeze(-1) * freqs
    enc = torch.cat([xf.sin(), xf.cos()], dim=-1).reshape(*x.shape[:-1], -1)
    return torch.cat([x, enc], dim=-1)


def _d(pos: torch.Tensor, idcs_airfoil: list[torch.Tensor]) -> torch.Tensor:
    batch_size, num_points, _ = pos.shape
    out = torch.empty(batch_size, num_points, device=pos.device, dtype=pos.dtype)
    for batch_idx in range(batch_size):
        idx = idcs_airfoil[batch_idx].to(pos.device).long()
        airfoil_pts = pos[batch_idx].index_select(0, idx)
        chunks = [torch.cdist(chunk, airfoil_pts).min(dim=1).values for chunk in pos[batch_idx].split(8192)]
        out[batch_idx] = torch.cat(chunks, dim=0)
    return out


class _C(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _U(nn.Module):
    def __init__(self, hidden: int, grid: tuple[int, int, int], ch_base: int):
        super().__init__()
        self.grid = tuple(int(v) for v in grid)
        self.enc1 = _C(hidden, ch_base)
        self.enc2 = _C(ch_base, ch_base * 2)
        self.enc3 = _C(ch_base * 2, ch_base * 4)
        self.dec2 = _C(ch_base * 4 + ch_base * 2, ch_base * 2)
        self.dec1 = _C(ch_base * 2 + ch_base, ch_base)
        self.out = nn.Conv3d(ch_base, hidden, 1)
        self.pool = nn.MaxPool3d(2)

    def _voxelize(self, x: torch.Tensor, pos01: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, channels = x.shape
        grid_x, grid_y, grid_z = self.grid
        idx = pos01.clamp(0, 1 - 1e-6)
        idx = torch.stack(
            [
                (idx[..., 0] * grid_x).floor().long(),
                (idx[..., 1] * grid_y).floor().long(),
                (idx[..., 2] * grid_z).floor().long(),
            ],
            dim=-1,
        )
        flat = idx[..., 0] * (grid_y * grid_z) + idx[..., 1] * grid_z + idx[..., 2]
        voxel = torch.zeros(batch_size, channels, grid_x * grid_y * grid_z, device=x.device, dtype=x.dtype)
        count = torch.zeros(batch_size, 1, grid_x * grid_y * grid_z, device=x.device, dtype=x.dtype)
        voxel.scatter_add_(2, flat.unsqueeze(1).expand(-1, channels, -1), x.transpose(1, 2))
        ones = torch.ones(batch_size, 1, num_points, device=x.device, dtype=x.dtype)
        count.scatter_add_(2, flat.unsqueeze(1), ones)
        voxel = voxel / count.clamp(min=1.0)
        return voxel.view(batch_size, channels, grid_x, grid_y, grid_z)

    def _devoxelize(self, voxel: torch.Tensor, pos01: torch.Tensor) -> torch.Tensor:
        batch_size, channels = voxel.shape[:2]
        num_points = pos01.shape[1]
        pos_grid = pos01 * 2.0 - 1.0
        sample_grid = pos_grid[..., [2, 1, 0]].view(batch_size, 1, 1, num_points, 3)
        out = F.grid_sample(voxel, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)
        return out.squeeze(2).squeeze(2).transpose(1, 2)

    def forward(self, x: torch.Tensor, pos01: torch.Tensor) -> torch.Tensor:
        v0 = self._voxelize(x, pos01)
        e1 = self.enc1(v0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        u2 = F.interpolate(e3, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self._devoxelize(self.out(d1), pos01)


class _R(nn.Module):
    def __init__(self, dim: int, mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _M(nn.Module):
    def __init__(
        self,
        hidden: int = 256,
        num_pre: int = 2,
        num_post: int = 4,
        grid: tuple[int, int, int] = (64, 32, 32),
        ch_base: int = 64,
        num_pos_freqs: int = 10,
        num_vel_freqs: int = 3,
        num_dist_freqs: int = 6,
    ):
        super().__init__()
        pos_dim = 3 * (1 + 2 * num_pos_freqs)
        vel_in_dim = TI * 3
        vel_fourier_dim = 3 * 2 * num_vel_freqs
        dist_dim = 2 + 2 * num_dist_freqs
        in_dim = pos_dim + vel_in_dim + vel_fourier_dim + dist_dim + 1
        self.num_pos_freqs = num_pos_freqs
        self.num_vel_freqs = num_vel_freqs
        self.num_dist_freqs = num_dist_freqs
        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks_pre = nn.ModuleList([_R(hidden) for _ in range(num_pre)])
        self.unet = _U(hidden, grid=grid, ch_base=ch_base)
        self.blocks_post = nn.ModuleList([_R(hidden) for _ in range(num_post)])
        self.norm_out = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, TO * 3)
        self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
        self.register_buffer("vel_std", torch.ones(1, 1, 1, 3))
        self.register_buffer("domain_min", A.view(1, 1, 3))
        self.register_buffer("domain_max", B.view(1, 1, 3))

    def forward(
        self,
        velocity_in: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        dist_airfoil: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_steps, num_points, _ = velocity_in.shape
        last_velocity = velocity_in[:, -1]
        vel_norm = (velocity_in - self.vel_mean) / self.vel_std
        pos_feat = _f(pos, self.num_pos_freqs)
        vel_flat = vel_norm.permute(0, 2, 1, 3).reshape(batch_size, num_points, num_steps * 3)
        last_norm = vel_norm[:, -1]
        vel_freqs = 2.0 ** torch.arange(self.num_vel_freqs, device=pos.device, dtype=pos.dtype) * math.pi
        vel_fourier = last_norm.unsqueeze(-1) * vel_freqs
        vel_fourier = torch.cat([vel_fourier.sin(), vel_fourier.cos()], dim=-1).reshape(batch_size, num_points, -1)
        airfoil_indicator = torch.zeros(batch_size, num_points, 1, device=pos.device, dtype=pos.dtype)
        for batch_idx, idx in enumerate(idcs_airfoil):
            airfoil_indicator[batch_idx, idx.to(pos.device).long(), 0] = 1.0
        dist = dist_airfoil.unsqueeze(-1)
        dist_log = torch.log1p(dist)
        dist_freqs = 2.0 ** torch.arange(self.num_dist_freqs, device=pos.device, dtype=pos.dtype) * math.pi
        dist_fourier = dist_log * dist_freqs
        dist_feat = torch.cat([dist, dist_log, dist_fourier.sin(), dist_fourier.cos()], dim=-1)
        feat = torch.cat([pos_feat, vel_flat, vel_fourier, dist_feat, airfoil_indicator], dim=-1)
        hidden = self.proj_in(feat)
        for block in self.blocks_pre:
            hidden = block(hidden)
        pos01 = (pos - self.domain_min) / (self.domain_max - self.domain_min)
        hidden = hidden + self.unet(hidden, pos01)
        for block in self.blocks_post:
            hidden = block(hidden)
        hidden = self.norm_out(hidden)
        delta_norm = self.proj_out(hidden).reshape(batch_size, num_points, TO, 3).permute(0, 2, 1, 3)
        delta = delta_norm * self.vel_std
        pred = last_velocity.unsqueeze(1) + delta
        for batch_idx, idx in enumerate(idcs_airfoil):
            pred[batch_idx, :, idx.to(pred.device).long(), :] = 0.0
        return pred


def _cfg(ckpt_path: Path) -> dict[str, object]:
    cfg_path = ckpt_path.parent / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def _load(model: _M, state_dict: dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    copied_state = dict(state_dict)
    for key, target_tensor in model_state.items():
        loaded_tensor = copied_state.get(key)
        if loaded_tensor is None or loaded_tensor.shape == target_tensor.shape:
            continue
        if loaded_tensor.ndim != target_tensor.ndim:
            continue
        patched_tensor = torch.zeros_like(target_tensor) if key in {"proj_in.weight", "unet.enc1.block.0.weight"} else target_tensor.clone()
        overlap = tuple(slice(0, min(dst, src)) for dst, src in zip(target_tensor.shape, loaded_tensor.shape))
        if not overlap:
            continue
        patched_tensor[overlap] = loaded_tensor[overlap].to(dtype=patched_tensor.dtype)
        copied_state[key] = patched_tensor
    model.load_state_dict(copied_state, strict=False)


def _build(ckpt_path: str | Path) -> _M:
    ckpt_path = Path(ckpt_path)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    saved_config = _cfg(ckpt_path)
    grid = tuple(int(v) for v in saved_config.get("voxel_grid", [64, 32, 32]))
    ch_base = int(saved_config.get("voxel_base_channels", int(state_dict["unet.enc1.block.0.weight"].shape[0])))
    hidden = int(state_dict["proj_in.weight"].shape[0])
    model = _M(hidden=hidden, num_pre=2, num_post=4, grid=grid, ch_base=ch_base)
    _load(model, state_dict)
    return model


def _tta(
    member: _M,
    velocity_in: torch.Tensor,
    pos: torch.Tensor,
    idcs_airfoil: list[torch.Tensor],
    airfoil_distance: torch.Tensor,
) -> torch.Tensor:
    direct = member(velocity_in, pos, idcs_airfoil, airfoil_distance)
    pos_flip = pos.clone()
    pos_flip[..., 1] = -pos_flip[..., 1]
    vel_flip = velocity_in.clone()
    vel_flip[..., 1] = -vel_flip[..., 1]
    flipped = member(vel_flip, pos_flip, idcs_airfoil, airfoil_distance)
    flipped = flipped.clone()
    flipped[..., 1] = -flipped[..., 1]
    return 0.5 * (direct + flipped)


class SubmissionModel(nn.Module):
    def __init__(self):
        super().__init__()
        manifest = json.loads((Path(__file__).resolve().parent / "manifest.json").read_text())
        root = Path(snapshot_download(repo_id=manifest["repo_id"], repo_type="model"))
        self.alphas = [float(pair["alpha"]) for pair in manifest["pairs"]]
        self.pairs = nn.ModuleList(
            [
                nn.ModuleList([_build(root / member / "best_state_dict.pt") for member in pair["members"]])
                for pair in manifest["pairs"]
            ]
        )

    def _pair_predict(
        self,
        members: nn.ModuleList,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
        airfoil_distance: torch.Tensor,
    ) -> torch.Tensor:
        agg = None
        for idx, member in enumerate(members):
            pred = _tta(member, velocity_in, pos, idcs_airfoil, airfoil_distance)
            weighted = pred * float(W[idx])
            agg = weighted if agg is None else agg + weighted
        return agg

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        del t
        airfoil_distance = _d(pos, idcs_airfoil)
        out = None
        for alpha, members in zip(self.alphas, self.pairs):
            pred = self._pair_predict(members, pos, idcs_airfoil, velocity_in, airfoil_distance)
            weighted = pred * alpha
            out = weighted if out is None else out + weighted
        return out
