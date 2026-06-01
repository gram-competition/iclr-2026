from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class FourierFeatures(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        num_freq: int = 6,
        logspace: bool = True,
        include_input: bool = True,
        include_pi: bool = True,
        use_fp32_trig: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_freq = int(num_freq)
        self.include_input = bool(include_input)
        self.include_pi = bool(include_pi)
        self.use_fp32_trig = bool(use_fp32_trig)
        if self.num_freq <= 0:
            self.register_buffer("freq_bands", torch.empty(0, dtype=torch.float32), persistent=False)
        else:
            if logspace:
                freq_bands = 2.0 ** torch.linspace(0, self.num_freq - 1, self.num_freq)
            else:
                freq_bands = torch.linspace(1.0, 2.0 ** (self.num_freq - 1), self.num_freq)
            if self.include_pi:
                freq_bands = freq_bands * math.pi
            self.register_buffer("freq_bands", freq_bands.to(torch.float32), persistent=False)
        base = self.in_dim if self.include_input else 0
        self.out_dim = base + self.in_dim * 2 * self.num_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freq <= 0:
            return x if self.include_input else x.new_zeros((*x.shape[:-1], 0))
        if self.use_fp32_trig and x.is_cuda:
            ctx = torch.cuda.amp.autocast(enabled=False)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            x32 = x.float()
            xb = x32[..., :, None] * self.freq_bands
            sc = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)
            enc = sc.flatten(-2)
            out = torch.cat([x32, enc], dim=-1) if self.include_input else enc
        return out.to(dtype=x.dtype)


class HashGridEncoder3D(nn.Module):
    def __init__(
        self,
        num_levels: int = 8,
        features_per_level: int = 2,
        hash_table_size: int = 2 ** 18,
        min_resolution: int = 16,
        max_resolution: int = 256,
    ):
        super().__init__()
        self.num_levels = int(num_levels)
        self.features_per_level = int(features_per_level)
        self.hash_table_size = int(hash_table_size)
        self.min_resolution = int(min_resolution)
        self.max_resolution = int(max_resolution)
        self.growth = 1.0 if self.num_levels == 1 else math.exp(math.log(self.max_resolution / self.min_resolution) / (self.num_levels - 1))
        self.tables = nn.ModuleList()
        for _ in range(self.num_levels):
            emb = nn.Embedding(self.hash_table_size, self.features_per_level)
            nn.init.uniform_(emb.weight, a=-1e-4, b=1e-4)
            self.tables.append(emb)
        self.out_dim = self.num_levels * self.features_per_level
        self.register_buffer("p1", torch.tensor(1_540_863, dtype=torch.int64), persistent=False)
        self.register_buffer("p2", torch.tensor(1_256_879, dtype=torch.int64), persistent=False)
        self.register_buffer("p3", torch.tensor(1_957_123, dtype=torch.int64), persistent=False)
        self.register_buffer(
            "corner_offsets",
            torch.tensor([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]], dtype=torch.int64),
            persistent=False,
        )

    def _hash(self, ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
        h = (ix * self.p1) ^ (iy * self.p2) ^ (iz * self.p3)
        return torch.remainder(h, self.hash_table_size)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        x01f = x01.clamp(0.0, 1.0).float()
        n = x01f.shape[0]
        offsets = self.corner_offsets.to(device=x01f.device)
        feats = []
        for lvl in range(self.num_levels):
            res = int(math.floor(self.min_resolution * (self.growth ** lvl) + 1e-6))
            pos = x01f * float(res)
            i0 = torch.floor(pos).to(torch.int64)
            f = pos - i0.to(pos.dtype)
            corners = i0.unsqueeze(1) + offsets.view(1, 8, 3)

            fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
            ox = offsets[:, 0].view(1, 8).to(device=x01f.device)
            oy = offsets[:, 1].view(1, 8).to(device=x01f.device)
            oz = offsets[:, 2].view(1, 8).to(device=x01f.device)
            wx = torch.where(ox == 0, 1.0 - fx, fx).expand(n, 8)
            wy = torch.where(oy == 0, 1.0 - fy, fy).expand(n, 8)
            wz = torch.where(oz == 0, 1.0 - fz, fz).expand(n, 8)
            w = wx * wy * wz

            ix, iy, iz = corners[:, :, 0], corners[:, :, 1], corners[:, :, 2]
            idx = self._hash(ix, iy, iz).to(torch.long)
            emb = self.tables[lvl](idx)
            feats.append((emb * w.unsqueeze(-1)).sum(dim=1))
        return torch.cat(feats, dim=-1).to(dtype=x01.dtype)


class LatentTokenAttentionPool(nn.Module):
    def __init__(self, token_dim: int, d_model: int = 128, n_heads: int = 4, n_seeds: int = 8, out_dim: int = 192):
        super().__init__()
        self.proj = nn.Linear(token_dim, d_model)
        self.seeds = nn.Parameter(torch.randn(n_seeds, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        tok = self.proj(tokens)
        q = self.seeds.unsqueeze(0).expand(b, -1, -1)
        pooled, _ = self.attn(q, tok, tok, need_weights=False)
        pooled = pooled.mean(dim=1)
        return self.out(pooled)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class CheckpointedBlockStack(nn.Module):
    def __init__(self, dim: int, n_blocks: int, dropout: float = 0.0, use_checkpoint: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(dim, dropout=dropout) for _ in range(n_blocks)])
        self.use_checkpoint = bool(use_checkpoint)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class DensityMoEHead(nn.Module):
    def __init__(self, dim: int, out_dim: int = 3, n_experts: int = 3, expert_hidden: Optional[int] = None):
        super().__init__()
        hidden = int(expert_hidden or dim)
        self.n_experts = int(n_experts)
        self.router = nn.Sequential(nn.Linear(dim + 1, dim), nn.GELU(), nn.Linear(dim, self.n_experts))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
            for _ in range(self.n_experts)
        ])

    def forward(self, h: torch.Tensor, logh: torch.Tensor) -> torch.Tensor:
        gate_logits = self.router(torch.cat([h, logh], dim=-1))
        gate = torch.softmax(gate_logits, dim=-1)
        expert_outs = [expert(h) for expert in self.experts]
        stack = torch.stack(expert_outs, dim=-1)
        return (stack * gate.unsqueeze(-2)).sum(dim=-1)


class TemporalMixerBlock(nn.Module):
    def __init__(self, seq_len: int, dim: int, token_hidden: int, channel_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_fc1 = nn.Linear(seq_len, token_hidden)
        self.token_fc2 = nn.Linear(token_hidden, seq_len)
        self.norm2 = nn.LayerNorm(dim)
        self.channel_fc1 = nn.Linear(dim, channel_hidden)
        self.channel_fc2 = nn.Linear(channel_hidden, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x).transpose(1, 2)
        h = F.gelu(self.token_fc1(h))
        h = self.dropout(h)
        h = self.token_fc2(h).transpose(1, 2)
        x = x + h
        h = self.norm2(x)
        h = F.gelu(self.channel_fc1(h))
        h = self.dropout(h)
        h = self.channel_fc2(h)
        return x + h


class PointTemporalMLPMixer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        seq_len: int = 5,
        embed_dim: int = 24,
        n_blocks: int = 2,
        token_hidden: int = 12,
        channel_hidden: int = 48,
        proj_dim: int = 96,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.in_channels = int(in_channels)
        self.embed = nn.Linear(in_channels, embed_dim)
        self.blocks = nn.ModuleList([
            TemporalMixerBlock(seq_len, embed_dim, token_hidden, channel_hidden, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.stat_dim = seq_len * in_channels + 6 * in_channels
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 3 + self.stat_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.out_dim = proj_dim

    def forward(self, velocity_in: torch.Tensor) -> torch.Tensor:
        b, t, n, c = velocity_in.shape
        x = velocity_in.permute(0, 2, 1, 3).contiguous().view(b * n, t, c)
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        h_last = h[:, -1, :]
        h_mean = h.mean(dim=1)
        h_diff = h[:, -1, :] - h[:, -2, :] if t >= 2 else torch.zeros_like(h_last)

        hist_flat = x.reshape(b * n, t * c)
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        last = x[:, -1, :]
        delta1 = x[:, -1, :] - x[:, -2, :]
        delta2 = x[:, -2, :] - x[:, -3, :]
        acc = delta1 - delta2
        stats = torch.cat([hist_flat, mean, std, last, delta1, delta2, acc], dim=-1)

        feat = torch.cat([h_last, h_mean, h_diff, stats], dim=-1)
        feat = self.proj(feat)
        return feat.view(b, n, -1)


class ZoneAwareDirectDecoder(nn.Module):
    def __init__(self, dim: int, global_dim: int, time_dim: int = 64, n_experts: int = 3):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.Linear(4, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, dim),
        )
        self.global_proj = nn.Linear(global_dim, dim)
        self.near_affine = nn.Sequential(nn.Linear(dim + 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.far_affine = nn.Sequential(nn.Linear(dim + 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.zone_gate = nn.Sequential(nn.Linear(dim + 2, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.head = DensityMoEHead(dim=dim, out_dim=3, n_experts=n_experts, expert_hidden=dim)

    def forward(
        self,
        h: torch.Tensor,
        global_cond: torch.Tensor,
        logh: torch.Tensor,
        near_w: torch.Tensor,
        dist: torch.Tensor,
        future_q: torch.Tensor,
    ) -> torch.Tensor:
        base = h + self.global_proj(global_cond).unsqueeze(1) + self.time_proj(future_q).unsqueeze(1)
        zone_feat = torch.cat([near_w, dist], dim=-1)
        near_h = base + self.near_affine(torch.cat([base, zone_feat], dim=-1))
        far_h = base + self.far_affine(torch.cat([base, zone_feat], dim=-1))
        mix = torch.sigmoid(self.zone_gate(torch.cat([base, zone_feat], dim=-1)))
        mix = 0.5 * mix + 0.5 * near_w
        zone_h = mix * near_h + (1.0 - mix) * far_h
        return self.head(zone_h, logh)


@dataclass
class AeroChronoMixerConfig:
    fourier_freqs: int = 6
    use_hash: bool = True
    hash_num_levels: int = 8
    hash_features_per_level: int = 2
    hash_table_size: int = 2 ** 18
    hash_min_resolution: int = 16
    hash_max_resolution: int = 256

    temporal_embed_dim: int = 24
    temporal_mixer_blocks: int = 2
    temporal_token_hidden: int = 12
    temporal_channel_hidden: int = 48
    temporal_proj_dim: int = 96
    temporal_dropout: float = 0.0

    trunk_width: int = 192
    trunk_blocks: int = 6
    trunk_dropout: float = 0.0
    trunk_use_checkpoint: bool = False

    n_experts: int = 3
    global_token_dim: int = 128
    global_dim: int = 192
    global_token_points: int = 320
    max_boundary_anchors: int = 160

    blend_sigma: float = 0.05
    near_sigma: float = 0.04
    logh_voxel_size: float = 0.04
    cache_size: int = 4096


class AeroChronoMixerForecaster(nn.Module):
    def __init__(self, cfg: Optional[AeroChronoMixerConfig] = None):
        super().__init__()
        self.cfg = cfg or AeroChronoMixerConfig()

        self.ff = FourierFeatures(in_dim=3, num_freq=self.cfg.fourier_freqs)
        if self.cfg.use_hash:
            self.hash = HashGridEncoder3D(
                num_levels=self.cfg.hash_num_levels,
                features_per_level=self.cfg.hash_features_per_level,
                hash_table_size=self.cfg.hash_table_size,
                min_resolution=self.cfg.hash_min_resolution,
                max_resolution=self.cfg.hash_max_resolution,
            )
            hash_dim = self.hash.out_dim
        else:
            self.hash = None
            hash_dim = 0
        self.coord_dim = self.ff.out_dim + hash_dim
        self.ff_scale = nn.Parameter(torch.tensor(1.0))
        self.hash_scale = nn.Parameter(torch.tensor(1.0))

        self.temporal_encoder = PointTemporalMLPMixer(
            in_channels=3,
            seq_len=5,
            embed_dim=self.cfg.temporal_embed_dim,
            n_blocks=self.cfg.temporal_mixer_blocks,
            token_hidden=self.cfg.temporal_token_hidden,
            channel_hidden=self.cfg.temporal_channel_hidden,
            proj_dim=self.cfg.temporal_proj_dim,
            dropout=self.cfg.temporal_dropout,
        )

        self.boundary_dim = 7
        token_in_dim = 3 + self.boundary_dim + self.cfg.temporal_proj_dim + 3
        self.token_proj = nn.Sequential(
            nn.Linear(token_in_dim, self.cfg.global_token_dim),
            nn.GELU(),
            nn.Linear(self.cfg.global_token_dim, self.cfg.global_token_dim),
        )
        self.global_pool = LatentTokenAttentionPool(
            token_dim=self.cfg.global_token_dim,
            d_model=128,
            n_heads=4,
            n_seeds=8,
            out_dim=self.cfg.global_dim,
        )

        trunk_in = self.coord_dim + self.cfg.temporal_proj_dim + self.boundary_dim + 1 + 3
        self.input_proj = nn.Sequential(
            nn.Linear(trunk_in, self.cfg.trunk_width),
            nn.GELU(),
            nn.Linear(self.cfg.trunk_width, self.cfg.trunk_width),
        )
        self.trunk = CheckpointedBlockStack(
            dim=self.cfg.trunk_width,
            n_blocks=self.cfg.trunk_blocks,
            dropout=self.cfg.trunk_dropout,
            use_checkpoint=self.cfg.trunk_use_checkpoint,
        )
        self.decoder = ZoneAwareDirectDecoder(
            dim=self.cfg.trunk_width,
            global_dim=self.cfg.global_dim,
            time_dim=64,
            n_experts=self.cfg.n_experts,
        )

        self._boundary_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._density_cache: Dict[str, torch.Tensor] = {}

    def _cache_key(self, pos: torch.Tensor) -> str:
        p = pos.detach()
        idx = torch.tensor([0, max(0, p.shape[0] // 2), max(0, p.shape[0] - 1)], device=p.device)
        probe = p.index_select(0, idx).flatten().float().cpu()
        vals = ",".join(f"{x:.5f}" for x in probe.tolist())
        return f"N{p.shape[0]}:{vals}"

    def _normalize01(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lo = x.min(dim=0).values
        hi = x.max(dim=0).values
        x01 = (x - lo) / (hi - lo + 1e-6)
        return x01.clamp(0.0, 1.0), lo, hi

    @torch.no_grad()
    def _compute_logh_single(self, pos: torch.Tensor, key: str) -> torch.Tensor:
        if key in self._density_cache:
            return self._density_cache[key].to(device=pos.device, dtype=pos.dtype)
        x01, _, _ = self._normalize01(pos)
        voxel = torch.floor(x01 / self.cfg.logh_voxel_size).to(torch.int64)
        voxel = voxel.clamp(min=0)
        mul = torch.tensor([1, 4096, 4096 * 4096], dtype=torch.int64, device=pos.device)
        keys = (voxel * mul).sum(dim=-1)
        _, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
        cnt = counts[inverse].float()
        h = self.cfg.logh_voxel_size * torch.pow(cnt.clamp_min(1.0), -1.0 / 3.0)
        logh = torch.log(h.clamp_min(1e-6)).unsqueeze(-1)
        if len(self._density_cache) < self.cfg.cache_size:
            self._density_cache[key] = logh.detach().cpu()
        return logh

    @torch.no_grad()
    def _boundary_anchors(self, pos: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        boundary = pos.index_select(0, idx)
        if boundary.shape[0] <= self.cfg.max_boundary_anchors:
            return boundary
        sel = torch.linspace(0, boundary.shape[0] - 1, steps=self.cfg.max_boundary_anchors, device=boundary.device)
        return boundary.index_select(0, sel.round().long())

    @torch.no_grad()
    def _compute_boundary_single(self, pos: torch.Tensor, idx: torch.Tensor, key: str) -> Dict[str, torch.Tensor]:
        if key in self._boundary_cache:
            cached = self._boundary_cache[key]
            return {k: v.to(device=pos.device, dtype=pos.dtype) for k, v in cached.items()}

        n = pos.shape[0]
        mask = torch.zeros(n, 1, device=pos.device, dtype=pos.dtype)
        if idx.numel() == 0:
            zeros = torch.zeros(n, 1, device=pos.device, dtype=pos.dtype)
            dirs = torch.zeros(n, 3, device=pos.device, dtype=pos.dtype)
            near_rbf = torch.zeros(n, 1, device=pos.device, dtype=pos.dtype)
            out = {"mask": mask, "dist": zeros, "inv_dist": zeros, "dir": dirs, "near_rbf": near_rbf}
        else:
            idx = idx.long().unique(sorted=True)
            mask[idx, 0] = 1.0
            anchors = self._boundary_anchors(pos, idx)
            dists = []
            dirs = []
            chunk = 8192
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                pts = pos[s:e]
                diff = pts[:, None, :] - anchors[None, :, :]
                d2 = (diff ** 2).sum(dim=-1)
                min_d2, min_idx = d2.min(dim=1)
                vec = anchors.index_select(0, min_idx) - pts
                d = torch.sqrt(min_d2.clamp_min(1e-12)).unsqueeze(-1)
                dirs.append(vec / d.clamp_min(1e-6))
                dists.append(d)
            dist = torch.cat(dists, dim=0)
            dirs = torch.cat(dirs, dim=0)
            inv = 1.0 / (dist + 1e-3)
            near_rbf = torch.exp(-(dist ** 2) / (2.0 * self.cfg.near_sigma * self.cfg.near_sigma))
            out = {"mask": mask, "dist": dist, "inv_dist": inv, "dir": dirs, "near_rbf": near_rbf}
        if len(self._boundary_cache) < self.cfg.cache_size:
            self._boundary_cache[key] = {k: v.detach().cpu() for k, v in out.items()}
        return out

    def _encode_coords_single(self, pos: torch.Tensor) -> torch.Tensor:
        ff = self.ff(pos) * self.ff_scale
        if self.hash is None:
            return ff
        x01, _, _ = self._normalize01(pos)
        hg = self.hash(x01) * self.hash_scale
        return torch.cat([ff, hg], dim=-1)

    def _sample_token_indices(self, n: int, boundary_idx: torch.Tensor, device: torch.device) -> torch.Tensor:
        target = min(n, self.cfg.global_token_points)
        if target <= 0:
            return torch.zeros(1, dtype=torch.long, device=device)
        boundary_idx = boundary_idx.long().unique(sorted=True)
        selected = []
        if boundary_idx.numel() > 0:
            nb = min(boundary_idx.numel(), max(16, target // 3))
            sel = torch.linspace(0, boundary_idx.numel() - 1, steps=nb, device=device).round().long()
            selected.append(boundary_idx.index_select(0, sel))
        selected = torch.cat(selected, dim=0).unique(sorted=True) if len(selected) > 0 else torch.empty(0, dtype=torch.long, device=device)
        if selected.numel() < target:
            mask = torch.ones(n, dtype=torch.bool, device=device)
            if selected.numel() > 0:
                mask[selected] = False
            rest = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            need = target - selected.numel()
            if rest.numel() > 0:
                if need >= rest.numel():
                    fill = rest
                else:
                    sel = torch.linspace(0, rest.numel() - 1, steps=need, device=device).round().long()
                    fill = rest.index_select(0, sel)
                selected = torch.cat([selected, fill], dim=0).unique(sorted=True)
        if selected.numel() == 0:
            return torch.zeros(1, dtype=torch.long, device=device)
        return selected[:target]

    def _prepare_common(
        self,
        pos: torch.Tensor,
        idcs_airfoil: List[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b, n, _ = pos.shape
        temporal_feat = self.temporal_encoder(velocity_in)
        last_v = velocity_in[:, -1]

        coord_list = []
        boundary_feat_list = []
        logh_list = []
        global_tokens = []

        for bi in range(b):
            key = self._cache_key(pos[bi])
            coord_i = self._encode_coords_single(pos[bi])
            boundary = self._compute_boundary_single(pos[bi], idcs_airfoil[bi].to(pos.device), key)
            logh = self._compute_logh_single(pos[bi], key)
            boundary_feat_i = torch.cat([boundary["mask"], boundary["dist"], boundary["inv_dist"], boundary["dir"], boundary["near_rbf"]], dim=-1)
            idx = self._sample_token_indices(n, idcs_airfoil[bi].to(pos.device).long(), pos.device)
            tok = torch.cat([pos[bi, idx], boundary_feat_i[idx], temporal_feat[bi, idx], last_v[bi, idx]], dim=-1)
            tok = self.token_proj(tok)
            global_tokens.append(self.global_pool(tok.unsqueeze(0)).squeeze(0))

            coord_list.append(coord_i)
            boundary_feat_list.append(boundary_feat_i)
            logh_list.append(logh)

        coord_feat = torch.stack(coord_list, dim=0)
        boundary_feat = torch.stack(boundary_feat_list, dim=0)
        logh = torch.stack(logh_list, dim=0)
        global_cond = torch.stack(global_tokens, dim=0)
        near_w = torch.maximum(boundary_feat[..., 0:1], boundary_feat[..., -1:])

        trunk_in = torch.cat([coord_feat, temporal_feat, boundary_feat, logh, last_v], dim=-1)
        h = self.input_proj(trunk_in)
        h = self.trunk(h)
        return {
            "h": h,
            "global_cond": global_cond,
            "logh": logh,
            "near_w": near_w,
            "dist": boundary_feat[..., 1:2],
        }

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: List[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        common = self._prepare_common(pos, idcs_airfoil, velocity_in)
        t_future = t[:, 5:]
        t_last = t[:, 4:5]
        dt_prev = (t[:, 4:5] - t[:, 3:4]).clamp_min(1e-6)
        outs = []
        for k in range(t_future.shape[1]):
            tf = t_future[:, k:k+1]
            dt = tf - t_last
            ratio = dt / dt_prev
            q = torch.cat([
                tf,
                dt,
                ratio,
                torch.full_like(tf, float(k + 1) / float(t_future.shape[1])),
            ], dim=-1)
            step = self.decoder(common["h"], common["global_cond"], common["logh"], common["near_w"], common["dist"], q)
            outs.append(step)
        out = torch.stack(outs, dim=1)
        for bi, idx in enumerate(idcs_airfoil):
            idx = idx.long().to(pos.device)
            if idx.numel() > 0:
                out[bi, :, idx, :] = 0.0
        return out


class AeroChronoMixer(AeroChronoMixerForecaster):
    """Competition-ready wrapper that loads trained weights during construction."""

    DEFAULT_WEIGHT_CANDIDATES = (
        "state_dict.pt",
    )

    def __init__(self):
        super().__init__(AeroChronoMixerConfig())
        self.eval()
        self._load_submission_weights()

    def _resolve_weight_path(self) -> Path:
        env_path = os.getenv("ACM_WEIGHTS")
        candidates: List[Path] = []
        if env_path:
            candidates.append(Path(env_path))
        base = Path(__file__).resolve().parent
        for item in self.DEFAULT_WEIGHT_CANDIDATES:
            p = Path(item)
            if not p.is_absolute():
                p = base / item
            candidates.append(p)
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            "Could not find trained weights. Place 'state_dict.pt' next to model.py, "
            "or set ACM_WEIGHTS to an absolute checkpoint path."
        )

    def _extract_state_dict(self, obj: object) -> Dict[str, torch.Tensor]:
        if isinstance(obj, dict):
            if "ema" in obj and isinstance(obj["ema"], dict) and len(obj["ema"]) > 0:
                return obj["ema"]
            if "model" in obj and isinstance(obj["model"], dict):
                return obj["model"]
        if isinstance(obj, dict):
            return obj  # raw state_dict.pt
        raise RuntimeError("Unsupported checkpoint format.")

    def _load_submission_weights(self) -> None:
        ckpt_path = self._resolve_weight_path()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = self._extract_state_dict(ckpt)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Weight loading mismatch for {ckpt_path}. Missing keys: {missing[:10]}; unexpected keys: {unexpected[:10]}"
            )
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
