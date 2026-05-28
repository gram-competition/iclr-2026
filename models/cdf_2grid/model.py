"""CDFDoubleGridModel — Multi-scale U-Net with temporal attention.

  - Multi-scale voxel pyramid: scatter into coarse AND fine grids dynamically.
  - Density-Adaptive CDF Mapping: Grid lines dynamically warp to follow point density.
  - Temporal cross-attention: 5 input frames → per-point temporal summary via
    a lightweight multi-head attention across the T dimension.
  - Fourier positional encoding.
  - Learned SDF embedding: small MLP maps log-sdf → 16-d embedding.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Constants

T_IN  = 5
T_OUT = 5
FOURIER_BANDS = 8   # number of frequency bands for positional encoding
SDF_EMB_DIM   = 16


# Utility: SDF & CDF computations

def compute_sdf_batch(
    pos: torch.Tensor,           # (B, N, 3)
    idcs_airfoil: list[torch.Tensor],
    chunk: int = 8192,           
) -> torch.Tensor:               # (B, N)
    """Euclidean distance from every domain point to the nearest airfoil point."""
    B, N, _ = pos.shape
    device = pos.device
    
    sdf = torch.zeros((B, N), device=device, dtype=pos.dtype)
    
    for b in range(B):
        surf = pos[b, idcs_airfoil[b].to(device)]   # (S, 3)
        
        for i in range(0, N, chunk):
            end = min(i + chunk, N)
            
            # dists shape: (chunk_size, S)
            dists = torch.cdist(pos[b, i:end], surf) 
            
            sdf[b, i:end] = dists.min(dim=-1).values
            
    return sdf



def compute_regularized_amr_metrics(
    pos_tensor: torch.Tensor,
    resolutions: tuple[int, ...] = (32, 64),
    sigma: float = 2.0,
    beta: float = 0.3,
    eps: float = 1e-6,
):
    device = pos_tensor.device
    pos_comp = torch.zeros_like(pos_tensor)
    widths_dict = {res: torch.zeros(3, res, device=device) for res in resolutions}
    max_res = max(resolutions)
    bins = max_res * 16 

    for d in range(3):
        coords, indices = torch.sort(pos_tensor[:, d])

        # 1. Histogram (density estimate on FULL data)
        hist = torch.histc(coords, bins=bins, min=0.0, max=1.0)
        uniform = torch.ones_like(hist)
        hist = (1 - beta) * hist + beta * uniform * hist.mean()

        # 2. Gaussian smoothing
        kernel_size = int(sigma * 6) | 1
        x = torch.linspace(-3 * sigma, 3 * sigma, kernel_size, device=device)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = (kernel / kernel.sum()).view(1, 1, -1)

        smoothed = F.conv1d(hist.view(1, 1, -1), kernel, padding=kernel_size // 2).view(-1)

        # 3. CDF construction
        cdf = torch.cumsum(smoothed, dim=0)
        cdf = cdf / cdf[-1]
        cdf[0] = 0.0
        cdf = cdf / cdf[-1]

        # 4. Forward map: x → CDF(x)
        bin_centers = torch.linspace(0, 1, bins, device=device)
        inds = torch.searchsorted(bin_centers, coords).clamp(1, bins - 1)

        x0, x1 = bin_centers[inds - 1], bin_centers[inds]
        y0, y1 = cdf[inds - 1], cdf[inds]

        t = (coords - x0) / (x1 - x0 + eps)
        pos_comp[indices, d] = y0 + t * (y1 - y0)

        # 5. Inverse map: CDF⁻¹ → voxel boundaries
        for res in resolutions:
            grid = torch.linspace(0, 1, res + 1, device=device)
            inds = torch.searchsorted(cdf, grid).clamp(1, bins - 1)

            f0, f1 = cdf[inds - 1], cdf[inds]
            x0, x1 = bin_centers[inds - 1], bin_centers[inds]

            t = (grid - f0) / (f1 - f0 + eps)
            bounds = x0 + t * (x1 - x0)
            widths_dict[res][d] = (bounds[1:] - bounds[:-1]).clamp(min=eps)

    return pos_comp, widths_dict


# Feature Embeddings

class FourierPosEnc(nn.Module):
    def __init__(self, n_bands: int = FOURIER_BANDS):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_bands).float() * math.pi
        self.register_buffer("freqs", freqs)
        self.out_dim = 3 * 2 * n_bands

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        x = pos.unsqueeze(-1) * self.freqs
        enc = torch.cat([x.sin(), x.cos()], dim=-1)
        return enc.flatten(-2)


class SDFEmbedding(nn.Module):
    def __init__(self, out_dim: int = SDF_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, sdf: torch.Tensor) -> torch.Tensor:
        sdf_raw = (sdf / 5.0).unsqueeze(-1)
        sdf_log = torch.log1p(sdf * 10.0).unsqueeze(-1) / 2.4
        return self.net(torch.cat([sdf_raw, sdf_log], dim=-1))


class TemporalAttn(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(dim)
        self.ff   = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def _inner(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        a, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        if self.training:
            x_flat = checkpoint(self._inner, x_flat, use_reentrant=False)
        else:
            x_flat = self._inner(x_flat)
        return x_flat.reshape(B, N, T, D).permute(0, 2, 1, 3)


# Spatial Blocks

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ConvBlock3D(nn.Module):
    def __init__(self, c_in: int, c_out: int, groups: int = 8):
        super().__init__()
        g_in, g_out = min(groups, c_in), min(groups, c_out)
        self.block = nn.Sequential(
            nn.Conv3d(c_in,  c_out, 3, padding=1),
            nn.GroupNorm(g_out, c_out),
            nn.GELU(),
            nn.Conv3d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(g_out, c_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, c_in: int, c_mid: int = 64, c_out: Optional[int] = None, groups: int = 8):
        super().__init__()
        c_out = c_out or c_in
        g = groups
        self.enc1 = ConvBlock3D(c_in,       c_mid,     g)
        self.enc2 = ConvBlock3D(c_mid,      c_mid * 2, g)
        self.enc3 = ConvBlock3D(c_mid * 2,  c_mid * 4, g)
        self.dec2 = ConvBlock3D(c_mid * 6,  c_mid * 2, g)
        self.dec1 = ConvBlock3D(c_mid * 3,  c_mid,     g)
        self.out  = nn.Conv3d(c_mid, c_out, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool3d(e1, 2))
        e3 = self.enc3(F.avg_pool3d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(e3, scale_factor=2, mode="trilinear", align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="trilinear", align_corners=False), e1], dim=1))
        return self.out(d1)


# Multi-scale CDF Voxel Spatial Mixing

class VoxelLevel(nn.Module):
    """Scatter → Concat Dynamic Context → Fuse → UNet → Trilinear sample-back."""

    def __init__(self, in_dim: int, res: int, unet_mid: int, pad: float = 0.05):
        super().__init__()
        self.res = res
        self.pad = pad
        
        ctx_dim = 7  # Dynamic dX, dY, dZ, Vol, X, Y, Z
        self.fuse_in = nn.Conv3d(in_dim + ctx_dim, in_dim, kernel_size=1)
        self.unet = UNet3D(c_in=in_dim, c_mid=unet_mid, c_out=in_dim)

    def _scatter_mean(self, feats: torch.Tensor, flat_idx: torch.Tensor, R: int) -> torch.Tensor:
        B, N, D = feats.shape
        vox = feats.new_zeros(B, D, R ** 3)
        cnt = feats.new_zeros(B, 1, R ** 3)
        vox.scatter_add_(2, flat_idx.unsqueeze(1).expand(-1, D, -1), feats.transpose(1, 2))
        cnt.scatter_add_(2, flat_idx.unsqueeze(1), torch.ones(B, 1, N, device=feats.device, dtype=feats.dtype))
        return vox / cnt.clamp(min=1.0)

    def forward(self, feats: torch.Tensor, pos_comp: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        B, N, D = feats.shape
        R = self.res
        
        idx = (pos_comp * R).long().clamp(0, R - 1)
        flat = idx[..., 0] * R * R + idx[..., 1] * R + idx[..., 2]
        vox = self._scatter_mean(feats, flat, R).view(B, D, R, R, R)
        

        dx = widths[:, 0].view(B, 1, 1, 1, R)
        dy = widths[:, 1].view(B, 1, 1, R, 1)
        dz = widths[:, 2].view(B, 1, R, 1, 1)
        
        dX = dx.expand(-1, 1, R, R, R)
        dY = dy.expand(-1, 1, R, R, R)
        dZ = dz.expand(-1, 1, R, R, R)
        vol = dX * dY * dZ

        eps_log = 1e-5
        dX = torch.log(dX.clamp(min=eps_log))
        dY = torch.log(dY.clamp(min=eps_log))
        dZ = torch.log(dZ.clamp(min=eps_log))
        vol = torch.log(vol.clamp(min=eps_log))
        
        x_bnd = torch.cat([torch.zeros(B, 1, device=widths.device), widths[:, 0].cumsum(dim=1)], dim=1)
        y_bnd = torch.cat([torch.zeros(B, 1, device=widths.device), widths[:, 1].cumsum(dim=1)], dim=1)
        z_bnd = torch.cat([torch.zeros(B, 1, device=widths.device), widths[:, 2].cumsum(dim=1)], dim=1)
        
        X = ((x_bnd[:, :-1] + x_bnd[:, 1:]) / 2.0).view(B, 1, 1, 1, R).expand(-1, 1, R, R, R)
        Y = ((y_bnd[:, :-1] + y_bnd[:, 1:]) / 2.0).view(B, 1, 1, R, 1).expand(-1, 1, R, R, R)
        Z = ((z_bnd[:, :-1] + z_bnd[:, 1:]) / 2.0).view(B, 1, R, 1, 1).expand(-1, 1, R, R, R)
        
        ctx = torch.cat([dX, dY, dZ, vol, X, Y, Z], dim=1)
        vox_with_ctx = torch.cat([vox, ctx], dim=1)
        vox = self.fuse_in(vox_with_ctx)

        if self.training:
            vox = checkpoint(self.unet, vox, use_reentrant=False)
        else:
            vox = self.unet(vox)

        # 4. Sample back out using COMPUTATIONAL coordinates
        grid = (pos_comp * 2.0 - 1.0)[:, None, None, :, [2, 1, 0]]
        sampled = F.grid_sample(vox, grid, mode="bilinear", align_corners=False, padding_mode="border")
        
        return sampled.squeeze(2).squeeze(2).transpose(1, 2)


class MultiScaleVoxel(nn.Module):
    """Two-level spatial pyramid: coarse and fine with dynamic density mapping."""

    def __init__(self, in_dim: int, out_dim: int,
                 res_coarse: int = 32, res_fine: int = 64,
                 coarse_mid: int = 48, fine_mid: int = 96, pad: float = 0.05):
        super().__init__()
        self.coarse = VoxelLevel(in_dim, res=res_coarse, unet_mid=coarse_mid, pad=pad)
        self.fine   = VoxelLevel(in_dim, res=res_fine,   unet_mid=fine_mid,   pad=pad)
        self.fuse   = nn.Sequential(
            nn.LayerNorm(in_dim * 2),
            nn.Linear(in_dim * 2, out_dim),
            nn.GELU(),
        )

    def forward(self, feats: torch.Tensor, pos_comp: torch.Tensor, widths_coarse: torch.Tensor, widths_fine: torch.Tensor) -> torch.Tensor:
        c = self.coarse(feats, pos_comp, widths_coarse)
        f = self.fine(feats, pos_comp, widths_fine)
        return self.fuse(torch.cat([c, f], dim=-1))


# Core network

class CDFDoubleGridNet(nn.Module):
    """Full forward pass (no competition wrapper)."""

    def __init__(
        self,
        hidden:         int = 192,
        n_pre:          int = 2,
        n_post:         int = 4,
        res_coarse:     int = 32,
        res_fine:       int = 64,
        coarse_mid:     int = 32,
        fine_mid:       int = 64,
        temp_heads:     int = 4,
        fourier_bands:  int = FOURIER_BANDS,
        sdf_emb_dim:    int = SDF_EMB_DIM,
    ):
        super().__init__()
        pos_dim  = 3 * 2 * fourier_bands   
        in_dim   = T_IN * 3 + pos_dim + 1 + sdf_emb_dim

        self.fourier_pos = FourierPosEnc(fourier_bands)
        self.sdf_emb     = SDFEmbedding(sdf_emb_dim)

        temp_hdim = 32
        self.frame_proj = nn.Linear(3, temp_hdim)
        self.temporal   = TemporalAttn(temp_hdim, heads=temp_heads)
        
        in_dim = T_IN * temp_hdim + pos_dim + 1 + sdf_emb_dim

        self.proj_in    = nn.Linear(in_dim, hidden)
        self.pre_blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_pre)])

        self.spatial = MultiScaleVoxel(hidden, hidden,
                                       res_coarse=res_coarse,
                                       res_fine=res_fine,
                                       coarse_mid=coarse_mid,
                                       fine_mid=fine_mid)

        self.post_blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_post)])
        self.norm_out    = nn.LayerNorm(hidden)
        self.proj_out    = nn.Linear(hidden, T_OUT * 3)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
        self.register_buffer("vel_std",  torch.ones(1, 1, 1, 3))
        self.register_buffer("pos_mean", torch.zeros(1, 1, 3))
        self.register_buffer("pos_std",  torch.ones(1, 1, 3))

    @staticmethod
    def _no_slip_mask(B: int, N: int, idcs_airfoil: list[torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.ones(B, 1, N, 1, device=device, dtype=dtype)
        for b, idcs in enumerate(idcs_airfoil):
            mask[b, 0, idcs.to(device), 0] = 0.0
        return mask

    @staticmethod
    def _airfoil_mask_feat(B: int, N: int, idcs_airfoil: list[torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.zeros(B, N, 1, device=device, dtype=dtype)
        for b, idcs in enumerate(idcs_airfoil):
            mask[b, idcs.to(device), 0] = 1.0
        return mask

    def forward(
        self,
        velocity_in:   torch.Tensor,          # (B, T_IN, N, 3)
        pos:           torch.Tensor,          # (B, N, 3) physical positions
        pos_comp:      torch.Tensor,          # (B, N, 3) computational CDF positions
        idcs_airfoil:  list[torch.Tensor],
        sdf:           torch.Tensor,          # (B, N)
        widths_coarse: torch.Tensor,          # (B, 3, R_coarse)
        widths_fine:   torch.Tensor,          # (B, 3, R_fine)
    ) -> torch.Tensor:                        # (B, T_OUT, N, 3)

        B, T, N, _ = velocity_in.shape
        device, dtype = pos.device, pos.dtype

        v_norm = (velocity_in - self.vel_mean) / self.vel_std

        t_feat = self.frame_proj(v_norm)
        t_feat = self.temporal(t_feat)
        t_flat = t_feat.permute(0, 2, 1, 3).reshape(B, N, T * t_feat.shape[-1])

        pos_norm = (pos - self.pos_mean) / self.pos_std   # unit-normalized coords
        pos_enc  = self.fourier_pos(pos_norm)
        sdf_emb = self.sdf_emb(sdf)
        af_mask = self._airfoil_mask_feat(B, N, idcs_airfoil, device, dtype)

        x = torch.cat([t_flat, pos_enc, af_mask, sdf_emb], dim=-1)
        x = self.proj_in(x)

        for blk in self.pre_blocks:
            x = blk(x)

        x = x + self.spatial(x, pos_comp, widths_coarse, widths_fine)

        for blk in self.post_blocks:
            x = blk(x)

        delta_norm = self.proj_out(self.norm_out(x))
        delta_norm = delta_norm.reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)
        delta      = delta_norm * self.vel_std

        last = velocity_in[:, -1:].expand(-1, T_OUT, -1, -1)
        pred = last + delta

        no_slip = self._no_slip_mask(B, N, idcs_airfoil, device, dtype)
        return pred * no_slip


# Competition-facing entry point

class Model(nn.Module):
    """Wraps CDFDoubleGridNet with competition signature + inference-time y-flip TTA."""

    _ARCH = dict(
        hidden      = 512,
        n_pre       = 2,
        n_post      = 4,
        res_coarse  = 32,
        res_fine    = 80,
        coarse_mid  = 32,
        fine_mid    = 128,
    )

    def __init__(self, amr_sigma: float = 2.0, amr_beta: float = 0.3):
        super().__init__()
        self.net = CDFDoubleGridNet(**self._ARCH)
        
        # Inference AMR parameters
        self.amr_sigma  = amr_sigma
        self.amr_beta   = amr_beta
        self.res_coarse = self._ARCH["res_coarse"]
        self.res_fine   = self._ARCH["res_fine"]

        from huggingface_hub import hf_hub_download

        ckpt_path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if not os.path.exists(ckpt_path):
            hf_hub_download(
                repo_id="Joorx/cdf-2grid",
                filename="state_dict.pt",
                local_dir=os.path.dirname(__file__),
            )
        
        state = torch.load(ckpt_path, map_location="cpu")
        
        if "model" in state:
            state = state["model"]
            
        self.net.load_state_dict(state)

    def _forward_single(
        self,
        pos:          torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in:  torch.Tensor,
    ) -> torch.Tensor:
        
        B = pos.shape[0]
        device = pos.device
        
        # 1. Compute exact SDF on the fly
        sdf = compute_sdf_batch(pos, idcs_airfoil)
        
        # 2. Compute exact grids (looping to match dataset logic perfectly)
        pos_comp_list = []
        w_coarse_list = []
        w_fine_list   = []
        
        for b in range(B):
            pos_b = pos[b]
            
            lo = pos_b.amin(dim=0, keepdim=True)
            hi = pos_b.amax(dim=0, keepdim=True)
            pos01 = (pos_b - lo) / (hi - lo).clamp(min=1e-6)
            
            p_comp, w_dict = compute_regularized_amr_metrics(
                pos01, 
                resolutions=(self.res_coarse, self.res_fine), 
                sigma=self.amr_sigma, 
                beta=self.amr_beta
            )
            
            pos_comp_list.append(p_comp)
            w_coarse_list.append(w_dict[self.res_coarse])
            w_fine_list.append(w_dict[self.res_fine])
            
        pos_comp = torch.stack(pos_comp_list, dim=0)
        widths_coarse = torch.stack(w_coarse_list, dim=0)
        widths_fine = torch.stack(w_fine_list, dim=0)
        
        return self.net(velocity_in, pos, pos_comp, idcs_airfoil, sdf, widths_coarse, widths_fine)

    def forward(
        self,
        t:            torch.Tensor,          # (B, 10) — unused but required by spec
        pos:          torch.Tensor,          # (B, N, 3)
        idcs_airfoil: list[torch.Tensor],
        velocity_in:  torch.Tensor,          # (B, 5, N, 3)
    ) -> torch.Tensor:                       # (B, 5, N, 3)

        pred = self._forward_single(pos, idcs_airfoil, velocity_in)

        if not self.training:
            # y-flip TTA
            pos_flip    = pos.clone()
            pos_flip[..., 1] = -pos_flip[..., 1]
            vel_flip    = velocity_in.clone()
            vel_flip[..., 1] = -vel_flip[..., 1]

            # The CDF mapping inside _forward_single will naturally adapt to the flipped coords
            pred_flip   = self._forward_single(pos_flip, idcs_airfoil, vel_flip)
            pred_flip[..., 1] = -pred_flip[..., 1]

            pred = (pred + pred_flip) * 0.5

        return pred