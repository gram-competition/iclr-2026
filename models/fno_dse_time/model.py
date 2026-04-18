
"""
FNO3d_dse_v2 — enhanced version of FNO3d_dse with two targeted improvements:

  1. Sinusoidal time embedding
       The actual time values t are encoded as sin/cos embeddings and added
       to every point's feature vector. The model now has an explicit clock
       rather than treating all timesteps as anonymous channels.

  2. Velocity residual skip connection
       The last input timestep of velocity_in is used as a direct skip to the
       output. The FNO learns to predict the *residual* on top of this prior.
       Low-frequency laminar flow is essentially free; all model capacity is
       focused on the high-frequency turbulent residual.

Everything else (VFT3d, SpectralConv3d_dse, training configs) is unchanged.
Same forward() signature as FNO3d_dse.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal time embedding
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 16):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    def forward(self, t):
        B, T = t.shape
        half = self.embed_dim // 2
        device = t.device
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=device) / (half - 1))
        args = t.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)

# ─────────────────────────────────────────────────────────────────────────────
# VFT3d  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class VFT3d:
    def __init__(self, x_pos, y_pos, z_pos, modes):
        B, N = x_pos.shape
        self.modes = modes
        self.B = B
        self.N = N
        dev = x_pos.device
        def scale(p):
            p = p - p.min(dim=1, keepdim=True).values
            mx = p.max(dim=1, keepdim=True).values
            mx = torch.where(mx < 1e-8, torch.ones_like(mx), mx)
            return p * (2 * np.pi) / mx
        self.xp = scale(x_pos)
        self.yp = scale(y_pos)
        self.zp = scale(z_pos)
        def freq(m):
            return torch.cat([
                torch.arange(m, device=dev),
                torch.arange(-m, 0, device=dev)
            ]).float()
        self.Kx = freq(modes)
        self.Ky = freq(modes)
        self.Kz = freq(modes)
        self.V_fwd, self.V_inv = self._make_matrix()
    def _make_matrix(self):
        B, N = self.B, self.N
        M  = self.modes * 2
        M3 = M ** 3
        Kx = self.Kx[None, :, None, None, None]
        Ky = self.Ky[None, None, :, None, None]
        Kz = self.Kz[None, None, None, :, None]
        xp = self.xp[:, None, None, None, :]
        yp = self.yp[:, None, None, None, :]
        zp = self.zp[:, None, None, None, :]
        phase = (Kx * xp + Ky * yp + Kz * zp).reshape(B, M3, N)
        V_fwd = torch.exp(-1j * phase) / np.sqrt(N)
        V_inv = torch.conj(V_fwd).permute(0, 2, 1)
        return V_fwd, V_inv
    def forward(self, data):
        return torch.bmm(self.V_fwd, data)
    def inverse(self, data):
        return torch.bmm(self.V_inv, data)

# ─────────────────────────────────────────────────────────────────────────────
# SpectralConv3d_dse  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class SpectralConv3d_dse(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_ch  = in_channels
        self.out_ch = out_channels
        self.modes  = modes
        M = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.rand(
                in_channels, out_channels, M, M, M, dtype=torch.cfloat
            ))
            for _ in range(8)
        ])
    def compl_mul3d(self, x, w):
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)
    def forward(self, x, transformer):
        B, C, N = x.shape
        M  = self.modes
        M2 = M * 2
        x_t  = x.permute(0, 2, 1).cfloat()
        x_ft = transformer.forward(x_t)
        x_ft = x_ft.permute(0, 2, 1).reshape(B, C, M2, M2, M2)
        out_ft = torch.zeros(B, self.out_ch, M2, M2, M2,
                             dtype=torch.cfloat, device=x.device)
        octants = [
            (slice(None, M),  slice(None, M),  slice(None, M)),
            (slice(None, M),  slice(None, M),  slice(-M, None)),
            (slice(None, M),  slice(-M, None), slice(None, M)),
            (slice(None, M),  slice(-M, None), slice(-M, None)),
            (slice(-M, None), slice(None, M),  slice(None, M)),
            (slice(-M, None), slice(None, M),  slice(-M, None)),
            (slice(-M, None), slice(-M, None), slice(None, M)),
            (slice(-M, None), slice(-M, None), slice(-M, None)),
        ]
        for (sx, sy, sz), w in zip(octants, self.weights):
            out_ft[:, :, sx, sy, sz] = self.compl_mul3d(
                x_ft[:, :, sx, sy, sz], w
            )
        out_flat = out_ft.reshape(B, self.out_ch, M2 ** 3).permute(0, 2, 1)
        x_out    = transformer.inverse(out_flat).permute(0, 2, 1)
        return x_out.real

# ─────────────────────────────────────────────────────────────────────────────
# FNO3d_dse_v2  — main model
# ─────────────────────────────────────────────────────────────────────────────
class FNO3d_dse_v2(nn.Module):
    configs = {
        'in_channels':     19,
        'out_channels':    15,
        'modes':           10,
        'width':           32,
        'n_layers':        4,
        'time_embed_dim':  16,
        'batch_size':      1,
        'epochs':          200,
        'learning_rate':   1e-3,
        'scheduler_step':  10,
        'scheduler_gamma': 0.97,
        'weight_decay':    1e-4,
    }
    def __init__(self, configs: dict):
        super().__init__()
        self.modes    = configs['modes']
        self.width    = configs['width']
        self.n_layers = configs.get('n_layers', 4)
        te_dim        = configs.get('time_embed_dim', 16)
        out_ch = configs['out_channels']
        W      = self.width
        M      = self.modes
        self.time_embed = SinusoidalTimeEmbedding(embed_dim=te_dim)
        in_ch_lifted = configs['in_channels'] + te_dim
        self.fc0 = nn.Linear(in_ch_lifted, W)
        self.convs = nn.ModuleList([
            SpectralConv3d_dse(W, W, M) for _ in range(self.n_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv1d(W, W, 1) for _ in range(self.n_layers)
        ])
        self.fc1 = nn.Linear(W, 128)
        self.fc2 = nn.Linear(128, out_ch)
        self.residual_proj = nn.Linear(3, out_ch)
    def _build_airfoil_mask(self, idcs_airfoil, B, N, device):
        mask = torch.zeros(B, N, 1, device=device)
        if isinstance(idcs_airfoil, (list, tuple)):
            for b, idx in enumerate(idcs_airfoil):
                if idx is None:
                    continue
                mask[b, idx.to(device), 0] = 1.0
        elif torch.is_tensor(idcs_airfoil):
            if idcs_airfoil.dim() == 2:
                for b in range(min(B, idcs_airfoil.shape[0])):
                    mask[b, idcs_airfoil[b].to(device), 0] = 1.0
            elif idcs_airfoil.dim() == 1:
                mask[:, idcs_airfoil.to(device), 0] = 1.0
        return mask
    def _build_time_features(self, t, N):
        t_in   = t[:, :5]
        t_emb  = self.time_embed(t_in)
        t_mean = t_emb.mean(dim=1, keepdim=True)
        return t_mean.expand(-1, N, -1)
    def _forward_features(self, x, pos):
        B, N, _ = x.shape
        transform = VFT3d(
            pos[:, :, 0], pos[:, :, 1], pos[:, :, 2],
            self.modes
        )
        h = self.fc0(x).permute(0, 2, 1)
        for conv, w in zip(self.convs, self.ws):
            h = F.gelu(conv(h, transform) + w(h))
        h = h.permute(0, 2, 1)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return h
    def forward(self, t, pos, idcs_airfoil, velocity_in):
        B, N, _ = pos.shape
        device  = pos.device
        vel_flat     = velocity_in.permute(0, 2, 1, 3).reshape(B, N, 15)
        airfoil_mask = self._build_airfoil_mask(idcs_airfoil, B, N, device)
        pos_norm     = pos - pos.amin(dim=1, keepdim=True)
        pos_norm     = pos_norm / pos_norm.amax(dim=1, keepdim=True).clamp(1e-8)
        t_feats = self._build_time_features(t, N)
        x = torch.cat([pos_norm, vel_flat, airfoil_mask, t_feats], dim=-1)
        residual_delta = self._forward_features(x, pos)
        vel_last  = velocity_in[:, -1, :, :]
        vel_prior = self.residual_proj(vel_last)
        out = vel_prior + residual_delta
        return out.reshape(B, N, 5, 3).permute(0, 2, 1, 3)

# Competition model wrapper: loads weights and exposes required interface
class FNO_DSE_TIME(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FNO3d_dse_v2(FNO3d_dse_v2.configs)
        checkpoint_path = 'models/fno_dse_time/best.pt'
        state = torch.load(os.path.abspath(checkpoint_path), map_location='cpu', weights_only=True)
        if 'model' in state:
            state = state['model']
        # If keys are still prefixed with 'model.', strip it
        if any(k.startswith('model.') for k in state.keys()):
            state = {k.replace('model.', '', 1): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()
    def forward(self, t, pos, idcs_airfoil, velocity_in):
        with torch.no_grad():
            return self.model(t, pos, idcs_airfoil, velocity_in)
