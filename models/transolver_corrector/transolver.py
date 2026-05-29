"""Time-conditioned Transolver+ flow-map backbone."""

import math

import torch
import torch.nn as nn

from .attention import PhysicsAttention1DEidetic


class TimeEncoder(nn.Module):
    def __init__(self, n_frequencies: int, omega_max: float, n_hidden: int,
                 n_sites: int, omega_min: float = 10.0) -> None:
        super().__init__()
        self.n_sites = n_sites
        self.n_hidden = n_hidden
        omegas = 2 * math.pi * torch.logspace(
            math.log10(omega_min), math.log10(omega_max), n_frequencies,
        )
        self.register_buffer("omegas", omegas)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_frequencies, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, 2 * n_hidden * n_sites),
        )

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        dt = delta_t.unsqueeze(-1) * self.omegas
        feats = torch.cat([torch.sin(dt), torch.cos(dt)], dim=-1)
        out = self.mlp(feats)
        B, K, _ = out.shape
        return out.reshape(B * K, self.n_sites, 2, self.n_hidden)


class FiLMLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, scale: torch.Tensor,
                shift: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransolverPlusFlowBlock(nn.Module):
    def __init__(self, n_hidden: int, n_head: int, slice_num: int,
                 mlp_ratio: int = 2, dropout: float = 0.0,
                 out_dim: int | None = None, last: bool = False) -> None:
        super().__init__()
        self.last = last
        self.n_film_sites = 3 if last else 2

        self.norm1 = FiLMLayerNorm(n_hidden)
        self.attn = PhysicsAttention1DEidetic(
            n_hidden, n_head=n_head, slice_num=slice_num, dropout=dropout,
        )
        self.norm2 = FiLMLayerNorm(n_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(n_hidden, n_hidden * mlp_ratio),
            nn.GELU(),
            nn.Linear(n_hidden * mlp_ratio, n_hidden),
        )
        if self.last:
            self.norm3 = FiLMLayerNorm(n_hidden)
            self.out_proj = nn.Linear(n_hidden, out_dim)

    def forward(self, fx: torch.Tensor, film: torch.Tensor) -> torch.Tensor:
        s0, sh0 = film[:, 0, 0], film[:, 0, 1]
        s1, sh1 = film[:, 1, 0], film[:, 1, 1]
        fx = self.attn(self.norm1(fx, s0, sh0)) + fx
        fx = self.mlp(self.norm2(fx, s1, sh1)) + fx
        if self.last:
            s2, sh2 = film[:, 2, 0], film[:, 2, 1]
            return self.out_proj(self.norm3(fx, s2, sh2))
        return fx


class TransolverPlusFlowModel(nn.Module):
    """Time-conditioned Transolver+ flow-map operator."""

    N_INPUT_T = 5

    def __init__(
        self,
        n_hidden: int = 192,
        n_layers: int = 3,
        n_head: int = 8,
        slice_num: int = 32,
        mlp_ratio: int = 2,
        dropout: float = 0.21,
        use_dist_to_airfoil: bool = True,
        time_n_frequencies: int = 16,
        time_omega_min: float = 10.0,
        time_omega_max: float = 1000.0,
        input_cond_n_frequencies: int = 8,
    ) -> None:
        super().__init__()
        self.use_dist_to_airfoil = use_dist_to_airfoil

        # multi_raw_dt mode: per step vxyz (3) + scalar dt (1) = 4
        dyn_dim = self.N_INPUT_T * 4
        in_dim = dyn_dim + 4 + int(self.use_dist_to_airfoil)

        self.preprocess = nn.Sequential(
            nn.Linear(in_dim, n_hidden * 2),
            nn.GELU(),
            nn.Linear(n_hidden * 2, n_hidden),
        )
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))

        n_sites = 1 + n_layers * 2 + 1
        self.time_encoder = TimeEncoder(
            n_frequencies=time_n_frequencies, omega_min=time_omega_min,
            omega_max=time_omega_max, n_hidden=n_hidden, n_sites=n_sites,
        )
        self.preprocess_norm = FiLMLayerNorm(n_hidden)

        self.blocks = nn.ModuleList([
            TransolverPlusFlowBlock(
                n_hidden=n_hidden, n_head=n_head, slice_num=slice_num,
                mlp_ratio=mlp_ratio, dropout=dropout, out_dim=3,
                last=(i == n_layers - 1),
            )
            for i in range(n_layers)
        ])

    @staticmethod
    def _dist_to_airfoil(pos: torch.Tensor, airfoil_mask: torch.Tensor,
                         chunk_size: int = 4096) -> torch.Tensor:
        B, N, _ = pos.shape
        out = pos.new_zeros(B, N, 1)
        for b in range(B):
            af_pts = pos[b, airfoil_mask[b] > 0.5]
            min_d = pos.new_empty(N)
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                dists = torch.cdist(pos[b, start:end], af_pts)
                min_d[start:end] = dists.min(dim=-1).values
            out[b, :, 0] = torch.log1p(min_d)
        return out

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        v_in = data["velocity_in"]
        delta_t = data["delta_t"]
        delta_t_inputs = data["delta_t_inputs"]
        pos = data["pos"]
        airfoil_mask = data["airfoil_mask"]

        B, T_in, N, C = v_in.shape
        K = delta_t.shape[1]

        ctx = v_in.unsqueeze(1).expand(B, K, T_in, N, C)
        dt_features = delta_t_inputs.unsqueeze(-1)  # multi_raw_dt: (B, K, T_in, 1)
        dt_features = dt_features.unsqueeze(3).expand(B, K, T_in, N, 1)
        step_features = torch.cat([ctx, dt_features], dim=-1)
        x = step_features.permute(0, 1, 3, 2, 4).reshape(B * K, N, -1)

        pos_rep = pos.unsqueeze(1).expand(B, K, N, 3).reshape(B * K, N, 3)
        mask_rep = airfoil_mask.unsqueeze(1).expand(B, K, N).reshape(B * K, N, 1)

        if self.use_dist_to_airfoil:
            dist_to_af = self._dist_to_airfoil(pos, airfoil_mask)
            dist_rep = dist_to_af.unsqueeze(1).expand(B, K, N, 1).reshape(B * K, N, 1)
            inp = torch.cat([x, pos_rep, mask_rep, dist_rep], dim=-1)
        else:
            inp = torch.cat([x, pos_rep, mask_rep], dim=-1)

        fx = self.preprocess(inp) + self.placeholder[None, None, :]
        film = self.time_encoder(delta_t)

        s, sh = film[:, 0, 0], film[:, 0, 1]
        fx = self.preprocess_norm(fx, s, sh)

        site = 1
        for block in self.blocks:
            n_block_sites = block.n_film_sites
            fx = block(fx, film[:, site:site + n_block_sites])
            site += n_block_sites

        delta = fx.reshape(B, K, N, C)
        v_seed = v_in[:, -1].unsqueeze(1).expand(B, K, N, C)
        out = v_seed + delta
        out = out * (1 - airfoil_mask[:, None, :, None])
        return out
