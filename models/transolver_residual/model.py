"""
TransolverResidual — full model for GRaM transient airflow prediction.

Forward pass:
    1. Polynomial extrapolation (no learned params) → baseline prediction
    2. Per-point feature computation (~52 channels)
    3. Per-point MLP encoder: 52 → hidden_dim
    4. L × Transolver blocks (Physics-Attention on irregular 3-D mesh)
    5. Per-point MLP decoder: hidden_dim → 15  (5 timesteps × 3 components)
    6. Output = poly_baseline + learned_correction
    7. Hard zero on airfoil surface (no-slip enforcement)

The model only learns the turbulent *correction* on top of the polynomial
baseline. This means:
  - it degrades gracefully (poly baseline is returned even if correction≈0)
  - the learning target is zero-mean  ← easier to optimise
  - capacity is focused on the wake, where prediction is hardest
"""

import os
import torch
import torch.nn as nn

from models.transolver_residual.features        import compute_features, FEATURE_DIM
from models.transolver_residual.physics_attention import TransolverBlock
from models.transolver_residual.polynomial      import poly_extrapolate


class TransolverResidual(nn.Module):
    """
    GRaM submission model.

    Can be instantiated with no arguments; weights are loaded automatically
    from the same directory when a weights file is present.

    Args:
        n_layers    : number of Transolver blocks (default 8)
        hidden_dim  : token / hidden dimension (default 256)
        n_heads     : attention heads in Physics-Attention (default 8)
        slice_num   : number of physics slices M (default 32)
        mlp_ratio   : FFN expansion factor in each block (default 1)
        dropout     : dropout probability (default 0.1)
        poly_degree : degree of polynomial extrapolation baseline (default 2)
    """

    def __init__(
        self,
        n_layers:     int   = 8,
        hidden_dim:   int   = 256,
        n_heads:      int   = 8,
        slice_num:    int   = 32,
        mlp_ratio:    int   = 1,
        dropout:      float = 0.1,
        poly_degree:  int   = 2,
        load_weights: bool  = True,
    ):
        super().__init__()
        self.poly_degree = poly_degree

        # ── Encoder: per-point MLP  52 → hidden_dim ─────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # ── Transolver backbone ──────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransolverBlock(
                dim=hidden_dim,
                heads=n_heads,
                slice_num=slice_num,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # ── Decoder: per-point linear  hidden_dim → 15 ──────────────────────
        self.norm    = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 15)   # 5 × 3

        # ── Weight initialisation ─────────────────────────────────────────────
        self._init_weights()

        # ── Auto-load weights if present ──────────────────────────────────────
        if load_weights:
            weights_path = os.path.join(os.path.dirname(__file__), "weights.pt")
            if os.path.exists(weights_path):
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
                try:
                    self.load_state_dict(state)
                    print("[TransolverResidual] Loaded weights from weights.pt")
                except RuntimeError as e:
                    print(f"[TransolverResidual] weights.pt is incompatible with current architecture — starting from scratch.\n  ({e})")

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init decoder so the model starts as a pure polynomial baseline
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        t:            torch.Tensor,        # (B, 10)
        pos:          torch.Tensor,        # (B, N, 3)
        idcs_airfoil: list,                # list[B] of variable-length int tensors
        velocity_in:  torch.Tensor,        # (B, 5, N, 3)
        dist_feats:   list = None,         # list[B] of precomputed (ia, dist, xsign) — (N,) each
    ) -> torch.Tensor:                     # (B, 5, N, 3)

        B, T_in, N, _ = velocity_in.shape

        # ── 1. Polynomial baseline ────────────────────────────────────────────
        poly_pred = poly_extrapolate(
            velocity_in, t, degree=self.poly_degree
        )                                              # (B, 5, N, 3)

        # ── 2. Per-point features ─────────────────────────────────────────────
        feats = compute_features(
            pos, velocity_in, idcs_airfoil, t, self.poly_degree,
            dist_feats=dist_feats,
        )                                              # (B, N, 52)

        # ── 3. MLP encoder ────────────────────────────────────────────────────
        x = self.encoder(feats)                        # (B, N, hidden_dim)

        # ── 4. Transolver blocks ──────────────────────────────────────────────
        for block in self.blocks:
            x = block(x)                               # (B, N, hidden_dim)

        # ── 5. Decode correction ──────────────────────────────────────────────
        correction = self.decoder(self.norm(x))        # (B, N, 15)
        correction = correction.reshape(B, N, 5, 3) \
                                .permute(0, 2, 1, 3)  # (B, 5, N, 3)

        # ── 6. Residual combination ───────────────────────────────────────────
        out = poly_pred + correction                   # (B, 5, N, 3)

        # ── 7. No-slip enforcement ────────────────────────────────────────────
        for b in range(B):
            out[b, :, idcs_airfoil[b]] = 0.0

        return out

    # ── Convenience ───────────────────────────────────────────────────────────

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
