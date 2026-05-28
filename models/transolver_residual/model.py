
import os
import torch
import torch.nn as nn

from models.transolver_residual.features        import (
    compute_features, get_feature_dim,
    precompute_distance_features, precompute_knn,
)
from models.transolver_residual.physics_attention import TransolverBlock
from models.transolver_residual.polynomial        import poly_extrapolate


class TransolverResidual(nn.Module):
    """
    GRaM submission model.

    Can be instantiated with no arguments; weights are loaded automatically
    from the same directory when a weights file is present.

    Args:
        n_layers             : number of Transolver blocks (default 8)
        hidden_dim           : token / hidden dimension (default 256)
        n_heads              : attention heads in Physics-Attention (default 8)
        slice_num            : number of physics slices M (default 32)
        mlp_ratio            : FFN expansion factor in each block (default 1)
        dropout              : dropout probability (default 0.1)
        poly_degree          : degree of polynomial extrapolation baseline (default 2)
        use_local_feats      : append mean neighbour velocity to features (default False)
        use_temporal_deltas  : append velocity differences to features (default False)
        load_weights         : auto-load weights.pt if present (default True)
    """

    def __init__(
        self,
        n_layers:            int   = 8,
        hidden_dim:          int   = 256,
        n_heads:             int   = 8,
        slice_num:           int   = 32,
        mlp_ratio:           int   = 1,
        dropout:             float = 0.1,
        poly_degree:         int   = 2,
        use_local_feats:     bool  = True,
        use_temporal_deltas: bool  = True,
        load_weights:        bool  = True,
    ):
        super().__init__()
        self.poly_degree         = poly_degree
        self.use_local_feats     = use_local_feats
        self.use_temporal_deltas = use_temporal_deltas

        feature_dim = get_feature_dim(use_local_feats, use_temporal_deltas)

        # ── Encoder: per-point MLP  D → hidden_dim ──────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
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

        # ── Geometry cache (keyed by pos fingerprint) ─────────────────────────
        # Populated lazily on the first forward call for each geometry.
        # Avoids recomputing dist/knn features for the same geometry across
        # multiple time windows or repeated calls by the evaluator.
        self._dist_cache: dict = {}   # fingerprint → (ia, dist, xsign) CPU tensors
        self._knn_cache:  dict = {}   # fingerprint → (N, k) int64 CPU tensor

        # ── Auto-load weights if present ──────────────────────────────────────
        if load_weights:
            weights_path = os.path.join(os.path.dirname(__file__), "weights.pt")
            if os.path.exists(weights_path):
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
                try:
                    self.load_state_dict(state)
                    print("[TransolverResidual] Loaded weights from weights.pt")
                except RuntimeError as e:
                    print(
                        f"[TransolverResidual] weights.pt incompatible with current "
                        f"architecture — starting from scratch.\n  ({e})"
                    )

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

    # ── Geometry cache helpers ────────────────────────────────────────────────

    @staticmethod
    def _fingerprint(pos_b: torch.Tensor) -> tuple:
        """
        Cheap, collision-resistant fingerprint for a single (N, 3) pos tensor.
        Uses shape + a few scattered values — fast and unique enough for
        distinct CFD meshes.
        """
        p = pos_b.cpu().float()
        N = p.shape[0]
        return (N,
                float(p[0].sum()),
                float(p[N // 4].sum()),
                float(p[N // 2].sum()),
                float(p[-1].sum()))

    def _ensure_geometry_cache(
        self,
        pos: torch.Tensor,         # (B, N, 3)
        idcs_airfoil: list,        # list[B]
    ):
        """
        Populate dist_cache and knn_cache for any geometry not yet seen.
        Called once at the start of forward() when the caller does not supply
        precomputed caches (i.e. at evaluation / competition inference time).
        """
        for b in range(pos.shape[0]):
            fp = self._fingerprint(pos[b])

            if fp not in self._dist_cache:
                print(f"[TransolverResidual] precomputing dist features for sample {b} ...",
                      flush=True)
                ia, dist, xsign = precompute_distance_features(
                    pos[b].cpu().float().numpy(),
                    idcs_airfoil[b].cpu().numpy().astype("int64"),
                )
                self._dist_cache[fp] = (
                    torch.from_numpy(ia),
                    torch.from_numpy(dist),
                    torch.from_numpy(xsign),
                )

            if self.use_local_feats and fp not in self._knn_cache:
                print(f"[TransolverResidual] precomputing k-NN for sample {b} ...",
                      flush=True)
                knn_idx = precompute_knn(pos[b].cpu().float().numpy())
                self._knn_cache[fp] = torch.from_numpy(knn_idx.astype("int64"))

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        t:            torch.Tensor,        # (B, 10)
        pos:          torch.Tensor,        # (B, N, 3)
        idcs_airfoil: list,                # list[B] of variable-length int tensors
        velocity_in:  torch.Tensor,        # (B, 5, N, 3)
        dist_feats:   list = None,         # list[B] of (ia, dist, xsign) — (N,) each
        knn_feats:    list = None,         # list[B] of (N, k) int tensors, or None
    ) -> torch.Tensor:                     # (B, 5, N, 3)

        B, T_in, N, _ = velocity_in.shape

        # ── 0. Geometry precomputation (inference path) ───────────────────────
        # When dist_feats or knn_feats are not supplied (e.g. competition
        # evaluator calling main.py), compute them now and cache by geometry
        # fingerprint so repeated calls for the same mesh are instant.
        if dist_feats is None or (self.use_local_feats and knn_feats is None):
            self._ensure_geometry_cache(pos, idcs_airfoil)

        if dist_feats is None:
            dist_feats = [self._dist_cache[self._fingerprint(pos[b])] for b in range(B)]

        if self.use_local_feats and knn_feats is None:
            knn_feats = [self._knn_cache[self._fingerprint(pos[b])] for b in range(B)]

        # ── 1. Polynomial baseline ────────────────────────────────────────────
        poly_pred = poly_extrapolate(
            velocity_in, t, degree=self.poly_degree
        )                                              # (B, 5, N, 3)

        # ── 2. Per-point features ─────────────────────────────────────────────
        feats = compute_features(
            pos, velocity_in, idcs_airfoil, t, self.poly_degree,
            dist_feats=dist_feats,
            use_local_feats=self.use_local_feats,
            use_temporal_deltas=self.use_temporal_deltas,
            knn_feats=knn_feats,
        )                                              # (B, N, D)

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

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
