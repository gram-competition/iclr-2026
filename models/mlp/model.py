from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

try:
    from torch_cluster import knn_graph as torch_cluster_knn_graph

    HAS_TORCH_CLUSTER = True
except ImportError:
    torch_cluster_knn_graph = None
    HAS_TORCH_CLUSTER = False


def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    num_hidden_layers: int = 2,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(num_hidden_layers - 1):
        layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.GELU()))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class GlobalDimensionShrinkageAttention(nn.Module):
    """Project point tokens to a low-rank mode space, attend globally, then lift back."""

    def __init__(self, latent_dim: int, num_modes: int, num_heads: int):
        super().__init__()
        self.down_projector = _make_mlp(latent_dim, latent_dim, num_modes)
        self.mode_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.up_projector = _make_mlp(latent_dim, latent_dim, num_modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_scores = torch.softmax(self.down_projector(x), dim=1)
        modes = torch.einsum("bnm,bnd->bmd", down_scores, x)

        attended_modes, _ = self.mode_attention(
            modes,
            modes,
            modes,
            need_weights=False,
        )

        up_scores = torch.softmax(self.up_projector(x), dim=-1)
        return torch.einsum("bnm,bmd->bnd", up_scores, attended_modes)


class LocalGraphAttention(nn.Module):
    """Attention over Euclidean kNN graph for unstructured point clouds."""

    def __init__(self, latent_dim: int, k: int, query_chunk_size: int = 2048):
        super().__init__()
        self.k = k
        self.query_chunk_size = query_chunk_size
        self.pos_encoding = _make_mlp(3, latent_dim, latent_dim)
        self.attn_kernel = _make_mlp(latent_dim, latent_dim, latent_dim)
        self.value_mlp = _make_mlp(latent_dim, latent_dim, latent_dim)

    @staticmethod
    def _batched_gather(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch_size, num_pos, feat_dim = features.shape
        _, num_query, k = indices.shape

        offsets = (
            torch.arange(batch_size, device=features.device, dtype=torch.long)
            .view(batch_size, 1, 1)
            .mul(num_pos)
        )
        flat_indices = (indices + offsets).reshape(-1)
        flat_features = features.reshape(batch_size * num_pos, feat_dim)
        gathered = flat_features.index_select(0, flat_indices)
        return gathered.view(batch_size, num_query, k, feat_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        knn_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_pos, _ = x.shape
        out = torch.empty_like(x)

        for start in range(0, num_pos, self.query_chunk_size):
            end = min(start + self.query_chunk_size, num_pos)
            idx_chunk = knn_indices[:, start:end, :]

            center_x = x[:, start:end, :].unsqueeze(2).expand(-1, -1, self.k, -1)
            center_pos = pos[:, start:end, :].unsqueeze(2)
            neigh_x = self._batched_gather(x, idx_chunk)
            neigh_pos = self._batched_gather(pos, idx_chunk)

            rel_pos = neigh_pos - center_pos
            delta = self.pos_encoding(rel_pos)

            attn_input = center_x - neigh_x + delta
            attn_weights = torch.softmax(self.attn_kernel(attn_input), dim=2)

            values = self.value_mlp(neigh_x) + delta
            out[:, start:end, :] = (attn_weights * values).sum(dim=2)

        return out


class MicroPointWiseAttention(nn.Module):
    """Point-wise token reweighting using MLP scores normalized across points."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.score_mlp = _make_mlp(latent_dim, latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize across sequence length N to match Micro attention formulation.
        score_p = torch.softmax(self.score_mlp(x), dim=1)
        return x + score_p * x


class MNOBlock(nn.Module):
    """Single Multiscale Neural Operator block with parallel branches and fusion."""

    def __init__(
        self,
        latent_dim: int,
        num_modes: int,
        num_heads: int,
        k: int,
        graph_query_chunk_size: int,
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(latent_dim)

        self.global_attention = GlobalDimensionShrinkageAttention(
            latent_dim=latent_dim,
            num_modes=num_modes,
            num_heads=num_heads,
        )
        self.local_attention = LocalGraphAttention(
            latent_dim=latent_dim,
            k=k,
            query_chunk_size=graph_query_chunk_size,
        )
        self.micro_attention = MicroPointWiseAttention(latent_dim=latent_dim)

        self.fusion_mlp = _make_mlp(latent_dim, 2 * latent_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        knn_indices: torch.Tensor,
    ) -> torch.Tensor:
        x_norm = self.pre_norm(x)

        x_global = self.global_attention(x_norm)
        x_local = self.local_attention(x_norm, pos, knn_indices)
        x_micro = self.micro_attention(x_norm)

        fused = x_global + x_local + x_micro
        return x + self.fusion_mlp(fused)


class FourierTimeEmbedding(nn.Module):
    """Map scalar timestamps to high-dimensional sinusoidal features.

    For each of ``num_freqs`` learnable frequencies, produces ``[sin(ω·t), cos(ω·t)]``,
    giving a ``2 * num_freqs``-dimensional embedding per scalar input.
    The frequencies are initialised log-linearly and kept trainable so the
    model can adapt the spectral coverage during training.
    """

    def __init__(self, num_freqs: int = 16):
        super().__init__()
        self.num_freqs = num_freqs
        self.out_dim = 2 * num_freqs
        # Log-linear init spanning a wide range of temporal scales.
        init_freqs = torch.linspace(0.0, 4.0, num_freqs).exp()  # ~1 … ~55
        self.freqs = nn.Parameter(init_freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``t``: arbitrary shape ``(...)``.  Returns ``(..., 2*num_freqs)``."""
        # (..., 1) * (num_freqs,) -> (..., num_freqs)
        angles = t.unsqueeze(-1) * self.freqs
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class MLP(nn.Module):
    """Encoder-MNO-Decoder baseline for CFD on unstructured 3D point clouds."""

    _DEFAULT_WEIGHTS_NAME = "best.pt"
    _DEFAULT_SUBMISSION_CHECKPOINT_DIR = "baseline_mno_seed118_base_2_32k_budget_crop_5580430"

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        num_modes: int = 256,
        num_heads: int = 8,
        num_blocks: int = 4,
        k: int = 16,
        num_t_in: int = 5,
        num_t_out: int = 5,
        output_channels: int = 3,
        knn_query_chunk_size: int = 1024,
        graph_query_chunk_size: int = 2048,
        use_torch_cluster_knn: bool = True,
        load_pretrained: bool = True,
        weights_path: str | Path | None = None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.k = k
        self.num_t_in = num_t_in
        self.num_t_out = num_t_out
        self.output_channels = output_channels
        self.knn_query_chunk_size = knn_query_chunk_size
        self.use_torch_cluster_knn = use_torch_cluster_knn and HAS_TORCH_CLUSTER

        # Fourier time embedding: each of the (num_t_in + num_t_out) timestamps
        # is mapped to 2*num_time_freqs sinusoidal features, replacing raw scalars.
        num_time_freqs = 16
        self.time_embedding = FourierTimeEmbedding(num_freqs=num_time_freqs)
        time_feat_dim = (num_t_in + num_t_out) * self.time_embedding.out_dim

        # Per-output-step horizon embedding: tells the decoder *which* future
        # step it is reconstructing so each output frame is time-aware.
        self.horizon_mlp = _make_mlp(self.time_embedding.out_dim, latent_dim, latent_dim, num_hidden_layers=1)

        # +1 for binary airfoil mask, +1 for continuous wall distance (SDF),
        # +9 for local surface coordinate frame (normal + 2 tangent vectors).
        aux_dim = (num_t_in * 3) + time_feat_dim + 1 + 1 + 9
        self.encoder = _make_mlp(3 + aux_dim, 2 * latent_dim, latent_dim)

        self.blocks = nn.ModuleList(
            [
                MNOBlock(
                    latent_dim=latent_dim,
                    num_modes=num_modes,
                    num_heads=num_heads,
                    k=k,
                    graph_query_chunk_size=graph_query_chunk_size,
                )
                for _ in range(num_blocks)
            ]
        )

        self.decoder = _make_mlp(
            latent_dim,
            2 * latent_dim,
            num_t_out * output_channels,
        )

        # Initialize residual head to zero so the model starts exactly at the
        # persistence baseline: pred = last_input + 0.
        final_layer = self.decoder[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

        if load_pretrained:
            self._load_pretrained_weights(weights_path)

    @classmethod
    def _resolve_weights_path(cls, weights_path: str | Path | None) -> Path:
        if weights_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            candidate_paths = (
                repo_root
                / cls._DEFAULT_SUBMISSION_CHECKPOINT_DIR
                / "checkpoints"
                / cls._DEFAULT_WEIGHTS_NAME,
                repo_root / "best_weights" / cls._DEFAULT_WEIGHTS_NAME,
                Path(__file__).resolve().with_name("state_dict.pt"),
            )
            for candidate_path in candidate_paths:
                if candidate_path.exists():
                    return candidate_path

            searched_paths = ", ".join(str(path) for path in candidate_paths)
            raise FileNotFoundError(
                "Bundled checkpoint not found in any default location: "
                f"{searched_paths}"
            )
        return Path(weights_path).expanduser().resolve()

    def _load_pretrained_weights(self, weights_path: str | Path | None) -> None:
        checkpoint_path = self._resolve_weights_path(weights_path)
        self.loaded_weights_path = checkpoint_path

        import numpy as np
        import numpy.dtypes as _np_dtypes
        _numpy_safe_globals = [
            np._core.multiarray._reconstruct,
            np._core.multiarray.scalar,
            np.ndarray,
            np.dtype,
        ]
        # Allowlist all concrete numpy dtype types (e.g. UInt32DType, Float64DType, …)
        _numpy_safe_globals += [
            getattr(_np_dtypes, _n)
            for _n in dir(_np_dtypes)
            if isinstance(getattr(_np_dtypes, _n), type)
        ]
        with torch.serialization.safe_globals(_numpy_safe_globals):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict)

    @staticmethod
    def _build_airfoil_mask(
        idcs_airfoil: list[torch.Tensor],
        batch_size: int,
        num_pos: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = torch.zeros((batch_size, num_pos, 1), device=device, dtype=dtype)
        for b, indices in enumerate(idcs_airfoil):
            if indices.numel() == 0:
                continue
            mask[b, indices.to(device=device, dtype=torch.long), 0] = 1.0
        return mask

    @staticmethod
    def _compute_wall_distance(
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute min Euclidean distance from each point to the nearest airfoil point.

        Returns a ``(B, N, 1)`` tensor of wall distances, log-transformed for
        better gradient behaviour: ``log(1 + d)``.
        """
        batch_size, num_pos, _ = pos.shape
        wall_dist = torch.zeros(
            (batch_size, num_pos, 1), device=pos.device, dtype=pos.dtype
        )
        for b, indices in enumerate(idcs_airfoil):
            if indices.numel() == 0:
                continue
            airfoil_pos = pos[b, indices.to(device=pos.device, dtype=torch.long), :]  # (A, 3)
            # cdist: (N, A) -> min over airfoil points
            dists = torch.cdist(pos[b].unsqueeze(0), airfoil_pos.unsqueeze(0)).squeeze(0)  # (N, A)
            min_dist = dists.min(dim=-1).values  # (N,)
            wall_dist[b, :, 0] = torch.log1p(min_dist)
        return wall_dist

    @staticmethod
    def _compute_surface_frame(
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute local surface coordinate frame on the fly (inference path).

        Returns a ``(B, N, 9)`` tensor: ``[n, t1, t2]`` per point.
        """
        batch_size, num_pos, _ = pos.shape
        frame = torch.zeros(
            (batch_size, num_pos, 9), device=pos.device, dtype=pos.dtype
        )
        for b, indices in enumerate(idcs_airfoil):
            if indices.numel() == 0:
                frame[b, :, 2] = 1.0   # default normal = z
                frame[b, :, 3] = 1.0   # default t1 = x
                frame[b, :, 7] = 1.0   # default t2 = y
                continue
            airfoil_pos = pos[b, indices.to(device=pos.device, dtype=torch.long), :]  # (A, 3)
            dists = torch.cdist(pos[b].unsqueeze(0), airfoil_pos.unsqueeze(0)).squeeze(0)  # (N, A)
            min_dist, min_idx = dists.min(dim=-1)
            nearest = airfoil_pos[min_idx]  # (N, 3)
            diff = pos[b] - nearest
            dist_clamped = min_dist.unsqueeze(-1).clamp(min=1e-12)
            normals = diff / dist_clamped

            # On-surface points: default normal to z-up
            on_surface = min_dist < 1e-8
            if on_surface.any():
                normals[on_surface] = torch.tensor(
                    [0.0, 0.0, 1.0], device=pos.device, dtype=pos.dtype
                )

            normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-12)

            ref = torch.zeros_like(normals)
            ref[:, 0] = 1.0
            parallel_mask = normals[:, 0].abs() > 0.9
            ref[parallel_mask, 0] = 0.0
            ref[parallel_mask, 1] = 1.0

            t1 = torch.cross(normals, ref, dim=-1)
            t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            t2 = torch.cross(normals, t1, dim=-1)
            t2 = t2 / t2.norm(dim=-1, keepdim=True).clamp(min=1e-12)

            frame[b, :, 0:3] = normals
            frame[b, :, 3:6] = t1
            frame[b, :, 6:9] = t2
        return frame

    @staticmethod
    def _pad_neighbors(indices: torch.Tensor, target_k: int) -> torch.Tensor:
        current_k = indices.size(-1)
        if current_k == target_k:
            return indices
        pad = indices[..., -1:].expand(-1, -1, target_k - current_k)
        return torch.cat((indices, pad), dim=-1)

    def _knn_cdist(self, pos: torch.Tensor) -> torch.Tensor:
        batch_size, num_pos, _ = pos.shape
        k_eff = min(self.k, max(1, num_pos - 1))
        all_indices = torch.empty(
            (batch_size, num_pos, k_eff),
            dtype=torch.long,
            device=pos.device,
        )

        for b in range(batch_size):
            points = pos[b]
            for start in range(0, num_pos, self.knn_query_chunk_size):
                end = min(start + self.knn_query_chunk_size, num_pos)
                dist = torch.cdist(points[start:end], points)
                row_indices = torch.arange(end - start, device=pos.device)
                col_indices = torch.arange(start, end, device=pos.device)
                dist[row_indices, col_indices] = float("inf")
                all_indices[b, start:end, :] = torch.topk(
                    dist,
                    k=k_eff,
                    dim=1,
                    largest=False,
                ).indices

        return self._pad_neighbors(all_indices, self.k)

    @torch._dynamo.disable
    def _knn_torch_cluster(self, pos: torch.Tensor) -> torch.Tensor:
        batch_size, num_pos, _ = pos.shape
        k_eff = min(self.k, max(1, num_pos - 1))
        all_indices = torch.empty(
            (batch_size, num_pos, k_eff),
            dtype=torch.long,
            device=pos.device,
        )

        for b in range(batch_size):
            edge_index = torch_cluster_knn_graph(
                pos[b],
                k=k_eff,
                loop=False,
            )
            src, dst = edge_index[0], edge_index[1]
            order = torch.argsort(dst)
            all_indices[b] = src[order].view(num_pos, k_eff)

        return self._pad_neighbors(all_indices, self.k)

    def _build_knn_graph(self, pos: torch.Tensor) -> torch.Tensor:
        if self.use_torch_cluster_knn:
            return self._knn_torch_cluster(pos)
        return self._knn_cdist(pos)

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
        velocity_mean: torch.Tensor | None = None,
        velocity_std: torch.Tensor | None = None,
        return_knn_indices: bool = False,
        wall_distance: torch.Tensor | None = None,
        surface_frame: torch.Tensor | None = None,
        knn_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_t_in, num_pos, _ = velocity_in.shape
        expected_t = self.num_t_in + self.num_t_out

        if num_t_in != self.num_t_in:
            raise ValueError(f"Expected num_t_in={self.num_t_in}, received {num_t_in}.")
        if t.shape != (batch_size, expected_t):
            raise ValueError(
                f"Expected t.shape={(batch_size, expected_t)}, got {tuple(t.shape)}."
            )
        if len(idcs_airfoil) != batch_size:
            raise ValueError(
                "idcs_airfoil must contain one index tensor per batch element."
            )

        # Auto-normalize: when called via the submission signature (no mean/std),
        # compute per-sample stats from velocity_in — identical to dataset.py —
        # and unscale the output before returning so callers get physical velocities.
        _auto_normalized = False
        if velocity_mean is None or velocity_std is None:
            with torch.no_grad():
                # Mean/std over time and spatial dims: (B,1,1,3) matching dataset.py
                velocity_mean_local = velocity_in.mean(dim=(1, 2), keepdim=True)
                velocity_std_local = velocity_in.std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(1e-6)
            velocity_in = (velocity_in - velocity_mean_local) / velocity_std_local
            velocity_mean = velocity_mean_local.squeeze(1).squeeze(1)  # (B, 3)
            velocity_std = velocity_std_local.squeeze(1).squeeze(1)    # (B, 3)
            _auto_normalized = True

        velocity_feat = velocity_in.permute(0, 2, 1, 3).reshape(
            batch_size,
            num_pos,
            num_t_in * 3,
        )
        # Fourier-embed each timestamp and flatten across the time axis.
        # t: (B, T_total) -> (B, T_total, 2*num_freqs) -> (B, T_total * 2*num_freqs)
        time_emb = self.time_embedding(t)  # (B, T_total, 2F)
        time_feat = time_emb.reshape(batch_size, -1)  # (B, T_total * 2F)
        time_feat = time_feat.unsqueeze(1).expand(-1, num_pos, -1)  # (B, N, T_total * 2F)
        airfoil_mask = self._build_airfoil_mask(
            idcs_airfoil,
            batch_size,
            num_pos,
            device=pos.device,
            dtype=pos.dtype,
        )

        with torch.no_grad():
            if wall_distance is not None:
                # Use precomputed raw wall distance (from dataset).
                # wall_distance shape: (B, N) -> apply log1p and unsqueeze.
                wall_distance_feat = torch.log1p(wall_distance).unsqueeze(-1)
            else:
                # Inference on unseen data: compute on the fly.
                wall_distance_feat = self._compute_wall_distance(pos, idcs_airfoil)

            if surface_frame is not None:
                # Precomputed from dataset: (B, N, 9)
                surface_frame_feat = surface_frame
            else:
                # Inference on unseen data: compute on the fly.
                surface_frame_feat = self._compute_surface_frame(pos, idcs_airfoil)

        encoder_input = torch.cat((pos, velocity_feat, time_feat, airfoil_mask, wall_distance_feat, surface_frame_feat), dim=-1)
        x = self.encoder(encoder_input)

        if knn_indices is None:
            with torch.no_grad():
                knn_indices = self._build_knn_graph(pos)

        for block in self.blocks:
            x = block(x, pos, knn_indices)

        decoded = self.decoder(x)
        residual = decoded.view(
            batch_size,
            num_pos,
            self.num_t_out,
            self.output_channels,
        ).permute(0, 2, 1, 3)  # (B, T_out, N, C)

        # Per-output-step horizon conditioning: embed each output timestamp and
        # produce a multiplicative gate so each future frame is time-aware.
        # time_emb: (B, T_total, 2F) — extract the output portion.
        output_time_emb = time_emb[:, self.num_t_in:, :]  # (B, T_out, 2F)
        horizon_gate = self.horizon_mlp(output_time_emb)  # (B, T_out, latent_dim)
        # Reduce latent_dim -> output_channels via a simple learned projection
        # reusing decoder weights is wasteful; instead use a sigmoid gate on the
        # residual, broadcast across spatial points.
        horizon_scale = torch.sigmoid(horizon_gate.mean(dim=-1, keepdim=True))  # (B, T_out, 1)
        residual = residual * horizon_scale.unsqueeze(2)  # (B, T_out, N, C)

        # Predict residual dynamics around a persistence baseline from the last input frame.
        last_input_frame = velocity_in[:, -1:, :, :]
        baseline = last_input_frame.expand(-1, self.num_t_out, -1, -1)
        velocity_out = baseline + residual

        # Hard no-slip on the final prediction in the correct scaled space.
        airfoil_mask_bool = airfoil_mask.bool().unsqueeze(1).expand(
            -1,
            self.num_t_out,
            -1,
            self.output_channels,
        )
        if velocity_mean is not None and velocity_std is not None:
            scaled_zero = ((0.0 - velocity_mean) / velocity_std).view(
                batch_size,
                1,
                1,
                self.output_channels,
            )
            scaled_zero = scaled_zero.to(device=velocity_out.device, dtype=velocity_out.dtype)
            velocity_out = torch.where(airfoil_mask_bool, scaled_zero, velocity_out)
        else:
            velocity_out = velocity_out * (~airfoil_mask_bool).to(dtype=velocity_out.dtype)

        output = velocity_out.contiguous()

        # Unscale back to physical velocity space when auto-normalization was applied.
        if _auto_normalized:
            vm = velocity_mean.view(batch_size, 1, 1, self.output_channels).to(dtype=output.dtype)
            vs = velocity_std.view(batch_size, 1, 1, self.output_channels).to(dtype=output.dtype)
            output = output * vs + vm

        if return_knn_indices:
            return output, knn_indices
        return output
