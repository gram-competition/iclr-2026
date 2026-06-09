from __future__ import annotations

import os

import torch
import torch.nn as nn

from .components import (
    HAS_TORCH_CLUSTER,
    FourierTimeEmbedding,
    MNOBlock,
    _make_mlp,
    build_airfoil_mask,
    compute_surface_frame,
    compute_wall_distance,
    torch_cluster_knn_graph,
)


class SpatiotemporalMNO(nn.Module):
    """Encoder-MNO-Decoder model for CFD on unstructured 3D point clouds.

    This is the basic Multiscale Neural Operator used for the submission: the
    full velocity history is flattened into a per-point feature vector, encoded,
    mixed by a stack of geometry-aware MNO blocks, and decoded to a residual
    around a persistence baseline. (An earlier GRU-based latent-temporal variant
    lived here but was not used for the submission.)
    """

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
        # Accepted for trainer compatibility; the basic MNO is light enough that
        # activation checkpointing is unnecessary, so the flag is a no-op here.
        activation_checkpointing: bool = False,
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
        self.horizon_mlp = _make_mlp(
            self.time_embedding.out_dim,
            latent_dim,
            latent_dim,
            num_hidden_layers=1,
        )

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
            self._load_pretrained_weights()

    def _load_pretrained_weights(self) -> None:
        """Load packaged weights from ``state_dict.pt`` if present.

        Mirrors the convention used by the other self-contained submissions: the
        trained ``state_dict.pt`` ships alongside this file and is loaded
        automatically at construction time. If it is absent (e.g. during a fresh
        training run before any checkpoint exists) the model keeps its random
        initialisation.
        """
        weights_path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.load_state_dict(state)
            print(f"[SpatiotemporalMNO] Loaded weights from {weights_path}")
        else:
            print(
                "[SpatiotemporalMNO] No state_dict.pt found next to model.py — "
                "using random initialisation"
            )

    @staticmethod
    def _build_airfoil_mask(
        idcs_airfoil: list[torch.Tensor],
        batch_size: int,
        num_pos: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return build_airfoil_mask(
            idcs_airfoil,
            batch_size,
            num_pos,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _compute_wall_distance(
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
    ) -> torch.Tensor:
        return compute_wall_distance(pos, idcs_airfoil)

    @staticmethod
    def _compute_surface_frame(
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
    ) -> torch.Tensor:
        return compute_surface_frame(pos, idcs_airfoil)

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

        for batch_idx in range(batch_size):
            points = pos[batch_idx]
            for start in range(0, num_pos, self.knn_query_chunk_size):
                end = min(start + self.knn_query_chunk_size, num_pos)
                dist = torch.cdist(points[start:end], points)
                row_indices = torch.arange(end - start, device=pos.device)
                col_indices = torch.arange(start, end, device=pos.device)
                dist[row_indices, col_indices] = float("inf")
                all_indices[batch_idx, start:end, :] = torch.topk(
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

        for batch_idx in range(batch_size):
            edge_index = torch_cluster_knn_graph(
                pos[batch_idx],
                k=k_eff,
                loop=False,
            )
            src, dst = edge_index[0], edge_index[1]
            order = torch.argsort(dst)
            all_indices[batch_idx] = src[order].view(num_pos, k_eff)

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
                velocity_std_local = velocity_in.std(
                    dim=(1, 2), unbiased=False, keepdim=True
                ).clamp_min(1e-6)
            velocity_in = (velocity_in - velocity_mean_local) / velocity_std_local
            velocity_mean = velocity_mean_local.squeeze(1).squeeze(1)  # (B, 3)
            velocity_std = velocity_std_local.squeeze(1).squeeze(1)  # (B, 3)
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
        time_feat = time_feat.unsqueeze(1).expand(-1, num_pos, -1)  # (B, N, T_total*2F)
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

        encoder_input = torch.cat(
            (
                pos,
                velocity_feat,
                time_feat,
                airfoil_mask,
                wall_distance_feat,
                surface_frame_feat,
            ),
            dim=-1,
        )
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
        horizon_scale = torch.sigmoid(
            horizon_gate.mean(dim=-1, keepdim=True)
        )  # (B, T_out, 1)
        residual = residual * horizon_scale.unsqueeze(2)  # (B, T_out, N, C)

        # Predict residual dynamics around a persistence baseline from the last
        # input frame.
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
            scaled_zero = scaled_zero.to(
                device=velocity_out.device, dtype=velocity_out.dtype
            )
            velocity_out = torch.where(airfoil_mask_bool, scaled_zero, velocity_out)
        else:
            velocity_out = velocity_out * (~airfoil_mask_bool).to(
                dtype=velocity_out.dtype
            )

        output = velocity_out.contiguous()

        # Unscale back to physical velocity space when auto-normalization applied.
        if _auto_normalized:
            vm = velocity_mean.view(
                batch_size, 1, 1, self.output_channels
            ).to(dtype=output.dtype)
            vs = velocity_std.view(
                batch_size, 1, 1, self.output_channels
            ).to(dtype=output.dtype)
            output = output * vs + vm

        if return_knn_indices:
            return output, knn_indices
        return output
