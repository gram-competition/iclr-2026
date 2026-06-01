import os
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class ImprovedMLP(nn.Module):
    """
    Point-wise MLP using all available inputs.

    Input per point (39 channels):
        pos           (3)   — xyz coordinate
        velocity_in   (15)  — 5 time steps × 3, flattened
        t             (10)  — broadcast to every point
        pressure      (10)  — 10 time steps of pressure, per point
        dist_airfoil  (1)   — normalised distance to airfoil centroid

    Output per point (15) → reshaped to (B, 5, N, 3)
    """

    IN_CHANNELS  = 3 + 15 + 10 + 10 + 1   # = 39
    OUT_CHANNELS = 15                       # 5 × 3
    CHUNK_SIZE   = 8192

    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            ResidualBlock(self.IN_CHANNELS, 256),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 256),
        )
        self.head     = nn.Linear(256, self.OUT_CHANNELS)
        self.vel_skip = nn.Linear(15, self.OUT_CHANNELS)   # strong physical prior

        self._init_weights()

        weights_path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state)
            print(f"[ImprovedMLP] Loaded weights from {weights_path}")
        else:
            print("[ImprovedMLP] No weights found — using random init")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _dist_feature(pos, idcs_airfoil):
        B, N, _ = pos.shape
        dist = torch.zeros(B, N, 1, device=pos.device, dtype=pos.dtype)
        for b in range(B):
            idcs     = idcs_airfoil[b]
            centroid = pos[b, idcs].mean(dim=0, keepdim=True)
            d        = (pos[b] - centroid).norm(dim=-1, keepdim=True)
            dist[b]  = d / (d.max() + 1e-8)
        return dist

    def _forward_chunk(self, x, vel_flat):
        return self.head(self.backbone(x)) + self.vel_skip(vel_flat)

    def forward(
        self,
        t:            torch.Tensor,   # (B, 10)
        pos:          torch.Tensor,   # (B, N, 3)
        idcs_airfoil: list,
        velocity_in:  torch.Tensor,   # (B, 5, N, 3)
        pressure:     torch.Tensor = None,  # (B, 10, N)  — optional at test time
    ) -> torch.Tensor:

        B, T_in, N, _ = velocity_in.shape
        vel_flat = velocity_in.permute(0, 2, 1, 3).reshape(B, N, T_in * 3)  # (B,N,15)
        dist     = self._dist_feature(pos, idcs_airfoil)                      # (B,N,1)

        # pressure: (B, 10, N) → (B, N, 10)
        if pressure is not None:
            pres = pressure.permute(0, 2, 1)   # (B, N, 10)
        else:
            pres = torch.zeros(B, N, 10, device=pos.device, dtype=pos.dtype)

        outputs = []
        for start in range(0, N, self.CHUNK_SIZE):
            end    = min(start + self.CHUNK_SIZE, N)
            t_bc   = t.unsqueeze(1).expand(B, end - start, -1)  # (B, C, 10)
            x_c    = torch.cat([
                pos[:, start:end, :],       # (B, C, 3)
                vel_flat[:, start:end, :],  # (B, C, 15)
                t_bc,                       # (B, C, 10)
                pres[:, start:end, :],      # (B, C, 10)
                dist[:, start:end, :],      # (B, C, 1)
            ], dim=-1)                      # (B, C, 39)
            out_c  = self._forward_chunk(x_c, vel_flat[:, start:end, :])
            outputs.append(out_c)

        out = torch.cat(outputs, dim=1)                          # (B, N, 15)
        return out.view(B, N, T_in, 3).permute(0, 2, 1, 3)      # (B, 5, N, 3)