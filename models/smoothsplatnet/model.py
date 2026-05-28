"""
SmoothSplatNet: submission wrapper around the best 3-member Smooth-Splat ensemble.

"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .backbone import SmoothSplatBackbone


class SmoothSplatNet(nn.Module):
    """Three-member standalone SmoothSplat backbone ensemble.

    The default submission path uses the strongest hard162 raw ensemble found in
    local ablations: seeds 52, 57, and 85, averaged equally with no TTA.
    """

    CHECKPOINT_NAMES = (
        "state_dict_seed52.pt",
        "state_dict_seed57.pt",
        "state_dict_seed85.pt",
    )

    def __init__(self):
        super().__init__()
        self.members = nn.ModuleList()

        for ckpt_name in self.CHECKPOINT_NAMES:
            ckpt_path = self._resolve_checkpoint(ckpt_name)
            member = self._build_member(ckpt_path)
            self.members.append(member)

        # Submission inference should default to eval mode so BatchNorm layers
        # use stored running statistics even if the caller does not call eval().
        self.eval()

    @classmethod
    def _resolve_checkpoint(cls, ckpt_name: str) -> Path:
        here = Path(__file__).resolve().parent
        ckpt_path = here / ckpt_name
        if ckpt_path.is_file():
            return ckpt_path
        raise FileNotFoundError(
            f"SmoothSplatNet could not find checkpoint {ckpt_name!r} at {ckpt_path}"
        )

    @staticmethod
    def _build_member(ckpt_path: Path) -> SmoothSplatBackbone:
        member = SmoothSplatBackbone()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        member.load_state_dict(state)
        member.eval()
        return member

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        pred_sum = None
        for member in self.members:
            pred = member(t, pos, idcs_airfoil, velocity_in)
            pred_sum = pred if pred_sum is None else pred_sum + pred
        return pred_sum / len(self.members)


class Model(SmoothSplatNet):
    """Challenge submission entrypoint."""
