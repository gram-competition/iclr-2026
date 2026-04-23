#!/usr/bin/env python3
"""Export ``best.pt`` for PR submission as ``models/fmgreco_stnn/state_dict.pt``.

Writes a small dict: ``{"model_state_dict", "args"}`` so ``infer.py`` can rebuild
the network with the same architecture as training (``k``, width, etc.).

Copy ``best.pt`` from the cluster first, e.g.::

    kubectl cp default/<pod>:/data/checkpoints/best.pt ./best.pt

Then::

    python scripts/export_best_for_submission.py ./best.pt \\
      --output models/fmgreco_stnn/state_dict.pt

Commit ``state_dict.pt`` via Git LFS if the file is large."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Training checkpoint (e.g. best.pt from /data/checkpoints/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/fmgreco_stnn/state_dict.pt"),
        help="Destination path (default: models/fmgreco_stnn/state_dict.pt).",
    )
    args = parser.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model_state_dict" not in ck:
        raise SystemExit(
            "Expected a training checkpoint dict with key 'model_state_dict'."
        )
    payload = {
        "model_state_dict": ck["model_state_dict"],
        "args": ck.get("args"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Wrote {args.output.resolve()} (model_state_dict + args)")


if __name__ == "__main__":
    main()
