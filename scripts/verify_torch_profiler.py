#!/usr/bin/env python3
"""
Smoke-test torch.profiler on a CUDA/HIP device (NVIDIA or ROCm PyTorch).

Run locally or inside rocm/pytorch on a GPU node, e.g.:

  python3 scripts/verify_torch_profiler.py
  TORCH_PROFILER_DIR=/tmp/pf python3 scripts/verify_torch_profiler.py

If this succeeds, torch.profiler + Chrome trace export works in your image.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP not available; run on a GPU pod / machine.")

    device = torch.device("cuda", 0)
    hip = getattr(torch.version, "hip", None)
    print(f"device={device} torch={torch.__version__} hip={hip}")

    out = Path(os.environ.get("TORCH_PROFILER_DIR", "/tmp/torch_profiler_verify"))
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "verify-chrome-trace.json"

    x = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    w = torch.randn(4096, 4096, device=device, dtype=torch.float32)

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
        on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)),
    ) as prof:
        for _ in range(5):
            y = x @ w
            y.sum().item()
            prof.step()

    print(f"Wrote Chrome trace: {trace_path.resolve()}")
    print("Open in chrome://tracing or https://ui.perfetto.dev/")

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=12)
    print(table)


if __name__ == "__main__":
    main()
