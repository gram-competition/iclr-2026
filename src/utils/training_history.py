from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TrainingEpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    train_hint_metric: float
    val_hint_metric: float
    lr: float
    cuda_mem_allocated_gib: float | None = None
    cuda_mem_reserved_gib: float | None = None
    cuda_peak_allocated_gib: float | None = None
    cuda_peak_reserved_gib: float | None = None


@dataclass(frozen=True)
class TrainingArtifacts:
    csv_path: Path
    json_path: Path
    plot_path: Path | None
    plot_error: str | None


def _sanitize_run_name(run_name: str) -> str:
    safe_name = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in run_name.strip()
    )
    safe_name = safe_name.strip("_")
    return safe_name or "run"


def _write_history_csv(records: Sequence[TrainingEpochRecord], output_path: Path) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_hint_metric",
        "val_hint_metric",
        "lr",
        "cuda_mem_allocated_gib",
        "cuda_mem_reserved_gib",
        "cuda_peak_allocated_gib",
        "cuda_peak_reserved_gib",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _write_history_json(records: Sequence[TrainingEpochRecord], output_path: Path) -> None:
    payload = [asdict(record) for record in records]
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def _plot_loss_curves(records: Sequence[TrainingEpochRecord], output_path: Path) -> None:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is not installed. Install it with 'pip install matplotlib' to save loss plots."
        ) from exc

    epochs = [record.epoch for record in records]
    train_losses = [record.train_loss for record in records]
    val_losses = [record.val_loss for record in records]

    figure = Figure(figsize=(8, 5))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(epochs, train_losses, label="Train RL2 (scaled)", linewidth=2.0)
    axis.plot(epochs, val_losses, label="Val RL2 (scaled)", linewidth=2.0)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training vs Validation Loss")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)


def persist_training_artifacts(
    records: Sequence[TrainingEpochRecord],
    *,
    output_dir: Path,
    run_name: str,
) -> TrainingArtifacts:
    if not records:
        raise ValueError("Cannot persist training artifacts without at least one epoch record.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_run_name = _sanitize_run_name(run_name)
    csv_path = output_dir / f"{safe_run_name}_history.csv"
    json_path = output_dir / f"{safe_run_name}_history.json"
    plot_candidate_path = output_dir / f"{safe_run_name}_loss.png"

    _write_history_csv(records, csv_path)
    _write_history_json(records, json_path)

    plot_path: Path | None = None
    plot_error: str | None = None
    try:
        _plot_loss_curves(records, plot_candidate_path)
        plot_path = plot_candidate_path
    except Exception as exc:
        plot_error = f"{type(exc).__name__}: {exc}"

    return TrainingArtifacts(
        csv_path=csv_path,
        json_path=json_path,
        plot_path=plot_path,
        plot_error=plot_error,
    )
