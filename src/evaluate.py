"""Evaluation entry point: confusion matrix and per-frame inference latency.

Reports what matters for the deployment question — *can this run in a frame
budget, and what does it get wrong?*

* **Confusion matrix** over the artifact taxonomy, printed as text and written
  as a PNG.  The off-diagonal structure is the interesting part: confusing
  ``shader_noise`` with ``stuck_pixels`` is understandable, confusing either
  with ``clean`` is a miss that matters.
* **Per-class precision / recall / F1** via scikit-learn.
* **Per-frame latency**, measured one frame at a time (batch size 1) because
  that is how a detector would run against a live render, plus a batched
  throughput figure for offline QA sweeps.  Warm-up iterations are discarded
  and CUDA is synchronised so the timings are real.

Usage::

    python -m src.evaluate --checkpoint checkpoints/best.pt
    python -m src.evaluate --checkpoint checkpoints/best.pt --latency-frames 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from . import config
from .dataset import build_datasets
from .model import CorruptionNet, build_model


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[CorruptionNet, dict]:
    """Rebuild the model described by a checkpoint and load its weights.

    The architecture is taken from the ``args`` the checkpoint was saved with,
    so the caller never has to restate ``--arch``/``--width``.

    Args:
        path: Checkpoint file written by :mod:`src.train`.
        device: Device to map the weights onto.

    Returns:
        ``(model_in_eval_mode, checkpoint_dict)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If the checkpoint is missing required fields.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {path}. Train one with `python -m src.train`."
        )

    # weights_only=False: our own checkpoint stores an argparse dict alongside
    # the tensors.  Only load checkpoints you produced.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    for key in ("model_state", "args"):
        if key not in checkpoint:
            raise KeyError(f"checkpoint {path} is missing required key {key!r}")

    saved = checkpoint["args"]
    model = build_model(
        arch=saved.get("arch", config.DEFAULT_ARCH),
        pretrained=False,               # weights come from the checkpoint
        freeze_backbone=False,
        width=saved.get("width", config.DEFAULT_WIDTH),
        dropout=saved.get("dropout", config.DEFAULT_DROPOUT),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_predictions(
    model: CorruptionNet, loader: DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    """Run the model over ``loader`` and gather predictions and truth."""
    model.eval()
    buffers: dict[str, list[np.ndarray]] = {
        "class_true": [], "class_pred": [], "binary_true": [], "binary_pred": [],
    }

    for images, binary_true, class_true in loader:
        outputs = model(images.to(device, non_blocking=True))
        buffers["class_true"].append(class_true.numpy())
        buffers["binary_true"].append(binary_true.numpy())
        buffers["class_pred"].append(outputs.class_logits.argmax(1).cpu().numpy())
        buffers["binary_pred"].append(outputs.binary_logits.argmax(1).cpu().numpy())

    return {key: np.concatenate(values) for key, values in buffers.items()}


def print_confusion_matrix(
    matrix: np.ndarray, class_names: tuple[str, ...]
) -> None:
    """Print the confusion matrix as an aligned text table (rows = truth)."""
    label_width = max(len(name) for name in class_names)
    cell_width = max(5, len(str(int(matrix.max()))) + 1)

    header = " " * (label_width + 2) + "".join(
        f"{index:>{cell_width}}" for index in range(len(class_names))
    )
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(header)
    for index, name in enumerate(class_names):
        row = "".join(f"{int(value):>{cell_width}}" for value in matrix[index])
        print(f"{name:<{label_width}}  {row}   [{index}]")


def save_confusion_matrix_plot(
    matrix: np.ndarray, class_names: tuple[str, ...], path: Path
) -> None:
    """Write a normalised confusion-matrix heatmap to ``path``."""
    import matplotlib
    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.divide(matrix, np.maximum(row_sums, 1))

    figure, axes = plt.subplots(figsize=(7, 6))
    image = axes.imshow(normalised, cmap="viridis", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axes, label="fraction of true class")

    axes.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axes.set_yticks(range(len(class_names)), class_names)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    axes.set_title("Artifact classification confusion matrix")

    # Annotate each cell with the raw count; white text on dark cells.
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axes.text(
                col, row, str(int(matrix[row, col])),
                ha="center", va="center", fontsize=8,
                color="white" if normalised[row, col] < 0.6 else "black",
            )

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------
def _synchronize(device: torch.device) -> None:
    """Block until queued GPU work finishes, so timings are not optimistic."""
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_latency(
    model: CorruptionNet,
    device: torch.device,
    image_size: int,
    frames: int = config.DEFAULT_LATENCY_FRAMES,
    warmup: int = config.DEFAULT_LATENCY_WARMUP,
    batch_size: int = 1,
) -> dict[str, float]:
    """Time ``frames`` forward passes and summarise per-frame latency.

    Uses synthetic input so the measurement isolates model cost from image
    decoding.  Warm-up passes are discarded: the first calls pay for lazy
    kernel selection and allocator growth.

    Returns:
        Dict with mean/median/p95/min/max milliseconds per frame and the
        implied frames per second.
    """
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    model.eval()
    batch = torch.randn(batch_size, 3, image_size, image_size, device=device)

    for _ in range(warmup):
        model(batch)
    _synchronize(device)

    timings_ms: list[float] = []
    for _ in range(frames):
        started = time.perf_counter()
        model(batch)
        _synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings_ms.append(elapsed_ms / batch_size)  # per frame, not per batch

    samples = np.asarray(timings_ms)
    return {
        "batch_size": float(batch_size),
        "frames": float(frames),
        "mean_ms": float(samples.mean()),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": float(samples.min()),
        "max_ms": float(samples.max()),
        "fps": float(1000.0 / samples.mean()) if samples.mean() > 0 else float("inf"),
    }


def print_latency(title: str, stats: dict[str, float]) -> None:
    """Print one latency block."""
    print(
        f"  {title:<22} mean {stats['mean_ms']:7.3f} ms | "
        f"median {stats['median_ms']:7.3f} ms | "
        f"p95 {stats['p95_ms']:7.3f} ms | "
        f"{stats['fps']:8.1f} fps"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained corruption detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / "best.pt",
                        help="checkpoint written by src.train")
    parser.add_argument("--clean-dir", type=Path, default=config.CLEAN_DIR)
    parser.add_argument("--image-size", type=int, default=None,
                        help="override the checkpoint's training resolution")
    parser.add_argument("--samples-per-image", type=int,
                        default=config.DEFAULT_SAMPLES_PER_IMAGE * 2,
                        help="evaluation samples per clean frame")
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--val-fraction", type=float,
                        default=config.DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--latency-frames", type=int,
                        default=config.DEFAULT_LATENCY_FRAMES)
    parser.add_argument("--latency-warmup", type=int,
                        default=config.DEFAULT_LATENCY_WARMUP)
    parser.add_argument("--output-dir", type=Path, default=config.CHECKPOINT_DIR,
                        help="where the confusion matrix PNG and metrics JSON go")
    parser.add_argument("--skip-plot", action="store_true",
                        help="skip writing the confusion matrix PNG")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate a checkpoint; returns a process exit code."""
    from .train import resolve_device  # local import avoids a circular import

    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)

    # A missing or malformed checkpoint is expected user error, not a crash.
    try:
        model, checkpoint = load_checkpoint(Path(args.checkpoint), device)
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}")
        return 2
    saved_args = checkpoint["args"]
    image_size = args.image_size or saved_args.get("image_size", config.IMAGE_SIZE)

    print(
        f"loaded {args.checkpoint} (epoch {checkpoint.get('epoch', '?')}, "
        f"arch {saved_args.get('arch', '?')}, "
        f"{model.num_parameters(trainable_only=False):,} params)"
    )

    # Rebuild the same held-out split the checkpoint was validated against.
    _, val_set = build_datasets(
        clean_dir=args.clean_dir,
        image_size=image_size,
        samples_per_image=args.samples_per_image,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    print(f"evaluating {len(val_set)} samples on {device}")

    predictions = collect_predictions(model, loader, device)
    class_names = config.CLASS_NAMES

    matrix = confusion_matrix(
        predictions["class_true"], predictions["class_pred"],
        labels=list(range(len(class_names))),
    )
    print_confusion_matrix(matrix, class_names)

    print("\nPer-class report (multi-class head):")
    report = classification_report(
        predictions["class_true"], predictions["class_pred"],
        labels=list(range(len(class_names))), target_names=list(class_names),
        digits=3, zero_division=0,
    )
    print(report)

    binary_accuracy = float(
        (predictions["binary_true"] == predictions["binary_pred"]).mean()
    )
    class_accuracy = float(
        (predictions["class_true"] == predictions["class_pred"]).mean()
    )
    print(f"binary accuracy (corrupted vs clean): {binary_accuracy:.4f}")
    print(f"multi-class accuracy:                 {class_accuracy:.4f}")

    print(f"\nInference latency at {image_size}x{image_size} on {device}:")
    single = measure_latency(
        model, device, image_size,
        frames=args.latency_frames, warmup=args.latency_warmup, batch_size=1,
    )
    print_latency("per frame (batch 1)", single)
    batched = measure_latency(
        model, device, image_size,
        frames=max(1, args.latency_frames // args.batch_size),
        warmup=args.latency_warmup, batch_size=args.batch_size,
    )
    print_latency(f"batched (batch {args.batch_size})", batched)

    output_dir = Path(args.output_dir)
    if not args.skip_plot:
        plot_path = output_dir / config.CONFUSION_MATRIX_FILENAME
        save_confusion_matrix_plot(matrix, class_names, plot_path)
        print(f"\nconfusion matrix written to {plot_path}")

    metrics_path = output_dir / config.METRICS_FILENAME
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "device": str(device),
                "image_size": image_size,
                "num_samples": int(predictions["class_true"].size),
                "binary_accuracy": binary_accuracy,
                "class_accuracy": class_accuracy,
                "class_names": list(class_names),
                "confusion_matrix": matrix.tolist(),
                "latency_single": single,
                "latency_batched": batched,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"metrics written to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
