"""Training entry point.

Runs a joint two-head objective:

    loss = CE(class_logits, class_label)
         + binary_weight * CE(binary_logits, binary_label)

The multi-class term is the main task; the binary term is an auxiliary signal
that pushes the trunk to separate "any artifact" from "clean" even when it
confuses two artifact types with each other.

Usage::

    python -m src.train --epochs 10 --arch small_cnn
    python -m src.train --arch resnet18 --pretrained --freeze-backbone --lr 1e-3

Checkpoints are written to ``checkpoints/``: ``last.pt`` every epoch and
``best.pt`` whenever validation multi-class accuracy improves.  Each checkpoint
carries the full arg namespace and the class names, so ``evaluate.py`` can
rebuild the exact model without being told the architecture again.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .dataset import build_datasets
from .model import build_model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class EpochMetrics:
    """Aggregated results for one pass over a dataloader."""

    loss: float = 0.0
    binary_accuracy: float = 0.0
    class_accuracy: float = 0.0
    per_class_accuracy: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line human-readable digest for the console."""
        return (
            f"loss {self.loss:.4f} | "
            f"binary acc {self.binary_accuracy:.3f} | "
            f"class acc {self.class_accuracy:.3f}"
        )


class MetricAccumulator:
    """Streaming accumulator for loss, both accuracies and per-class recall.

    Kept out of the loop bodies so train and validation share one definition of
    every metric.
    """

    def __init__(self, class_names: tuple[str, ...]) -> None:
        self.class_names = class_names
        self._loss_sum = 0.0
        self._count = 0
        self._binary_correct = 0
        self._class_correct = 0
        self._per_class_correct = np.zeros(len(class_names), dtype=np.int64)
        self._per_class_total = np.zeros(len(class_names), dtype=np.int64)

    def update(
        self,
        loss: float,
        binary_pred: torch.Tensor,
        binary_true: torch.Tensor,
        class_pred: torch.Tensor,
        class_true: torch.Tensor,
    ) -> None:
        """Fold one batch into the running totals."""
        batch_size = class_true.numel()
        self._loss_sum += loss * batch_size
        self._count += batch_size
        self._binary_correct += int((binary_pred == binary_true).sum())
        self._class_correct += int((class_pred == class_true).sum())

        truth = class_true.cpu().numpy()
        hits = (class_pred == class_true).cpu().numpy()
        for class_index in range(len(self.class_names)):
            mask = truth == class_index
            self._per_class_total[class_index] += int(mask.sum())
            self._per_class_correct[class_index] += int(hits[mask].sum())

    def result(self) -> EpochMetrics:
        """Finalise the running totals into an :class:`EpochMetrics`."""
        if self._count == 0:
            return EpochMetrics()
        # Classes absent from this epoch report NaN rather than a misleading 0.
        with np.errstate(invalid="ignore"):
            per_class = np.where(
                self._per_class_total > 0,
                self._per_class_correct / np.maximum(self._per_class_total, 1),
                np.nan,
            )
        return EpochMetrics(
            loss=self._loss_sum / self._count,
            binary_accuracy=self._binary_correct / self._count,
            class_accuracy=self._class_correct / self._count,
            per_class_accuracy={
                name: float(value) for name, value in zip(self.class_names, per_class)
            },
        )


def format_per_class(metrics: EpochMetrics) -> str:
    """Render per-class accuracy as an aligned block for the console."""
    width = max(len(name) for name in metrics.per_class_accuracy) if (
        metrics.per_class_accuracy
    ) else 0
    lines = []
    for name, value in metrics.per_class_accuracy.items():
        shown = "  n/a" if np.isnan(value) else f"{value:.3f}"
        lines.append(f"    {name:<{width}}  {shown}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Train / validate loops
# ---------------------------------------------------------------------------
def compute_loss(
    outputs,
    binary_true: torch.Tensor,
    class_true: torch.Tensor,
    criterion: nn.Module,
    binary_weight: float,
) -> torch.Tensor:
    """Joint multi-class + weighted auxiliary binary cross-entropy."""
    class_loss = criterion(outputs.class_logits, class_true)
    binary_loss = criterion(outputs.binary_logits, binary_true)
    return class_loss + binary_weight * binary_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    binary_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    """Run one pass over ``loader``.

    Passing an ``optimizer`` makes this a training pass (gradients enabled);
    omitting it makes it a validation pass under ``torch.no_grad``.
    """
    training = optimizer is not None
    model.train(training)
    accumulator = MetricAccumulator(config.CLASS_NAMES)

    for images, binary_true, class_true in loader:
        images = images.to(device, non_blocking=True)
        binary_true = binary_true.to(device, non_blocking=True)
        class_true = class_true.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = compute_loss(
                outputs, binary_true, class_true, criterion, binary_weight
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        accumulator.update(
            loss=float(loss.detach()),
            binary_pred=outputs.binary_logits.argmax(dim=1),
            binary_true=binary_true,
            class_pred=outputs.class_logits.argmax(dim=1),
            class_true=class_true,
        )

    return accumulator.result()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: EpochMetrics,
    args: argparse.Namespace,
) -> None:
    """Write a self-describing checkpoint.

    The saved ``args`` and ``class_names`` let :mod:`src.evaluate` rebuild the
    identical architecture from the file alone.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": {
                "loss": metrics.loss,
                "binary_accuracy": metrics.binary_accuracy,
                "class_accuracy": metrics.class_accuracy,
                "per_class_accuracy": metrics.per_class_accuracy,
            },
            "args": vars(args),
            "class_names": list(config.CLASS_NAMES),
            "binary_class_names": list(config.BINARY_CLASS_NAMES),
        },
        path,
    )


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto|cpu|cuda`` to a concrete device, warning on fallback."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI."""
    parser = argparse.ArgumentParser(
        description="Train a GPU render-artifact detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--clean-dir", type=Path, default=config.CLEAN_DIR,
                      help="directory of clean source frames")
    data.add_argument("--image-size", type=int, default=config.IMAGE_SIZE,
                      help="frames are resized to this square resolution")
    data.add_argument("--samples-per-image", type=int,
                      default=config.DEFAULT_SAMPLES_PER_IMAGE,
                      help="generated samples per clean frame per epoch")
    data.add_argument("--clean-ratio", type=float, default=config.DEFAULT_CLEAN_RATIO,
                      help="fraction of samples left uncorrupted")
    data.add_argument("--severity-min", type=float,
                      default=config.DEFAULT_SEVERITY_RANGE[0],
                      help="lower bound of the sampled artifact severity")
    data.add_argument("--severity-max", type=float,
                      default=config.DEFAULT_SEVERITY_RANGE[1],
                      help="upper bound of the sampled artifact severity")
    data.add_argument("--val-fraction", type=float, default=config.DEFAULT_VAL_FRACTION,
                      help="fraction of clean frames held out for validation")
    data.add_argument("--num-workers", type=int, default=0,
                      help="dataloader worker processes (0 is safest on Windows)")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--arch", choices=config.SUPPORTED_ARCHS,
                             default=config.DEFAULT_ARCH, help="architecture")
    model_group.add_argument("--pretrained", action="store_true",
                             help="resnet18: start from ImageNet weights (downloads)")
    model_group.add_argument("--freeze-backbone", action="store_true",
                             help="resnet18: train only the heads")
    model_group.add_argument("--width", type=int, default=config.DEFAULT_WIDTH,
                             help="small_cnn: base channel count")
    model_group.add_argument("--dropout", type=float, default=config.DEFAULT_DROPOUT,
                             help="dropout probability")

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    optim.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    optim.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    optim.add_argument("--weight-decay", type=float, default=config.DEFAULT_WEIGHT_DECAY)
    optim.add_argument("--binary-loss-weight", type=float,
                       default=config.DEFAULT_BINARY_LOSS_WEIGHT,
                       help="weight of the auxiliary binary head loss")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    runtime.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    runtime.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    runtime.add_argument("--history-file", type=Path, default=None,
                         help="optional JSON file to append per-epoch metrics to")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid CLI combinations up front rather than mid-training."""
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.lr <= 0:
        raise ValueError(f"--lr must be > 0, got {args.lr}")
    if not 0.0 <= args.severity_min <= args.severity_max <= 1.0:
        raise ValueError(
            "severity bounds must satisfy 0 <= min <= max <= 1, got "
            f"({args.severity_min}, {args.severity_max})"
        )
    if args.arch != "resnet18" and (args.pretrained or args.freeze_backbone):
        print(
            f"[warn] --pretrained/--freeze-backbone apply to resnet18 only; "
            f"ignored for arch={args.arch}"
        )


def main(argv: list[str] | None = None) -> int:
    """Train the detector; returns a process exit code."""
    args = build_parser().parse_args(argv)

    # Turn expected setup failures into a clean message rather than a traceback;
    # anything else is a real bug and should surface in full.
    try:
        validate_args(args)
        set_seed(args.seed)
        device = resolve_device(args.device)
        train_set, val_set = build_datasets(
            clean_dir=args.clean_dir,
            image_size=args.image_size,
            samples_per_image=args.samples_per_image,
            clean_ratio=args.clean_ratio,
            severity_range=(args.severity_min, args.severity_max),
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}")
        return 2

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(
        arch=args.arch,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        width=args.width,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(
        f"device={device} arch={args.arch} "
        f"params={model.num_parameters():,} "
        f"train={len(train_set)} val={len(val_set)} samples"
    )

    history: list[dict] = []
    best_accuracy = -1.0
    checkpoint_dir = Path(args.checkpoint_dir)

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, device, criterion,
            args.binary_loss_weight, optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model, val_loader, device, criterion, args.binary_loss_weight
        )
        scheduler.step()
        elapsed = time.perf_counter() - started

        print(f"\nepoch {epoch}/{args.epochs}  ({elapsed:.1f}s)")
        print(f"  train  {train_metrics.summary()}")
        print(f"  val    {val_metrics.summary()}")
        print("  val per-class accuracy:")
        print(format_per_class(val_metrics))

        history.append({
            "epoch": epoch,
            "train": vars(train_metrics),
            "val": vars(val_metrics),
        })

        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, epoch, val_metrics, args
        )
        if val_metrics.class_accuracy > best_accuracy:
            best_accuracy = val_metrics.class_accuracy
            save_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, epoch, val_metrics, args
            )
            print(f"  -> new best (class acc {best_accuracy:.3f}), saved best.pt")

    if args.history_file is not None:
        args.history_file.parent.mkdir(parents=True, exist_ok=True)
        args.history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"\nwrote training history to {args.history_file}")

    print(f"\ndone. best validation class accuracy: {best_accuracy:.3f}")
    print(f"checkpoints in {checkpoint_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
