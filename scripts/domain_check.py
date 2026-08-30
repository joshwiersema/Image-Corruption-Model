"""Measure how well a trained detector transfers to a different source of frames.

The model learns from whatever sits in ``data/clean``.  If those frames are
procedurally drawn, the fault signatures it learns are entangled with the look
of that generator — flat colour, hard edges, no lens or sensor character.  This
script quantifies the gap by scoring the same checkpoint twice: once on the
frames it was built around, once on a folder of genuinely different images.

Usage::

    python scripts/domain_check.py --real-dir "C:/Windows/Web/Wallpaper"
    python scripts/domain_check.py --real-dir photos/ --samples 6

A large drop means the detector learned the generator, not the fault.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.corruption import apply_corruption  # noqa: E402
from src.dataset import list_clean_images, load_image, split_paths, to_tensor  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402
import viz_style as vs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_SAMPLES_PER_FAULT: int = 8
SHOWCASE_FAULTS: tuple[str, ...] = ("block_corruption", "shader_noise", "texture_smear")


def collect_images(directory: Path, limit: int | None = None) -> list[Path]:
    """Return image paths under ``directory``, searching subfolders too."""
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"no images found under {directory}")
    return paths[:limit] if limit else paths


@torch.no_grad()
def score(
    model, sources: list[np.ndarray], samples_per_fault: int, seed: int
) -> dict[str, float]:
    """Score a set of clean frames: per-fault recall plus overall detection.

    Returns:
        Per-artifact naming recall, keyed by artifact name, plus ``_detected``
        (share of corrupted frames flagged as broken at all) and ``_clean_ok``
        (share of untouched frames correctly left alone).
    """
    results: dict[str, float] = {}
    detected: list[float] = []

    for artifact_index, artifact in enumerate(config.ARTIFACT_TYPES):
        rng = np.random.default_rng([seed, artifact_index])
        batch = torch.stack([
            to_tensor(apply_corruption(
                sources[int(rng.integers(0, len(sources)))],
                artifact, float(rng.uniform(0.3, 0.9)), rng,
            ))
            for _ in range(samples_per_fault)
        ])
        output = model(batch)
        results[artifact] = float(
            (output.class_logits.argmax(1) == artifact_index + 1).float().mean()
        )
        detected.append(float((output.binary_logits.argmax(1) == 1).float().mean()))

    clean_batch = torch.stack([to_tensor(image) for image in sources])
    clean_output = model(clean_batch)
    results["_clean_ok"] = float(
        (clean_output.class_logits.argmax(1) == config.CLEAN_CLASS_INDEX).float().mean()
    )
    results["_detected"] = float(np.mean(detected))
    return results


def figure_comparison(
    model,
    real_sources: list[np.ndarray],
    drawn: dict[str, float],
    real: dict[str, float],
) -> Path:
    """Plot faults on real photos beside the drawn-vs-real accuracy gap."""
    columns = len(SHOWCASE_FAULTS) + 1
    figure = plt.figure(figsize=(4.1 * columns * 0.62 + 7.2, 5.6))
    # One empty column between the image strip and the chart, so the chart's
    # left-hand class labels never land on top of the frames.
    grid = figure.add_gridspec(2, columns + 4, hspace=0.35, wspace=0.3)

    rng = np.random.default_rng(4)
    for row in range(2):
        # Spread the picks across the folder rather than taking neighbours,
        # which in a wallpaper set are often variants of one image.
        source = real_sources[(row * 2 + 1) * len(real_sources) // 4]
        axis = figure.add_subplot(grid[row, 0])
        axis.imshow(source)
        vs.strip_axes(axis)
        if row == 0:
            axis.set_title("real photo", fontsize=8.5, color=vs.MUTED, family=vs.MONO)

        for column, artifact in enumerate(SHOWCASE_FAULTS, start=1):
            damaged = apply_corruption(source, artifact, 0.7, rng)
            axis = figure.add_subplot(grid[row, column])
            axis.imshow(damaged)
            vs.strip_axes(axis)
            with torch.no_grad():
                predicted = int(
                    model(to_tensor(damaged).unsqueeze(0)).class_logits.argmax(1)
                )
            correct = config.CLASS_NAMES[predicted] == artifact
            for spine in axis.spines.values():
                spine.set_color(vs.GOOD if correct else vs.BAD)
                spine.set_linewidth(1.6)
            if row == 0:
                axis.set_title(
                    artifact.replace("_", "\n"), fontsize=8, color=vs.CLASS_COLORS[artifact],
                    family=vs.MONO,
                )
            axis.set_xlabel(
                f"said {config.CLASS_NAMES[predicted]}", fontsize=7, family=vs.MONO,
                color=vs.GOOD if correct else vs.BAD, labelpad=4,
            )

    bar_axis = figure.add_subplot(grid[:, columns + 1:])
    names = list(config.ARTIFACT_TYPES)
    positions = np.arange(len(names))
    height = 0.38
    bar_axis.barh(positions - height / 2, [drawn[n] for n in names], height,
                  color=vs.ACCENT, label="drawn frames (what it trained on)")
    bar_axis.barh(positions + height / 2, [real[n] for n in names], height,
                  color=vs.CLASS_COLORS["channel_swap"], label="real photos (never seen)")
    bar_axis.set_yticks(positions, names, fontsize=8.5)
    for tick, name in zip(bar_axis.get_yticklabels(), names):
        tick.set_color(vs.CLASS_COLORS[name])
    bar_axis.invert_yaxis()
    bar_axis.set_xlim(0, 1)
    bar_axis.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    bar_axis.legend(fontsize=8, loc="lower right")
    vs.label(bar_axis, "fault named correctly")

    figure.suptitle(
        "The same detector, pointed at pictures it was not built around",
        color=vs.TEXT, fontsize=12.5, fontweight="bold",
    )
    path = config.PROJECT_ROOT / "visuals" / "real_vs_drawn.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor=vs.GROUND)
    plt.close(figure)
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build the domain-check CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real-dir", type=Path, required=True,
                        help="folder of genuinely different images to test against")
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_PER_FAULT,
                        help="corrupted samples generated per fault type")
    parser.add_argument("--limit", type=int, default=24,
                        help="maximum real images to load")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Compare in-domain and out-of-domain performance; returns an exit code."""
    args = build_parser().parse_args(argv)
    vs.apply_style()

    try:
        model, checkpoint = load_checkpoint(Path(args.checkpoint), torch.device("cpu"))
        real_paths = collect_images(Path(args.real_dir), args.limit)
    except (FileNotFoundError, KeyError) as error:
        print(f"error: {error}")
        return 2

    saved = checkpoint["args"]
    image_size = saved["image_size"]
    _, drawn_paths = split_paths(
        list_clean_images(config.CLEAN_DIR), saved["val_fraction"], saved["seed"]
    )

    drawn_sources = [load_image(path, image_size) for path in drawn_paths]
    real_sources = [load_image(path, image_size) for path in real_paths]
    print(f"drawn frames: {len(drawn_sources)}   real images: {len(real_sources)}")

    drawn = score(model, drawn_sources, args.samples, args.seed)
    real = score(model, real_sources, args.samples, args.seed)

    print(f"\n{'fault':<20}{'drawn':>9}{'real':>9}{'change':>10}")
    print("-" * 48)
    for artifact in config.ARTIFACT_TYPES:
        delta = real[artifact] - drawn[artifact]
        print(f"{artifact:<20}{drawn[artifact]:>8.0%}{real[artifact]:>9.0%}{delta:>+10.0%}")
    print("-" * 48)
    for key, label in (("_detected", "caught at all"), ("_clean_ok", "clean left alone")):
        delta = real[key] - drawn[key]
        print(f"{label:<20}{drawn[key]:>8.0%}{real[key]:>9.0%}{delta:>+10.0%}")

    path = figure_comparison(model, real_sources, drawn, real)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
