"""Score two checkpoints against the same frames and report the difference.

Used to answer one question: does training on real game captures beat training
on procedurally drawn frames, when both are judged on real captures?  Both
models see an identical set of faults injected into identical held-out frames,
so the only variable is what each learned from.

Usage::

    python scripts/compare_models.py \
        --baseline checkpoints/best.pt --candidate checkpoints/real/best.pt \
        --frames data/game_frames
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

DEFAULT_SAMPLES_PER_FAULT: int = 60
SEVERITY_BANDS: tuple[tuple[str, float, float], ...] = (
    ("faint\n0.2-0.4", 0.2, 0.4),
    ("mild\n0.4-0.6", 0.4, 0.6),
    ("clear\n0.6-0.8", 0.6, 0.8),
    ("severe\n0.8-1.0", 0.8, 1.0),
)


@torch.no_grad()
def evaluate_model(
    model,
    sources: list[np.ndarray],
    image_size: int,
    samples_per_fault: int,
    seed: int,
) -> dict[str, object]:
    """Score one model: per-fault recall, per-severity detection, clean accuracy.

    Every model is handed the same frames and the same seeded severities, so the
    comparison is like-for-like.
    """
    per_fault: dict[str, float] = {}
    detection_by_band: dict[str, list[float]] = {name: [] for name, _, _ in SEVERITY_BANDS}

    for artifact_index, artifact in enumerate(config.ARTIFACT_TYPES):
        correct = 0
        total = 0
        for band_name, low, high in SEVERITY_BANDS:
            rng = np.random.default_rng([seed, artifact_index, int(low * 100)])
            batch = torch.stack([
                to_tensor(apply_corruption(
                    sources[int(rng.integers(0, len(sources)))],
                    artifact, float(rng.uniform(low, high)), rng,
                ))
                for _ in range(samples_per_fault)
            ])
            output = model(batch)
            correct += int((output.class_logits.argmax(1) == artifact_index + 1).sum())
            total += samples_per_fault
            detection_by_band[band_name].append(
                float((output.binary_logits.argmax(1) == 1).float().mean())
            )
        per_fault[artifact] = correct / total

    clean_batch = torch.stack([to_tensor(image) for image in sources])
    clean_output = model(clean_batch)
    clean_accuracy = float(
        (clean_output.binary_logits.argmax(1) == 0).float().mean()
    )

    return {
        "per_fault": per_fault,
        "detection": {name: float(np.mean(values))
                      for name, values in detection_by_band.items()},
        "clean_ok": clean_accuracy,
        "overall": float(np.mean(list(per_fault.values()))),
    }


def figure_comparison(
    baseline: dict, candidate: dict, labels: tuple[str, str], output_name: str
) -> Path:
    """Plot the two models side by side, per fault and per severity band."""
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    names = list(config.ARTIFACT_TYPES)
    positions = np.arange(len(names))
    height = 0.38

    axes[0].barh(positions - height / 2, [baseline["per_fault"][n] for n in names],
                 height, color=vs.MUTED, label=labels[0])
    axes[0].barh(positions + height / 2, [candidate["per_fault"][n] for n in names],
                 height, color=vs.ACCENT, label=labels[1])
    axes[0].set_yticks(positions, names, fontsize=8.5)
    for tick, name in zip(axes[0].get_yticklabels(), names):
        tick.set_color(vs.CLASS_COLORS[name])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    vs.label(axes[0], "fault named correctly, on real frames")

    bands = [name for name, _, _ in SEVERITY_BANDS]
    band_positions = np.arange(len(bands))
    width = 0.38
    axes[1].bar(band_positions - width / 2, [baseline["detection"][b] for b in bands],
                width, color=vs.MUTED, label=labels[0])
    axes[1].bar(band_positions + width / 2, [candidate["detection"][b] for b in bands],
                width, color=vs.ACCENT, label=labels[1])
    axes[1].set_xticks(band_positions, bands, fontsize=8.5)
    axes[1].set_ylim(0, 1.05)
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    vs.label(axes[1], "caught as broken, by how faint the fault is")

    # One shared legend below the plots: an in-axes legend lands on the bars,
    # which in the right-hand panel are full height.
    handles, labels_text = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels_text, fontsize=8.5, ncols=2,
                  loc="lower center", bbox_to_anchor=(0.5, -0.04))
    # The title names whatever the two checkpoints actually are, so a
    # comparison run with custom --labels does not ship a stale headline.
    title = f"{labels[0]} vs {labels[1]}"
    figure.suptitle(
        title[:1].upper() + title[1:],
        color=vs.TEXT, fontsize=12.5, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    path = config.PROJECT_ROOT / "visuals" / f"{output_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor=vs.GROUND)
    plt.close(figure)
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build the comparison CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, default=config.CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--candidate", type=Path,
                        default=config.CHECKPOINT_DIR / "real" / "best.pt")
    parser.add_argument("--frames", type=Path, default=config.DATA_ROOT / "game_frames",
                        help="frame folder both models are judged on")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_PER_FAULT,
                        help="samples per fault per severity band")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--output-name", default="model_comparison",
                        help="figure filename stem written into visuals/")
    parser.add_argument("--labels", nargs=2,
                        default=("trained on drawings", "trained on real captures"),
                        help="legend labels for the baseline and the candidate")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the comparison; returns a process exit code."""
    args = build_parser().parse_args(argv)
    vs.apply_style()
    device = torch.device("cpu")

    try:
        baseline_model, baseline_ck = load_checkpoint(Path(args.baseline), device)
        candidate_model, candidate_ck = load_checkpoint(Path(args.candidate), device)
    except (FileNotFoundError, KeyError) as error:
        print(f"error: {error}")
        return 2

    # Judge both at the resolution the candidate was trained for, and hold out
    # the same frames the candidate never saw.
    image_size = candidate_ck["args"]["image_size"]
    _, held_out = split_paths(
        list_clean_images(args.frames),
        candidate_ck["args"]["val_fraction"],
        candidate_ck["args"]["seed"],
    )
    sources = [load_image(path, image_size) for path in held_out]
    print(f"judging both models on {len(sources)} held-out real frames at {image_size}px")

    baseline = evaluate_model(baseline_model, sources, image_size, args.samples, args.seed)
    candidate = evaluate_model(candidate_model, sources, image_size, args.samples, args.seed)

    print(f"\n{'fault':<20}{'drawn':>9}{'real':>9}{'change':>10}")
    print("-" * 48)
    for artifact in config.ARTIFACT_TYPES:
        delta = candidate["per_fault"][artifact] - baseline["per_fault"][artifact]
        print(f"{artifact:<20}{baseline['per_fault'][artifact]:>8.0%}"
              f"{candidate['per_fault'][artifact]:>9.0%}{delta:>+10.0%}")
    print("-" * 48)
    for band, _, _ in SEVERITY_BANDS:
        delta = candidate["detection"][band] - baseline["detection"][band]
        label = f"caught @ {band.split(chr(10))[1]}"
        print(f"{label:<20}{baseline['detection'][band]:>8.0%}"
              f"{candidate['detection'][band]:>9.0%}{delta:>+10.0%}")
    print("-" * 48)
    for key, label in (("clean_ok", "clean left alone"), ("overall", "overall naming")):
        delta = candidate[key] - baseline[key]
        print(f"{label:<20}{baseline[key]:>8.0%}{candidate[key]:>9.0%}{delta:>+10.0%}")

    path = figure_comparison(
        baseline, candidate, tuple(args.labels), args.output_name
    )
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
