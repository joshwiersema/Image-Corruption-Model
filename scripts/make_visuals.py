"""Render every figure that explains what the detector sees and how it does.

Usage::

    python scripts/make_visuals.py                       # all figures
    python scripts/make_visuals.py --only catalog curves

Figures are written to ``visuals/``.  Each one answers a specific question:

``artifact_catalog``   what does each fault actually look like, by severity?
``channel_swap_flaw``  why is one artifact unlearnable at full severity?
``severity_curves``    where does the detector stop seeing the fault?
``confusion``          what does it mistake for what?
``class_balance``      how often does it see each class while training?
``training_curves``    did training finish, or was it still improving?
``predictions``        show me actual verdicts on actual frames.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Both the project root (for ``src``) and this directory (for ``viz_style``):
# importing ``scripts.viz_style`` as a package would collide with any other
# top-level ``scripts`` package that happens to be on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.corruption import apply_corruption  # noqa: E402
from src.dataset import list_clean_images, load_image, split_paths, to_tensor  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402
import viz_style as vs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

VISUAL_DIR: Path = config.PROJECT_ROOT / "visuals"
SEVERITY_STEPS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
GALLERY_COLUMNS: int = 6
GALLERY_ROWS: int = 2
#: Held-out accuracy gain over the last few epochs below which a run has plateaued.
PLATEAU_TOLERANCE: float = 0.01


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def load_context(checkpoint: Path, device: torch.device):
    """Return ``(model, saved_args, val_paths)`` for the trained checkpoint."""
    model, ck = load_checkpoint(checkpoint, device)
    saved = ck["args"]
    # The frames a checkpoint was validated against are named in the checkpoint
    # itself; never assume the demo folder.
    paths = list_clean_images(saved.get("clean_dir", config.CLEAN_DIR))
    _, val_paths = split_paths(paths, saved["val_fraction"], saved["seed"])
    return model, saved, val_paths, ck


@torch.no_grad()
def verdict(model, image: np.ndarray) -> tuple[int, float, float]:
    """Return ``(predicted_class, class_confidence, p_corrupted)`` for a frame."""
    out = model(to_tensor(image).unsqueeze(0))
    class_probs = out.class_logits.softmax(1)[0]
    index = int(class_probs.argmax())
    p_corrupt = float(out.binary_logits.softmax(1)[0, 1])
    return index, float(class_probs[index]), p_corrupt


def pick_showcase_frame(
    paths: list[Path], image_size: int, candidates: int = 40
) -> np.ndarray:
    """Choose the frame that shows faults most legibly.

    A dark, flat frame hides colour-shift and smear artifacts entirely, which
    makes for a useless catalog. Score a sample on brightness spread and colour
    variety and take the best, so the figure works whatever folder it is given.
    """
    step = max(1, len(paths) // candidates)
    sample = paths[::step][:candidates]
    best_image, best_score = None, -1.0

    for path in sample:
        image = load_image(path, image_size)
        pixels = image.astype(np.float32)
        # Detail (per-pixel spread) plus colourfulness (spread between channels).
        score = float(pixels.std() + 2.0 * pixels.mean(axis=(0, 1)).std())
        if score > best_score:
            best_image, best_score = image, score

    if best_image is None:  # pragma: no cover - sample is never empty
        raise ValueError("no usable source frames")
    return best_image


def save(figure, name: str) -> Path:
    """Write a figure into ``visuals/`` and close it."""
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    path = VISUAL_DIR / f"{name}.png"
    figure.savefig(path, bbox_inches="tight", facecolor=vs.GROUND)
    plt.close(figure)
    print(f"  wrote {path.relative_to(config.PROJECT_ROOT)}")
    return path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_artifact_catalog(source: np.ndarray, suffix: str = "") -> None:
    """Every artifact across the severity range, against the clean original."""
    columns = len(SEVERITY_STEPS) + 1
    figure, axes = plt.subplots(
        len(config.ARTIFACT_TYPES), columns,
        figsize=(1.35 * columns, 1.35 * len(config.ARTIFACT_TYPES)),
    )

    for row, artifact in enumerate(config.ARTIFACT_TYPES):
        vs.strip_axes(axes[row, 0])
        axes[row, 0].imshow(source)
        axes[row, 0].set_ylabel(
            artifact.replace("_", "\n"), fontsize=7.5, family=vs.MONO,
            color=vs.CLASS_COLORS[artifact], rotation=0, ha="right", va="center",
            labelpad=12,
        )
        if row == 0:
            axes[row, 0].set_title("clean", fontsize=8, color=vs.MUTED, family=vs.MONO)

        for col, severity in enumerate(SEVERITY_STEPS, start=1):
            rng = np.random.default_rng([11, row, int(severity * 100)])
            axes[row, col].imshow(apply_corruption(source, artifact, severity, rng))
            vs.strip_axes(axes[row, col])
            if row == 0:
                axes[row, col].set_title(
                    f"{severity:.1f}", fontsize=8, color=vs.MUTED, family=vs.MONO
                )

    figure.suptitle(
        "The six faults, from barely there to catastrophic",
        color=vs.TEXT, fontsize=12, fontweight="bold", y=0.995,
    )
    figure.text(
        0.5, 0.005, "severity  →", ha="center", color=vs.MUTED,
        fontsize=8, family=vs.MONO,
    )
    figure.tight_layout(rect=(0, 0.015, 1, 0.985))
    save(figure, f"artifact_catalog{suffix}")


def figure_channel_swap_flaw(model, source: np.ndarray) -> None:
    """Show why a full-frame channel swap cannot be learned."""
    rng = np.random.default_rng(3)
    panels = [
        ("clean frame", source, "labelled clean"),
        ("channel_swap @ 0.6", apply_corruption(source, "channel_swap", 0.6, rng),
         "labelled corrupted"),
        ("channel_swap @ 1.0", apply_corruption(source, "channel_swap", 1.0, rng),
         "labelled corrupted"),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(9, 3.6))
    for axis, (title, image, label_text) in zip(axes, panels):
        axis.imshow(image)
        vs.strip_axes(axis)
        index, _, p_corrupt = verdict(model, image)
        called = config.CLASS_NAMES[index]
        agrees = (called == "clean") == (label_text == "labelled clean")
        axis.set_title(title, fontsize=9, color=vs.TEXT, family=vs.MONO, pad=6)
        axis.set_xlabel(
            f"{label_text}\nmodel says: {called}  ({p_corrupt:.0%} corrupted)",
            fontsize=8, family=vs.MONO, labelpad=8,
            color=vs.GOOD if agrees else vs.BAD,
        )

    figure.suptitle(
        "Even at full severity the swap stops short of the frame edge",
        color=vs.TEXT, fontsize=11.5, fontweight="bold",
    )
    figure.tight_layout()
    save(figure, "channel_swap_flaw")


def figure_severity_curves(model, val_paths: list[Path], image_size: int) -> None:
    """Detection and naming accuracy as a function of fault severity."""
    detected = {a: [] for a in config.ARTIFACT_TYPES}
    named = {a: [] for a in config.ARTIFACT_TYPES}
    sources = [load_image(p, image_size) for p in val_paths]

    for severity in SEVERITY_STEPS:
        for artifact_index, artifact in enumerate(config.ARTIFACT_TYPES):
            rng = np.random.default_rng([13, artifact_index, int(severity * 100)])
            batch = torch.stack([
                to_tensor(apply_corruption(image, artifact, severity, rng))
                for image in sources
            ])
            with torch.no_grad():
                out = model(batch)
            detected[artifact].append(
                float((out.binary_logits.argmax(1) == 1).float().mean())
            )
            named[artifact].append(
                float((out.class_logits.argmax(1) == artifact_index + 1).float().mean())
            )

    # Several artifacts sit pinned at 100%; distinct dash patterns keep them
    # tellable apart where the lines coincide.
    dashes = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for axis, series, title in (
        (axes[0], detected, 'caught as "something is wrong"'),
        (axes[1], named, "fault named correctly"),
    ):
        for offset, (artifact, values) in enumerate(series.items()):
            axis.plot(
                SEVERITY_STEPS, values, marker="o", markersize=4, linewidth=1.8,
                linestyle=dashes[offset % len(dashes)],
                color=vs.CLASS_COLORS[artifact], label=artifact,
            )
        vs.label(axis, title)
        axis.set_xlabel("severity")
        axis.set_ylim(-0.06, 1.08)
        axis.set_xticks(SEVERITY_STEPS)
        axis.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    axes[0].set_ylabel("share of frames")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, fontsize=8, ncols=6, loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
    )
    figure.suptitle(
        "The single accuracy number hides where the detector goes blind",
        color=vs.TEXT, fontsize=12, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save(figure, "severity_curves")


def figure_confusion(matrix: np.ndarray) -> None:
    """Confusion matrix with row-normalised shading and raw counts."""
    names = config.CLASS_NAMES
    row_sums = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    normalised = matrix / row_sums

    figure, axes = plt.subplots(figsize=(6.8, 5.8))
    image = axes.imshow(normalised, cmap="magma", vmin=0.0, vmax=1.0)
    axes.grid(False)
    bar = figure.colorbar(image, ax=axes, fraction=0.045)
    bar.set_label("share of that true class", color=vs.MUTED, fontsize=8)
    bar.ax.tick_params(colors=vs.MUTED)
    bar.outline.set_edgecolor(vs.HAIRLINE)

    for row in range(len(names)):
        for col in range(len(names)):
            count = int(matrix[row, col])
            if count == 0:
                continue
            axes.text(
                col, row, str(count), ha="center", va="center", fontsize=8,
                family=vs.MONO,
                color="#0A0D12" if normalised[row, col] > 0.55 else "#E8EDF3",
            )

    axes.set_xticks(range(len(names)), names, rotation=42, ha="right", fontsize=8)
    axes.set_yticks(range(len(names)), names, fontsize=8)
    for tick, name in zip(axes.get_yticklabels(), names):
        tick.set_color(vs.CLASS_COLORS[name])
    axes.set_xlabel("what the model said")
    axes.set_ylabel("what it actually was")
    axes.set_title(
        "Correct answers run down the diagonal",
        color=vs.TEXT, fontsize=11.5, pad=12,
    )
    figure.tight_layout()
    save(figure, "confusion")


def figure_class_balance(counts: np.ndarray) -> None:
    """How often each class appears in training — the source of the clean bias."""
    names = config.CLASS_NAMES
    share = counts / counts.sum()
    even = 1.0 / len(names)

    figure, axes = plt.subplots(figsize=(8, 3.4))
    bars = axes.bar(
        range(len(names)), share,
        color=[vs.CLASS_COLORS[n] for n in names], width=0.66,
    )
    axes.axhline(even, color=vs.TEXT, linestyle="--", linewidth=1, alpha=0.7)
    axes.text(
        len(names) - 0.4, even + 0.008, "even split", color=vs.TEXT,
        fontsize=8, family=vs.MONO, ha="right",
    )
    for rect, value in zip(bars, share):
        axes.text(
            rect.get_x() + rect.get_width() / 2, value + 0.006, f"{value:.0%}",
            ha="center", fontsize=8, family=vs.MONO, color=vs.TEXT,
        )

    axes.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
    axes.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes.set_ylabel("share of training frames")
    ratio = share[0] / share[1:].mean()
    vs.label(axes, f"'clean' shows up {ratio:.1f}x as often as any single fault")
    figure.tight_layout()
    save(figure, "class_balance")


def figure_training_curves(history: list[dict]) -> None:
    """Loss and accuracy per epoch, with the still-improving trend called out."""
    epochs = [entry["epoch"] for entry in history]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    axes[0].plot(epochs, [e["train"]["loss"] for e in history],
                 marker="o", markersize=4, color=vs.MUTED, label="training")
    axes[0].plot(epochs, [e["val"]["loss"] for e in history],
                 marker="o", markersize=4, color=vs.ACCENT, label="held-out")
    vs.label(axes[0], "how wrong it is (lower is better)")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=8)

    axes[1].plot(epochs, [e["val"]["binary_accuracy"] for e in history],
                 marker="o", markersize=4, color=vs.GOOD, label="is it broken?")
    axes[1].plot(epochs, [e["val"]["class_accuracy"] for e in history],
                 marker="o", markersize=4, color=vs.CLASS_COLORS["channel_swap"],
                 label="which fault?")
    vs.label(axes[1], "accuracy on frames it never trained on")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0.3, 1.0)
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[1].legend(fontsize=8, loc="lower right")

    # Whether the run finished is a property of the curve, not of the caption:
    # compare the last few epochs' gain against a "has clearly flattened" floor.
    accuracies = [entry["val"]["class_accuracy"] for entry in history]
    tail = accuracies[-min(4, len(accuracies)):]
    still_climbing = (tail[-1] - tail[0]) > PLATEAU_TOLERANCE
    note = ("still climbing when\ntraining stopped" if still_climbing
            else "flattened — training\nran to completion")
    axes[1].annotate(
        note,
        xy=(epochs[-1], accuracies[-1]),
        xytext=(epochs[-1] - len(epochs) * 0.42, min(tail) - 0.28),
        fontsize=8, family=vs.MONO, color=vs.ACCENT,
        arrowprops=dict(arrowstyle="->", color=vs.ACCENT, linewidth=1),
    )

    figure.suptitle(
        f"{len(epochs)} epochs — "
        + ("the curve had not flattened" if still_climbing
           else f"finished at {accuracies[-1]:.1%} on held-out frames"),
        color=vs.TEXT, fontsize=12, fontweight="bold",
    )
    figure.tight_layout()
    save(figure, "training_curves")


def figure_dataset_sheet(paths: list[Path], image_size: int) -> None:
    """Contact sheet of the source frames, labelled by which game they came from."""
    columns, rows = 6, 4
    step = max(1, len(paths) // (columns * rows))
    picks = paths[::step][:columns * rows]

    figure, axes = plt.subplots(rows, columns, figsize=(2.0 * columns, 1.55 * rows))
    for slot in range(rows * columns):
        axis = axes[slot // columns, slot % columns]
        vs.strip_axes(axis)
        if slot >= len(picks):
            axis.set_visible(False)
            continue
        axis.imshow(load_image(picks[slot], image_size))
        # Filenames are "<game>_<index>.png"; the stem before the last underscore
        # is the title the frame was captured from.
        axis.set_title(
            picks[slot].stem.rsplit("_", 1)[0].replace("_", " "),
            fontsize=7.5, family=vs.MONO, color=vs.MUTED, pad=3,
        )

    figure.suptitle(
        f"The new source frames — {len(paths)} real game captures",
        color=vs.TEXT, fontsize=12, fontweight="bold",
    )
    figure.tight_layout()
    save(figure, "dataset_sheet")


def figure_predictions(model, val_paths: list[Path], image_size: int) -> None:
    """A gallery of real verdicts, hits in green and misses in red."""
    rng = np.random.default_rng(21)
    total = GALLERY_COLUMNS * GALLERY_ROWS
    figure, axes = plt.subplots(
        GALLERY_ROWS, GALLERY_COLUMNS,
        figsize=(2.05 * GALLERY_COLUMNS, 3.05 * GALLERY_ROWS),
    )

    # Frame folders sort by title, so consecutive paths are all the same game.
    # Stride across the whole held-out set to get a gallery worth looking at.
    step = max(1, len(val_paths) // total)
    picks = (val_paths[::step] * total)[:total]

    for slot in range(total):
        axis = axes[slot // GALLERY_COLUMNS, slot % GALLERY_COLUMNS]
        source = load_image(picks[slot], image_size)

        if rng.random() < 0.3:
            image, truth = source, "clean"
        else:
            truth = config.ARTIFACT_TYPES[
                int(rng.integers(0, len(config.ARTIFACT_TYPES)))
            ]
            image = apply_corruption(source, truth, float(rng.uniform(0.2, 1.0)), rng)

        index, confidence, _ = verdict(model, image)
        called = config.CLASS_NAMES[index]
        correct = called == truth

        axis.imshow(image)
        vs.strip_axes(axis)
        for spine in axis.spines.values():
            spine.set_color(vs.GOOD if correct else vs.BAD)
            spine.set_linewidth(1.6)
        axis.set_title(
            f"is: {truth}", fontsize=7.5, family=vs.MONO, color=vs.MUTED, pad=4
        )
        axis.set_xlabel(
            f"said: {called}\n{confidence:.0%} sure", fontsize=7.5, family=vs.MONO,
            color=vs.GOOD if correct else vs.BAD, labelpad=6,
        )

    figure.suptitle(
        "Actual verdicts on held-out frames",
        color=vs.TEXT, fontsize=12, fontweight="bold",
    )
    # h_pad keeps a row's two-line caption clear of the next row's title.
    figure.tight_layout(h_pad=3.2)
    save(figure, "predictions")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
FIGURE_NAMES: tuple[str, ...] = (
    "catalog", "flaw", "curves", "confusion", "balance", "training", "predictions",
    "dataset",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the visuals CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--only", nargs="+", choices=FIGURE_NAMES, default=None,
                        help="render a subset of the figures")
    parser.add_argument("--source-dir", type=Path, default=None,
                        help="draw the catalog and contact sheet from this frame "
                             "folder instead of the checkpoint's own clean_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render the requested figures; returns a process exit code."""
    from src.dataset import CorruptionDataset

    args = build_parser().parse_args(argv)
    wanted = set(args.only or FIGURE_NAMES)
    vs.apply_style()

    try:
        model, saved, val_paths, checkpoint = load_context(
            Path(args.checkpoint), torch.device("cpu")
        )
    except (FileNotFoundError, KeyError) as error:
        print(f"error: {error}")
        return 2

    image_size = saved["image_size"]
    source_paths = (
        list_clean_images(args.source_dir) if args.source_dir else val_paths
    )
    source = pick_showcase_frame(source_paths, image_size)
    print(f"rendering into {VISUAL_DIR} from {len(source_paths)} source frames")

    if "catalog" in wanted:
        # A catalog drawn from a different frame folder is a second figure, not a
        # replacement: the report shows the drawn and the real one side by side.
        figure_artifact_catalog(source, "_real" if args.source_dir else "")
    if "flaw" in wanted:
        figure_channel_swap_flaw(model, source)
    if "curves" in wanted:
        figure_severity_curves(model, val_paths, image_size)
    if "confusion" in wanted:
        import json
        metrics_path = Path(args.checkpoint).parent / config.METRICS_FILENAME
        matrix = np.array(json.loads(metrics_path.read_text())["confusion_matrix"])
        figure_confusion(matrix)
    if "balance" in wanted:
        train_paths, _ = split_paths(
            list_clean_images(saved.get("clean_dir", config.CLEAN_DIR)),
            saved["val_fraction"], saved["seed"],
        )
        dataset = CorruptionDataset(
            train_paths, image_size=64, samples_per_image=20,
            clean_ratio=saved["clean_ratio"], seed=5, deterministic=True,
        )
        counts = np.zeros(config.NUM_CLASSES, dtype=int)
        for index in range(len(dataset)):
            counts[int(dataset[index][2])] += 1
        figure_class_balance(counts)
    if "training" in wanted:
        import json
        history_path = Path(args.checkpoint).parent / "history.json"
        if history_path.is_file():
            figure_training_curves(json.loads(history_path.read_text()))
        else:
            print(f"  skipped training curves: no {history_path}")
    if "dataset" in wanted:
        figure_dataset_sheet(source_paths, image_size)
    if "predictions" in wanted:
        figure_predictions(model, val_paths, image_size)

    print(f"done — {checkpoint.get('epoch', '?')}-epoch checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
