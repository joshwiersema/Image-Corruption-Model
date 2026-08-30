"""Watch the detector work, one frame at a time.

Opens a window showing a stream of frames — some clean, some corrupted — with
the model's verdict updating live beside each one: the confidence bar for every
class, the "is it broken" probability, and how long the decision took.

Usage::

    python scripts/live_view.py                       # live window
    python scripts/live_view.py --interval 0.25       # faster stream
    python scripts/live_view.py --gif visuals/live_demo.gif --frames 36

``--gif`` records the same stream to an animated GIF instead of opening a
window, which is how you get a shareable version of the demo.
"""

from __future__ import annotations

import argparse
import sys
import time
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

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

DEFAULT_INTERVAL: float = 0.6
DEFAULT_FRAMES: int = 40
GIF_FPS: int = 2
GIF_DPI: int = 88


class FrameStream:
    """Yields ``(image, true_class_index)`` pairs from the held-out frames."""

    def __init__(
        self,
        paths: list[Path],
        image_size: int,
        clean_ratio: float,
        severity_range: tuple[float, float],
        seed: int,
    ) -> None:
        if not paths:
            raise ValueError("FrameStream requires at least one source frame")
        self.sources = [load_image(path, image_size) for path in paths]
        self.clean_ratio = clean_ratio
        self.severity_range = severity_range
        self.rng = np.random.default_rng(seed)
        self.severity = 0.0

    def next_frame(self) -> tuple[np.ndarray, int]:
        """Draw the next frame, corrupting it most of the time."""
        source = self.sources[int(self.rng.integers(0, len(self.sources)))]
        if self.rng.random() < self.clean_ratio:
            self.severity = 0.0
            return source, config.CLEAN_CLASS_INDEX

        artifact_index = int(self.rng.integers(0, len(config.ARTIFACT_TYPES)))
        low, high = self.severity_range
        self.severity = float(self.rng.uniform(low, high))
        corrupted = apply_corruption(
            source, config.ARTIFACT_TYPES[artifact_index], self.severity, self.rng
        )
        return corrupted, artifact_index + 1


class LiveView:
    """The two-panel display: the frame on the left, the verdict on the right."""

    def __init__(self, model, stream: FrameStream) -> None:
        self.model = model
        self.stream = stream

        self.figure, (self.frame_axis, self.bar_axis) = plt.subplots(
            1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 1.15]}
        )
        self.figure.patch.set_facecolor(vs.GROUND)

        blank = np.zeros((8, 8, 3), dtype=np.uint8)
        self.image_artist = self.frame_axis.imshow(blank)
        vs.strip_axes(self.frame_axis)

        positions = np.arange(config.NUM_CLASSES)
        self.bars = self.bar_axis.barh(
            positions, np.zeros(config.NUM_CLASSES),
            color=[vs.CLASS_COLORS[n] for n in config.CLASS_NAMES], height=0.66,
        )
        self.bar_axis.set_xlim(0, 1)
        self.bar_axis.set_yticks(positions, config.CLASS_NAMES, fontsize=9)
        for tick, name in zip(self.bar_axis.get_yticklabels(), config.CLASS_NAMES):
            tick.set_color(vs.CLASS_COLORS[name])
        self.bar_axis.invert_yaxis()
        self.bar_axis.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        vs.label(self.bar_axis, "how sure it is of each answer")

        self.verdict_text = self.figure.text(
            0.5, 0.955, "", ha="center", fontsize=13, fontweight="bold",
            color=vs.TEXT, family=vs.MONO,
        )
        self.detail_text = self.figure.text(
            0.5, 0.045, "", ha="center", fontsize=9, color=vs.MUTED, family=vs.MONO,
        )
        self.figure.tight_layout(rect=(0, 0.08, 1, 0.92))

    def step(self, _frame_number: int):
        """Advance one frame: corrupt, classify, redraw."""
        image, truth = self.stream.next_frame()

        started = time.perf_counter()
        with torch.no_grad():
            output = self.model(to_tensor(image).unsqueeze(0))
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        probabilities = output.class_logits.softmax(1)[0].numpy()
        p_corrupt = float(output.binary_logits.softmax(1)[0, 1])
        predicted = int(probabilities.argmax())
        correct = predicted == truth

        self.image_artist.set_data(image)
        for bar, value in zip(self.bars, probabilities):
            bar.set_width(float(value))

        for spine in self.frame_axis.spines.values():
            spine.set_color(vs.GOOD if correct else vs.BAD)
            spine.set_linewidth(2.2)

        truth_name = config.CLASS_NAMES[truth]
        said = config.CLASS_NAMES[predicted]
        mark = "correct" if correct else "wrong"
        self.verdict_text.set_text(f"{truth_name}  ->  said {said}   [{mark}]")
        self.verdict_text.set_color(vs.GOOD if correct else vs.BAD)

        severity = "clean frame" if truth == config.CLEAN_CLASS_INDEX else (
            f"severity {self.stream.severity:.2f}"
        )
        self.detail_text.set_text(
            f"{severity}   |   broken: {p_corrupt:.0%}   |   decided in {elapsed_ms:.1f} ms"
        )
        return [self.image_artist, self.verdict_text, self.detail_text, *self.bars]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the live-view CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help="seconds between frames")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help="frames to record when writing a GIF")
    parser.add_argument("--gif", type=Path, default=None,
                        help="record to this GIF instead of opening a window")
    parser.add_argument("--clean-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the live viewer or record a GIF; returns a process exit code."""
    args = build_parser().parse_args(argv)
    if args.interval <= 0:
        print(f"error: --interval must be > 0, got {args.interval}")
        return 2

    if args.gif is not None:
        matplotlib.use("Agg")
    vs.apply_style()

    try:
        model, checkpoint = load_checkpoint(Path(args.checkpoint), torch.device("cpu"))
    except (FileNotFoundError, KeyError) as error:
        print(f"error: {error}")
        return 2

    saved = checkpoint["args"]
    _, val_paths = split_paths(
        list_clean_images(saved.get("clean_dir", config.CLEAN_DIR)),
        saved["val_fraction"], saved["seed"],
    )
    stream = FrameStream(
        val_paths, saved["image_size"], args.clean_ratio,
        (saved["severity_min"], saved["severity_max"]), args.seed,
    )
    view = LiveView(model, stream)

    if args.gif is not None:
        # A GIF is played back small and every frame is stored whole, so render
        # at a lower dpi than the interactive window; it roughly halves the file.
        view.figure.set_dpi(GIF_DPI)
        animation = FuncAnimation(
            view.figure, view.step, frames=args.frames, interval=1000 // GIF_FPS,
            blit=False, repeat=False,
        )
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        animation.save(
            args.gif, writer=PillowWriter(fps=GIF_FPS),
            savefig_kwargs={"facecolor": vs.GROUND},
        )
        print(f"wrote {args.gif}")
        return 0

    print("streaming — close the window to stop")
    animation = FuncAnimation(  # noqa: F841 (must stay referenced while showing)
        view.figure, view.step, interval=int(args.interval * 1000), blit=False,
        cache_frame_data=False,
    )
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
