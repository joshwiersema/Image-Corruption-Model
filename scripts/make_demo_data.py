"""Generate synthetic 'clean game frames' so the pipeline runs with no downloads.

Real training data would be captured render output.  For a self-contained demo
we procedurally draw frames that carry the structures the detector must not
mistake for artifacts:

* a sky/ground gradient — smooth low-frequency content;
* checkerboard floors and brick walls — hard, regular edges that superficially
  resemble block corruption;
* horizontal strata and railings — strong horizontal structure that a tearing
  detector must not fire on;
* silhouettes, discs and a HUD bar — solid shapes with saturated colours;
* light film grain — so "any noise at all" is not a giveaway for shader noise.

Each scene is drawn from a seeded generator, so re-running produces the same
frames.

Usage::

    python scripts/make_demo_data.py                 # 12 frames at 256x256
    python scripts/make_demo_data.py --count 40 --size 320 --preview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Allow `python scripts/make_demo_data.py` as well as `-m scripts.make_demo_data`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402  (import after sys.path fix)

DEFAULT_COUNT: int = 12
GRAIN_SIGMA: float = 4.0        # film grain std-dev in 8-bit levels
HUD_HEIGHT_FRAC: float = 0.04
PREVIEW_FILENAME: str = "demo_preview.png"


def _gradient_sky(size: int, rng: np.random.Generator) -> np.ndarray:
    """Vertical two-colour gradient standing in for sky over ground."""
    top = rng.integers(40, 200, 3).astype(np.float32)
    bottom = rng.integers(10, 140, 3).astype(np.float32)
    ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    column = top[None, :] * (1.0 - ramp) + bottom[None, :] * ramp
    return np.repeat(column[:, None, :], size, axis=1)


def _draw_checkerboard(canvas: np.ndarray, rng: np.random.Generator) -> None:
    """Draw a perspective-ish checkerboard floor across the lower frame."""
    size = canvas.shape[0]
    horizon = int(size * rng.uniform(0.45, 0.65))
    tile = int(rng.integers(8, 28))
    light = rng.integers(120, 230, 3).astype(np.float32)
    dark = rng.integers(20, 110, 3).astype(np.float32)

    rows = np.arange(horizon, size)[:, None]
    cols = np.arange(size)[None, :]
    parity = ((rows // tile) + (cols // tile)) % 2 == 0
    floor = np.where(parity[..., None], light, dark)
    canvas[horizon:] = floor


def _draw_wall_bricks(canvas: np.ndarray, rng: np.random.Generator) -> None:
    """Draw an offset brick wall — regular hard edges, like block artifacts."""
    size = canvas.shape[0]
    height = int(size * rng.uniform(0.25, 0.5))
    brick_h = int(rng.integers(6, 18))
    brick_w = int(rng.integers(14, 40))
    base = rng.integers(60, 180, 3).astype(np.float32)

    for row in range(0, height, brick_h):
        offset = (row // brick_h % 2) * (brick_w // 2)
        for col in range(-offset, size, brick_w):
            shade = float(rng.uniform(0.75, 1.25))
            left, right = max(0, col), min(size, col + brick_w - 1)
            if left >= right:
                continue
            canvas[row:row + brick_h - 1, left:right] = np.clip(base * shade, 0, 255)


def _draw_strata(canvas: np.ndarray, rng: np.random.Generator) -> None:
    """Draw horizontal bands — legitimate horizontal structure."""
    size = canvas.shape[0]
    row = 0
    while row < size:
        band = int(rng.integers(3, 20))
        colour = rng.integers(20, 235, 3).astype(np.float32)
        canvas[row:row + band] = colour
        row += band


def _draw_shapes(canvas: np.ndarray, rng: np.random.Generator) -> None:
    """Overlay a few solid discs and rectangles as scene props."""
    size = canvas.shape[0]
    rows, cols = np.mgrid[0:size, 0:size]

    for _ in range(int(rng.integers(2, 7))):
        colour = rng.integers(0, 256, 3).astype(np.float32)
        if rng.random() < 0.5:                      # disc
            centre_r = int(rng.integers(0, size))
            centre_c = int(rng.integers(0, size))
            radius = int(rng.integers(size // 16, size // 4))
            mask = (rows - centre_r) ** 2 + (cols - centre_c) ** 2 <= radius ** 2
            canvas[mask] = colour
        else:                                        # rectangle
            height = int(rng.integers(size // 12, size // 3))
            width = int(rng.integers(size // 12, size // 3))
            row = int(rng.integers(0, max(1, size - height)))
            col = int(rng.integers(0, max(1, size - width)))
            canvas[row:row + height, col:col + width] = colour


def _draw_hud(canvas: np.ndarray, rng: np.random.Generator) -> None:
    """Draw a HUD strip with segment ticks along the bottom edge."""
    size = canvas.shape[0]
    bar_h = max(2, int(size * HUD_HEIGHT_FRAC))
    top = size - bar_h * 2
    canvas[top:top + bar_h] = np.array([25, 25, 30], dtype=np.float32)

    fill = float(rng.uniform(0.15, 0.95))
    fill_colour = rng.integers(120, 256, 3).astype(np.float32)
    canvas[top + 1:top + bar_h - 1, 2:max(3, int(size * fill))] = fill_colour


def make_frame(size: int, rng: np.random.Generator) -> np.ndarray:
    """Compose one synthetic clean frame as a ``uint8`` RGB array."""
    canvas = _gradient_sky(size, rng)

    # Pick one dominant background structure per frame so the set stays varied.
    structure = int(rng.integers(0, 3))
    if structure == 0:
        _draw_checkerboard(canvas, rng)
    elif structure == 1:
        _draw_wall_bricks(canvas, rng)
    else:
        _draw_strata(canvas, rng)

    _draw_shapes(canvas, rng)
    _draw_hud(canvas, rng)

    # Light grain: without it, "the image has noise" would trivially separate
    # clean frames from shader_noise ones.
    canvas += rng.normal(0.0, GRAIN_SIGMA, canvas.shape)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def write_preview(frames: list[np.ndarray], path: Path) -> None:
    """Write a contact sheet of the generated frames for a quick eyeball check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = min(4, len(frames))
    rows = (len(frames) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
    flat = np.atleast_1d(np.asarray(axes)).ravel()

    for index, axis in enumerate(flat):
        axis.axis("off")
        if index < len(frames):
            axis.imshow(frames[index])
            axis.set_title(f"frame_{index:03d}", fontsize=8)

    figure.suptitle("Synthetic clean demo frames")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Build the demo-data CLI."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic stand-in frames into data/clean.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help="number of frames to generate")
    parser.add_argument("--size", type=int, default=config.IMAGE_SIZE,
                        help="square frame resolution in pixels")
    parser.add_argument("--output-dir", type=Path, default=config.DEMO_CLEAN_DIR,
                        help="destination directory for the PNG frames")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--preview", action="store_true",
                        help=f"also write a {PREVIEW_FILENAME} contact sheet")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate frames even if the directory is populated")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the demo frames; returns a process exit code."""
    args = build_parser().parse_args(argv)
    if args.count < 1:
        raise ValueError(f"--count must be >= 1, got {args.count}")
    if args.size < 32:
        raise ValueError(f"--size must be >= 32, got {args.size}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("frame_*.png"))
    if existing and not args.overwrite:
        print(
            f"{len(existing)} frames already in {output_dir}; "
            "pass --overwrite to regenerate."
        )
        return 0

    rng = np.random.default_rng(args.seed)
    frames = []
    for index in range(args.count):
        frame = make_frame(args.size, rng)
        Image.fromarray(frame).save(output_dir / f"frame_{index:03d}.png")
        frames.append(frame)

    print(f"wrote {len(frames)} {args.size}x{args.size} frames to {output_dir}")

    if args.preview:
        preview_path = output_dir.parent / PREVIEW_FILENAME
        write_preview(frames, preview_path)
        print(f"wrote preview contact sheet to {preview_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
