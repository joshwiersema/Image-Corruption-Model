"""Shared plot styling so every figure in ``visuals/`` reads as one set.

The palette is dark-ground on purpose: these figures are read next to frame
captures, and a dark surround stops the eye from re-white-balancing between
the plot and the image content it describes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: never needs a display
import matplotlib.pyplot as plt

from src import config

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
GROUND: str = "#0A0D12"
PANEL: str = "#131922"
HAIRLINE: str = "#2A3441"
TEXT: str = "#CBD5E1"
MUTED: str = "#78889B"
ACCENT: str = "#E8A33D"
GOOD: str = "#5FBF8F"
BAD: str = "#E2564D"

#: One colour per class, index-aligned with :data:`src.config.CLASS_NAMES`.
CLASS_COLORS: dict[str, str] = {
    "clean": "#78889B",
    "block_corruption": "#E2564D",
    "horizontal_tear": "#E8A33D",
    "shader_noise": "#5FBF8F",
    "stuck_pixels": "#58A6E8",
    "channel_swap": "#B98BE0",
    "texture_smear": "#E07BA8",
}

# Fail loudly if the taxonomy grows without the palette following it.
if set(CLASS_COLORS) != set(config.CLASS_NAMES):
    raise RuntimeError(
        f"CLASS_COLORS must cover every class name; got {sorted(CLASS_COLORS)} "
        f"vs {sorted(config.CLASS_NAMES)}"
    )

MONO: list[str] = ["DejaVu Sans Mono", "Consolas", "monospace"]
SANS: list[str] = ["DejaVu Sans", "Segoe UI", "sans-serif"]


def apply_style() -> None:
    """Install the shared rcParams. Call once before building figures."""
    plt.rcParams.update({
        "figure.facecolor": GROUND,
        "savefig.facecolor": GROUND,
        "axes.facecolor": PANEL,
        "axes.edgecolor": HAIRLINE,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": HAIRLINE,
        "grid.alpha": 0.55,
        "grid.linewidth": 0.7,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.family": SANS,
        "font.size": 9,
        "legend.frameon": False,
        "legend.labelcolor": TEXT,
        "figure.dpi": 130,
    })


def label(axes, text: str, *, size: int = 8) -> None:
    """Set a small uppercase mono eyebrow as the axes title."""
    axes.set_title(
        text.upper(), fontsize=size, color=MUTED, family=MONO,
        loc="left", pad=8, fontweight="normal",
    )


def strip_axes(axes) -> None:
    """Turn an axes into a bare image frame."""
    axes.set_xticks([])
    axes.set_yticks([])
    axes.grid(False)
    for spine in axes.spines.values():
        spine.set_color(HAIRLINE)
