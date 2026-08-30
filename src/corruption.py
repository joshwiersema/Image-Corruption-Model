"""Synthetic GPU render-artifact injection.

Each public ``apply_*`` function takes a *clean* frame and returns a **new**
corrupted frame — inputs are never mutated.  Every function shares the same
signature::

    apply_x(image: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray

``image``
    ``uint8`` array of shape ``(H, W, 3)`` in RGB order.
``severity``
    Scalar in ``[0, 1]``.  ``0`` is a barely perceptible artifact, ``1`` is a
    severe one.  Severity is validated at the boundary and clamped-checked, not
    silently coerced.
``rng``
    ``numpy`` generator, so that every sample is reproducible from a seed.

The artifacts model failure signatures seen on real GPUs:

``block_corruption``
    Tile/wavefront-sized regions written with garbage — the classic symptom of
    a bad memory write or a compression block decoded incorrectly.
``horizontal_tear``
    Rows displaced horizontally (scanout desync / missing vsync) plus, at
    higher severities, a vertical tear seam where two frames are stitched.
``shader_noise``
    High-frequency per-pixel noise and speckle from an unstable shader core or
    a denoiser fed NaNs.
``stuck_dead_pixels``
    Individual pixels latched to a constant colour (stuck) or to black (dead),
    as caused by defective framebuffer cells.
``channel_swap``
    Colour channels permuted or partially crossed — a wrong surface format or
    swizzle descriptor.
``texture_smear``
    Directional streaking from a texture fetch returning a stale/oversized mip
    or an unresolved sampler.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .config import ARTIFACT_TYPES

# ---------------------------------------------------------------------------
# Tunable constants for the injectors.  Kept here so no magic numbers hide in
# the functions; each is expressed as the value reached at severity == 1.0.
# ---------------------------------------------------------------------------
MIN_SEVERITY: float = 0.0
MAX_SEVERITY: float = 1.0

# block_corruption
BLOCK_SIZES: tuple[int, ...] = (8, 16, 32, 64)
BLOCK_MAX_COUNT: int = 40
BLOCK_MIN_COUNT: int = 1

# horizontal_tear
TEAR_MAX_BANDS: int = 24
TEAR_MIN_BANDS: int = 1
TEAR_MAX_BAND_HEIGHT_FRAC: float = 0.12
TEAR_MAX_SHIFT_FRAC: float = 0.35
TEAR_SEAM_SEVERITY: float = 0.5      # above this, add a full-frame tear seam

# shader_noise
NOISE_MAX_SIGMA: float = 90.0        # gaussian sigma in 8-bit levels
NOISE_MAX_SPECKLE_FRAC: float = 0.06  # fraction of pixels blown to extremes

# stuck_dead_pixels
PIXEL_MAX_FRAC: float = 0.02         # fraction of pixels affected
PIXEL_DEAD_RATIO: float = 0.5        # of the affected pixels, share set black
STUCK_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255),
)

# channel_swap
CHANNEL_PERMUTATIONS: tuple[tuple[int, int, int], ...] = (
    (2, 1, 0),  # RGB -> BGR
    (1, 0, 2),  # RGB -> GRB
    (0, 2, 1),  # RGB -> RBG
    (1, 2, 0),  # RGB -> GBR
    (2, 0, 1),  # RGB -> BRG
)
CHANNEL_MIN_REGION_FRAC: float = 0.25  # smallest affected region at severity 0
# Largest affected region, at severity 1.0.  Deliberately below 1.0: a swap that
# covers the whole frame leaves no boundary and no unswapped reference, so it is
# indistinguishable from a clean frame that simply used other colours.  Such a
# sample is labelled "corrupted" while being identical in kind to a clean one,
# which is unlearnable by construction rather than merely hard.  Detecting a
# whole-surface swizzle fault needs a reference the detector does not have from a
# single frame — a previous frame, or a model of expected scene colour.
CHANNEL_MAX_REGION_FRAC: float = 0.85

# texture_smear
SMEAR_MAX_LENGTH_FRAC: float = 0.25   # trail length as a fraction of the frame
SMEAR_MIN_LENGTH: int = 2
SMEAR_MAX_PATCHES: int = 6
SMEAR_MIN_PATCHES: int = 1
SMEAR_PATCH_MIN_FRAC: float = 0.15
SMEAR_PATCH_MAX_FRAC: float = 0.55


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate(image: np.ndarray, severity: float) -> None:
    """Validate injector input at the boundary; raise rather than guess."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {image.dtype}")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError(f"image must be at least 2x2, got {image.shape[:2]}")
    if not np.isfinite(severity):
        raise ValueError("severity must be a finite number")
    if not MIN_SEVERITY <= severity <= MAX_SEVERITY:
        raise ValueError(
            f"severity must be in [{MIN_SEVERITY}, {MAX_SEVERITY}], got {severity}"
        )


def _lerp_int(low: int, high: int, severity: float) -> int:
    """Interpolate an integer count between ``low`` and ``high`` by severity."""
    return int(round(low + (high - low) * severity))


def _rand_span(rng: np.random.Generator, limit: int, length: int) -> int:
    """Random start offset so that ``[start, start + length)`` fits in ``limit``."""
    return int(rng.integers(0, max(1, limit - length + 1)))


# ---------------------------------------------------------------------------
# Artifact injectors
# ---------------------------------------------------------------------------
def apply_block_corruption(
    image: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Overwrite tile-aligned blocks with garbage data.

    Models a bad memory write or a DCC/compression block decoded with the wrong
    key: rectangular tiles of the framebuffer hold content unrelated to the
    scene.  Blocks are filled with one of three plausible garbage patterns —
    uniform noise, a flat wrong colour, or a copy of an unrelated tile.
    """
    _validate(image, severity)
    out = image.copy()
    height, width = out.shape[:2]

    n_blocks = _lerp_int(BLOCK_MIN_COUNT, BLOCK_MAX_COUNT, severity)
    usable_sizes = [s for s in BLOCK_SIZES if s <= min(height, width)] or [
        max(2, min(height, width) // 2)
    ]

    for _ in range(n_blocks):
        size = int(rng.choice(usable_sizes))
        block_h = min(size, height)
        block_w = min(size, width)
        row = _rand_span(rng, height, block_h)
        col = _rand_span(rng, width, block_w)

        mode = int(rng.integers(0, 3))
        if mode == 0:                                    # uniform noise garbage
            patch = rng.integers(0, 256, (block_h, block_w, 3), dtype=np.uint8)
        elif mode == 1:                                  # flat wrong colour
            colour = rng.integers(0, 256, 3, dtype=np.uint8)
            patch = np.broadcast_to(colour, (block_h, block_w, 3)).copy()
        else:                                            # stale tile from elsewhere
            src_row = _rand_span(rng, height, block_h)
            src_col = _rand_span(rng, width, block_w)
            patch = out[src_row:src_row + block_h, src_col:src_col + block_w].copy()

        out[row:row + block_h, col:col + block_w] = patch

    return out


def apply_horizontal_tear(
    image: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Displace horizontal bands of rows and, when severe, add a tear seam.

    Models scanout desynchronisation: bands of scanlines are shifted sideways
    (line shift), and above ``TEAR_SEAM_SEVERITY`` the frame is split at a
    random row with the lower half rolled, reproducing the seam seen when two
    frames are presented without vsync.
    """
    _validate(image, severity)
    out = image.copy()
    height, width = out.shape[:2]

    n_bands = _lerp_int(TEAR_MIN_BANDS, TEAR_MAX_BANDS, severity)
    max_band_h = max(1, int(height * TEAR_MAX_BAND_HEIGHT_FRAC))
    max_shift = max(1, int(width * TEAR_MAX_SHIFT_FRAC * severity))

    for _ in range(n_bands):
        band_h = int(rng.integers(1, max_band_h + 1))
        row = _rand_span(rng, height, band_h)
        shift = int(rng.integers(-max_shift, max_shift + 1))
        if shift == 0:
            continue
        out[row:row + band_h] = np.roll(out[row:row + band_h], shift, axis=1)

    if severity > TEAR_SEAM_SEVERITY and height > 2:
        seam = int(rng.integers(1, height - 1))
        seam_shift = int(rng.integers(1, max(2, max_shift + 1)))
        out[seam:] = np.roll(out[seam:], seam_shift, axis=1)

    return out


def apply_shader_noise(
    image: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Add high-frequency gaussian noise plus extreme speckle.

    Models an unstable shader core or a post-process stage fed invalid values:
    a dense grain over the whole frame, with a fraction of pixels driven to
    pure black or white.
    """
    _validate(image, severity)
    height, width = image.shape[:2]

    sigma = NOISE_MAX_SIGMA * severity
    noise = rng.normal(0.0, sigma, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)

    speckle_frac = NOISE_MAX_SPECKLE_FRAC * severity
    n_speckle = int(height * width * speckle_frac)
    if n_speckle > 0:
        rows = rng.integers(0, height, n_speckle)
        cols = rng.integers(0, width, n_speckle)
        # Half the speckle blows out to white, half crushes to black.
        extremes = rng.integers(0, 2, (n_speckle, 1)) * 255.0
        noisy[rows, cols] = extremes

    return noisy.astype(np.uint8)


def apply_stuck_dead_pixels(
    image: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Latch scattered pixels to a constant colour (stuck) or to black (dead).

    Models defective framebuffer / display cells.  Unlike ``shader_noise`` the
    affected pixels are sparse and take saturated values, which gives the
    network a distinguishable sparse-impulse signature.
    """
    _validate(image, severity)
    out = image.copy()
    height, width = out.shape[:2]

    n_pixels = int(height * width * PIXEL_MAX_FRAC * severity)
    if n_pixels == 0:
        return out

    rows = rng.integers(0, height, n_pixels)
    cols = rng.integers(0, width, n_pixels)

    n_dead = int(n_pixels * PIXEL_DEAD_RATIO)
    out[rows[:n_dead], cols[:n_dead]] = 0                      # dead: black

    stuck_rows, stuck_cols = rows[n_dead:], cols[n_dead:]
    if stuck_rows.size:
        palette = np.array(STUCK_COLORS, dtype=np.uint8)
        picks = rng.integers(0, len(palette), stuck_rows.size)
        out[stuck_rows, stuck_cols] = palette[picks]           # stuck: saturated

    return out


def apply_channel_swap(
    image: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Permute colour channels over a region of the frame.

    Models a wrong surface format or swizzle descriptor (e.g. RGBA read as
    BGRA).  Severity controls how much of the frame is affected: a small patch
    at low severity, up to :data:`CHANNEL_MAX_REGION_FRAC` of each axis at
    ``severity == 1``.  The region never covers the frame entirely, so some
    correctly-coloured content always remains for the swap to be seen against.
    """
    _validate(image, severity)
    out = image.copy()
    height, width = out.shape[:2]

    permutation = CHANNEL_PERMUTATIONS[int(rng.integers(0, len(CHANNEL_PERMUTATIONS)))]

    span = CHANNEL_MAX_REGION_FRAC - CHANNEL_MIN_REGION_FRAC
    region_frac = CHANNEL_MIN_REGION_FRAC + span * severity

    region_h = max(1, int(height * region_frac))
    region_w = max(1, int(width * region_frac))
    row = _rand_span(rng, height, region_h)
    col = _rand_span(rng, width, region_w)

    region = out[row:row + region_h, col:col + region_w]
    out[row:row + region_h, col:col + region_w] = region[:, :, permutation]
    return out


def apply_texture_smear(
    image: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Streak patches of the frame along one axis.

    Models a texture fetch returning a stale sample or an unresolved sampler.
    Two effects compose, both scaled by severity:

    1. **Hold/stretch** — the patch is subsampled every ``stride`` lines and
       each surviving line is repeated, so content is dragged into hard bands.
       This is the dominant, visually obvious part of the artifact.
    2. **Decaying trail** — a max-blend of shifted copies adds a fading streak
       tail off the stretched bands.

    The result is a hard-edged directional smear, not a soft motion blur.
    """
    _validate(image, severity)
    out = image.copy()
    height, width = out.shape[:2]

    n_patches = _lerp_int(SMEAR_MIN_PATCHES, SMEAR_MAX_PATCHES, severity)

    for _ in range(n_patches):
        horizontal = bool(rng.integers(0, 2))
        frac = float(rng.uniform(SMEAR_PATCH_MIN_FRAC, SMEAR_PATCH_MAX_FRAC))
        patch_h = max(4, int(height * frac))
        patch_w = max(4, int(width * frac))
        row = _rand_span(rng, height, patch_h)
        col = _rand_span(rng, width, patch_w)

        patch = out[row:row + patch_h, col:col + patch_w].astype(np.float32)
        shift_axis = 1 if horizontal else 0
        axis_len = patch.shape[shift_axis]

        # 1. Hold/stretch: repeat every ``stride``-th line to fill its stride.
        stride = max(2, int(axis_len * SMEAR_MAX_LENGTH_FRAC * severity))
        stride = min(stride, axis_len)
        indices = np.minimum(
            (np.arange(axis_len) // stride) * stride, axis_len - 1
        )
        smeared = patch[:, indices] if horizontal else patch[indices, :]

        # 2. Decaying trail off the stretched bands.
        trail = min(stride, axis_len - 1)
        for step in range(1, trail + 1):
            weight = 1.0 - step / (trail + 1.0)
            shifted = np.roll(smeared, step, axis=shift_axis) * weight
            smeared = np.maximum(smeared, shifted)

        out[row:row + patch_h, col:col + patch_w] = np.clip(smeared, 0, 255).astype(
            np.uint8
        )

    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
CorruptionFn = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]

#: Maps the artifact names in :data:`src.config.ARTIFACT_TYPES` to injectors.
CORRUPTION_FUNCTIONS: Dict[str, CorruptionFn] = {
    "block_corruption": apply_block_corruption,
    "horizontal_tear": apply_horizontal_tear,
    "shader_noise": apply_shader_noise,
    "stuck_pixels": apply_stuck_dead_pixels,
    "channel_swap": apply_channel_swap,
    "texture_smear": apply_texture_smear,
}

# Fail loudly at import time if the taxonomy and the registry drift apart.
if tuple(CORRUPTION_FUNCTIONS) != ARTIFACT_TYPES:
    raise RuntimeError(
        "CORRUPTION_FUNCTIONS keys must match config.ARTIFACT_TYPES exactly; "
        f"got {tuple(CORRUPTION_FUNCTIONS)} vs {ARTIFACT_TYPES}"
    )


def apply_corruption(
    image: np.ndarray,
    artifact: str,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the named artifact to ``image``.

    Args:
        image: ``uint8`` RGB array of shape ``(H, W, 3)``.
        artifact: One of :data:`src.config.ARTIFACT_TYPES`.
        severity: Strength in ``[0, 1]``.
        rng: Seeded generator driving all randomness.

    Returns:
        A new corrupted array; ``image`` is left untouched.

    Raises:
        KeyError: If ``artifact`` is not a known artifact name.
    """
    try:
        fn = CORRUPTION_FUNCTIONS[artifact]
    except KeyError as exc:
        raise KeyError(
            f"unknown artifact {artifact!r}; expected one of {ARTIFACT_TYPES}"
        ) from exc
    return fn(image, severity, rng)
