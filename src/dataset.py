"""On-the-fly corruption dataset.

``CorruptionDataset`` reads clean frames from ``data/clean`` and synthesises a
corrupted variant for most of them at access time.  Generating corruption in
``__getitem__`` rather than pre-rendering to disk means:

* a handful of clean frames yields an effectively unlimited training set;
* every epoch sees fresh severities and artifact placements, which is free
  augmentation and makes the model rely on the artifact signature rather than
  on memorised pixel positions;
* no corrupted data has to be shipped or downloaded.

Each sample carries two labels, matching the two heads of the model:

``binary``
    ``0`` clean / ``1`` corrupted.
``multiclass``
    Index into :data:`src.config.CLASS_NAMES`, where ``0`` is ``clean`` and the
    remaining indices are the artifact types.

Determinism: a sample's artifact, severity and placement are derived from a
generator seeded with ``(dataset_seed, index)``, so sample *i* is identical on
every epoch and across processes — important for a reproducible validation
split.  Pass ``deterministic=False`` for training if fresh randomness per epoch
is preferred.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config
from .corruption import apply_corruption


def list_clean_images(clean_dir: Path | str) -> list[Path]:
    """Return the sorted list of readable image paths under ``clean_dir``.

    Args:
        clean_dir: Directory holding the clean source frames.

    Returns:
        Sorted list of image paths (sorted so splits are reproducible).

    Raises:
        FileNotFoundError: If the directory is missing or holds no images.
    """
    root = Path(clean_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"clean image directory not found: {root}. "
            "Run scripts/make_demo_data.py to generate demo frames."
        )
    paths = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(
            f"no images with extensions {config.IMAGE_EXTENSIONS} in {root}. "
            "Run scripts/make_demo_data.py to generate demo frames."
        )
    return paths


def split_paths(
    paths: Sequence[Path],
    val_fraction: float = config.DEFAULT_VAL_FRACTION,
    seed: int = config.DEFAULT_SEED,
) -> tuple[list[Path], list[Path]]:
    """Split image paths into train/val subsets.

    The split is by *source image*, not by generated sample, so no clean frame
    ever appears in both subsets.

    Args:
        paths: Clean image paths.
        val_fraction: Fraction of images held out for validation.
        seed: Seed for the shuffle.

    Returns:
        ``(train_paths, val_paths)``; each holds at least one image whenever
        two or more images are available.

    Raises:
        ValueError: If ``val_fraction`` is outside ``[0, 1)`` or ``paths`` is
            empty.
    """
    if not paths:
        raise ValueError("cannot split an empty list of paths")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    ordered = list(paths)
    if len(ordered) == 1:
        # Degenerate but useful for smoke tests: reuse the single frame.
        return ordered, ordered

    indices = np.random.default_rng(seed).permutation(len(ordered))
    n_val = max(1, int(round(len(ordered) * val_fraction)))
    n_val = min(n_val, len(ordered) - 1)  # always leave a training image
    val_idx = set(indices[:n_val].tolist())

    train = [p for i, p in enumerate(ordered) if i not in val_idx]
    val = [p for i, p in enumerate(ordered) if i in val_idx]
    return train, val


def load_image(path: Path, image_size: int) -> np.ndarray:
    """Load an image as a resized ``uint8`` RGB array of shape ``(S, S, 3)``."""
    with Image.open(path) as img:
        rgb = img.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        # np.array (not asarray) copies: PIL hands back a read-only buffer, and
        # torch.from_numpy warns on non-writable arrays.
        return np.array(rgb, dtype=np.uint8)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert a ``uint8`` HWC array to a normalised float CHW tensor."""
    chw = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(config.NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(config.NORM_STD).view(3, 1, 1)
    return (chw - mean) / std


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Invert :func:`to_tensor` for visualisation; returns a ``uint8`` HWC array."""
    mean = torch.tensor(config.NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(config.NORM_STD).view(3, 1, 1)
    chw = (tensor.detach().cpu() * std + mean).clamp(0, 1) * 255.0
    return chw.permute(1, 2, 0).numpy().astype(np.uint8)


class CorruptionDataset(Dataset):
    """Clean frames in, (tensor, binary label, multi-class label) out.

    Args:
        paths: Clean source image paths.
        image_size: Square side length every frame is resized to.
        samples_per_image: Virtual epoch multiplier — ``len(dataset)`` is
            ``len(paths) * samples_per_image``, each entry a different random
            corruption of one source frame.
        clean_ratio: Fraction of samples left uncorrupted (label ``clean``).
        severity_range: ``(low, high)`` bounds sampled uniformly per corrupted
            sample.
        seed: Base seed; combined with the sample index for per-sample RNG.
        deterministic: When ``True`` sample *i* is byte-identical on every
            epoch.  When ``False`` a fresh entropy source is mixed in, giving
            new corruption on every access (preferred for training).
        cache_images: Keep decoded source frames in memory.  Demo datasets are
            tiny, so this is on by default and removes disk IO from the loop.

    Attributes:
        class_names: Multi-class label names, index-aligned with the labels.
        binary_class_names: Binary label names.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        image_size: int = config.IMAGE_SIZE,
        samples_per_image: int = config.DEFAULT_SAMPLES_PER_IMAGE,
        clean_ratio: float = config.DEFAULT_CLEAN_RATIO,
        severity_range: tuple[float, float] = config.DEFAULT_SEVERITY_RANGE,
        seed: int = config.DEFAULT_SEED,
        deterministic: bool = True,
        cache_images: bool = True,
    ) -> None:
        if not paths:
            raise ValueError("CorruptionDataset requires at least one image path")
        if image_size < 2:
            raise ValueError(f"image_size must be >= 2, got {image_size}")
        if samples_per_image < 1:
            raise ValueError(
                f"samples_per_image must be >= 1, got {samples_per_image}"
            )
        if not 0.0 <= clean_ratio <= 1.0:
            raise ValueError(f"clean_ratio must be in [0, 1], got {clean_ratio}")
        low, high = severity_range
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(
                f"severity_range must satisfy 0 <= low <= high <= 1, got {severity_range}"
            )

        self.paths = list(paths)
        self.image_size = image_size
        self.samples_per_image = samples_per_image
        self.clean_ratio = clean_ratio
        self.severity_range = (low, high)
        self.seed = seed
        self.deterministic = deterministic
        self.cache_images = cache_images

        self.class_names = config.CLASS_NAMES
        self.binary_class_names = config.BINARY_CLASS_NAMES
        self._cache: dict[int, np.ndarray] = {}

    # -- torch.utils.data.Dataset API ---------------------------------------
    def __len__(self) -> int:
        return len(self.paths) * self.samples_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(image_tensor, binary_label, multiclass_label)``."""
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range for {len(self)} samples")

        source = self._source_image(index % len(self.paths))
        rng = self._rng_for(index)
        corrupted, class_index = self._corrupt(source, rng)

        binary_label = int(class_index != config.CLEAN_CLASS_INDEX)
        return (
            to_tensor(corrupted),
            torch.tensor(binary_label, dtype=torch.long),
            torch.tensor(class_index, dtype=torch.long),
        )

    # -- Helpers ------------------------------------------------------------
    def _rng_for(self, index: int) -> np.random.Generator:
        """Per-sample generator, reproducible in deterministic mode."""
        if self.deterministic:
            return np.random.default_rng([self.seed, index])
        # Fresh OS entropy, so each epoch draws different corruption.  Worker
        # processes get independent streams because the entropy is drawn per
        # call rather than inherited from the parent's state.
        return np.random.default_rng()

    def _source_image(self, path_index: int) -> np.ndarray:
        """Load (and optionally cache) the decoded clean frame."""
        if self.cache_images and path_index in self._cache:
            return self._cache[path_index]
        image = load_image(self.paths[path_index], self.image_size)
        if self.cache_images:
            self._cache[path_index] = image
        return image

    def _corrupt(
        self, source: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, int]:
        """Decide clean-vs-corrupted, then apply one artifact if corrupted."""
        if rng.random() < self.clean_ratio:
            return source, config.CLEAN_CLASS_INDEX

        artifact_index = int(rng.integers(0, len(config.ARTIFACT_TYPES)))
        artifact = config.ARTIFACT_TYPES[artifact_index]
        low, high = self.severity_range
        severity = float(rng.uniform(low, high))

        corrupted = apply_corruption(source, artifact, severity, rng)
        # +1 because index 0 of CLASS_NAMES is reserved for "clean".
        return corrupted, artifact_index + 1


def build_datasets(
    clean_dir: Path | str = config.CLEAN_DIR,
    image_size: int = config.IMAGE_SIZE,
    samples_per_image: int = config.DEFAULT_SAMPLES_PER_IMAGE,
    clean_ratio: float = config.DEFAULT_CLEAN_RATIO,
    severity_range: tuple[float, float] = config.DEFAULT_SEVERITY_RANGE,
    val_fraction: float = config.DEFAULT_VAL_FRACTION,
    seed: int = config.DEFAULT_SEED,
) -> tuple[CorruptionDataset, CorruptionDataset]:
    """Build the train/validation dataset pair from a clean-image directory.

    Training uses non-deterministic corruption (fresh artifacts every epoch);
    validation is deterministic so the metric is comparable across epochs.

    Returns:
        ``(train_dataset, val_dataset)``.
    """
    paths = list_clean_images(clean_dir)
    train_paths, val_paths = split_paths(paths, val_fraction=val_fraction, seed=seed)

    common = dict(
        image_size=image_size,
        clean_ratio=clean_ratio,
        severity_range=severity_range,
    )
    train = CorruptionDataset(
        train_paths,
        samples_per_image=samples_per_image,
        seed=seed,
        deterministic=False,
        **common,
    )
    val = CorruptionDataset(
        val_paths,
        samples_per_image=samples_per_image,
        seed=seed + 1,
        deterministic=True,
        **common,
    )
    return train, val
