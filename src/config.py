"""Central configuration constants shared across the project.

Everything tunable that is not a CLI flag lives here so that no magic
numbers are buried inside the training / dataset / model code.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_ROOT: Path = PROJECT_ROOT / "data"
# Real game captures are the source set; fetch them with
# ``scripts/fetch_game_frames.py``.  Training on procedurally drawn frames was
# measured to cost ~27 points of naming accuracy on real frames, so the drawn
# frames are a fallback for offline use only, never the default.
CLEAN_DIR: Path = DATA_ROOT / "game_frames"
#: Where ``scripts/make_demo_data.py`` writes its synthetic stand-in frames.
DEMO_CLEAN_DIR: Path = DATA_ROOT / "clean"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# --------------------------------------------------------------------------
# Artifact taxonomy
#
# ``ARTIFACT_TYPES`` is the ordered list of synthetic GPU render artifacts we
# inject.  ``CLASS_NAMES`` prepends the "clean" class, so the multi-class head
# predicts ``len(CLASS_NAMES)`` logits and index 0 always means "no corruption".
# --------------------------------------------------------------------------
CLEAN_CLASS_NAME: str = "clean"
ARTIFACT_TYPES: tuple[str, ...] = (
    "block_corruption",
    "horizontal_tear",
    "shader_noise",
    "stuck_pixels",
    "channel_swap",
    "texture_smear",
)
CLASS_NAMES: tuple[str, ...] = (CLEAN_CLASS_NAME, *ARTIFACT_TYPES)
NUM_CLASSES: int = len(CLASS_NAMES)
CLEAN_CLASS_INDEX: int = 0

# Binary head: 0 = clean, 1 = corrupted.
BINARY_CLASS_NAMES: tuple[str, ...] = ("clean", "corrupted")

# --------------------------------------------------------------------------
# Image / dataset defaults
# --------------------------------------------------------------------------
IMAGE_SIZE: int = 256
DEFAULT_CLEAN_RATIO: float = 0.35          # fraction of samples left uncorrupted
DEFAULT_SEVERITY_RANGE: tuple[float, float] = (0.20, 1.0)
DEFAULT_SAMPLES_PER_IMAGE: int = 8         # virtual epoch multiplier
DEFAULT_VAL_FRACTION: float = 0.2
DEFAULT_SEED: int = 1337

# ImageNet statistics — used for both the custom CNN and the pretrained
# ResNet18 so a single normalisation path serves every architecture.
NORM_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
NORM_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------
# Model / training defaults
# --------------------------------------------------------------------------
DEFAULT_ARCH: str = "small_cnn"
SUPPORTED_ARCHS: tuple[str, ...] = ("small_cnn", "resnet18")
DEFAULT_WIDTH: int = 32                    # base channel count of the small CNN
DEFAULT_DROPOUT: float = 0.2

DEFAULT_EPOCHS: int = 10
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LR: float = 3e-4
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_BINARY_LOSS_WEIGHT: float = 0.3    # weight of the auxiliary binary head

# --------------------------------------------------------------------------
# Evaluation defaults
# --------------------------------------------------------------------------
DEFAULT_LATENCY_FRAMES: int = 100
DEFAULT_LATENCY_WARMUP: int = 10
CONFUSION_MATRIX_FILENAME: str = "confusion_matrix.png"
METRICS_FILENAME: str = "eval_metrics.json"
