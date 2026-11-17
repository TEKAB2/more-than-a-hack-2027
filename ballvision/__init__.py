"""Auto-supervised ball detector (TrackNet-style) from ATD pseudo-labels.

Pseudo-labels come only from the ATD ball projected through the provided
homography — no human annotation. Training and inference must never dump
frame-level pixels or positions into logs; only aggregate metrics.

Weights are optional: without them the ATD loader is a no-op for vision fill.
"""

__version__ = "0.1.0"

SEED = 42
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
PATCH_SIZE = 256
HEATMAP_SIGMA = 4.0
VAL_FRACTION = 0.2
DEFAULT_CACHE_DIR = "cache/ballvision"
