"""TrackNet-style ball detector helpers (optional fill for missing ATD ball).

Organizer rule: do NOT train/fine-tune/calibrate on BAR-ATH video or on
ATD-derived patches from that match. Train on external or synthetic data only.
BAR-ATH video/ATD may be used as inference INPUT with frozen weights.

See ``ballvision.policy`` and ``docs/TECHNICAL.md`` §2.1.
"""

__version__ = "0.1.0"

SEED = 42
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
PATCH_SIZE = 256
HEATMAP_SIGMA = 4.0
VAL_FRACTION = 0.2
DEFAULT_CACHE_DIR = "cache/ballvision"
