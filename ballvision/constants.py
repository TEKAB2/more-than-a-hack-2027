"""Shared constants for the ballvision package (no I/O)."""

SEED = 42
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
PATCH_SIZE = 256
HEATMAP_SIGMA = 4.0
VAL_FRACTION = 0.2
FPS = 25

# Pitch size used when converting metres <-> normalised ATD coordinates.
# Overridden by games.GAMES[game]["pitch"] when available.
DEFAULT_PITCH = (105.3, 68.0)

# Inference: reject detections that teleport faster than this (m/s).
MAX_BALL_SPEED_M_S = 45.0
# Minimum peak value on the normalised heatmap to accept a detection.
MIN_HEATMAP_PEAK = 0.35
# Presence head threshold (sigmoid).
MIN_PRESENCE = 0.45
# Search half-window around the kinematic prior, in metres on the pitch.
SEARCH_RADIUS_M = 8.0

DEFAULT_CACHE_DIRNAME = "ballvision"
WEIGHTS_NAME = "tracknet_unet.pt"
FILL_CACHE_NAME = "fill_{game}.npz"
INDEX_NAME = "index.npz"
