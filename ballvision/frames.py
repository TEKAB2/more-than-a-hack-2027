"""Frame sources for training / inference.

Real video decoding (OpenCV, BSD-3-Clause) is optional and only used on a
machine that holds the gated match assets. Synthetic sources drive unit tests
without any mp4.
"""

from __future__ import annotations

import os
from typing import Optional, Protocol, Tuple

import numpy as np

from ballvision.constants import IMAGE_HEIGHT, IMAGE_WIDTH


class FrameSource(Protocol):
    """RGB uint8 frames indexed by video frame (0-based)."""

    def n_frames(self) -> int:
        ...

    def get_rgb(self, frame: int) -> np.ndarray:
        """Return (H, W, 3) uint8 RGB. Must not print pixel content."""
        ...

    def close(self) -> None:
        ...


class SyntheticFrameSource:
    """Deterministic fake frames: a bright disk at a known pixel for tests."""

    def __init__(
        self,
        n: int,
        positions_xy: np.ndarray,
        height: int = IMAGE_HEIGHT,
        width: int = IMAGE_WIDTH,
        radius: int = 6,
        seed: int = 0,
    ):
        """positions_xy: (n, 2) float pixel centres; NaN => no ball drawn."""
        self._n = int(n)
        self._pos = np.asarray(positions_xy, dtype=np.float64)
        if self._pos.shape != (self._n, 2):
            raise ValueError("positions_xy must be (n_frames, 2)")
        self.height = int(height)
        self.width = int(width)
        self.radius = int(radius)
        self._rng = np.random.default_rng(seed)
        # Fixed background noise pattern (not re-rolled per call — deterministic).
        self._noise = (self._rng.integers(0, 40, size=(height, width, 3),
                                          dtype=np.uint8))

    def n_frames(self) -> int:
        return self._n

    def get_rgb(self, frame: int) -> np.ndarray:
        if not 0 <= frame < self._n:
            raise IndexError("frame {} out of range".format(frame))
        img = self._noise.copy()
        x, y = self._pos[frame]
        if np.isfinite(x) and np.isfinite(y):
            cx, cy = int(round(x)), int(round(y))
            yy, xx = np.ogrid[-self.radius:self.radius + 1, -self.radius:self.radius + 1]
            mask = xx * xx + yy * yy <= self.radius * self.radius
            y0 = max(0, cy - self.radius)
            x0 = max(0, cx - self.radius)
            y1 = min(self.height, cy + self.radius + 1)
            x1 = min(self.width, cx + self.radius + 1)
            my0 = y0 - (cy - self.radius)
            mx0 = x0 - (cx - self.radius)
            my1 = my0 + (y1 - y0)
            mx1 = mx0 + (x1 - x0)
            patch = img[y0:y1, x0:x1]
            m = mask[my0:my1, mx0:mx1]
            patch[m] = (240, 240, 40)
        return img

    def close(self) -> None:
        return None


class VideoFrameSource:
    """OpenCV VideoCapture wrapper. Import cv2 lazily (optional dep)."""

    def __init__(self, path: str, expect_size: Optional[Tuple[int, int]] = None):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        import cv2  # noqa: WPS433 — optional, BSD-3-Clause
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError("cannot open video: {}".format(path))
        self._n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._expect = expect_size  # (width, height) or None
        self._path = path

    def n_frames(self) -> int:
        return self._n

    def get_rgb(self, frame: int) -> np.ndarray:
        if not 0 <= frame < self._n:
            raise IndexError("frame {} out of range".format(frame))
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise IOError("failed to read frame {} from {}".format(frame, self._path))
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        if self._expect is not None:
            w, h = self._expect
            if rgb.shape[1] != w or rgb.shape[0] != h:
                rgb = self._cv2.resize(rgb, (w, h), interpolation=self._cv2.INTER_AREA)
        return rgb

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def open_video(path: str, expect_size: Optional[Tuple[int, int]] = None) -> VideoFrameSource:
    """Factory used by train/infer CLIs on Joy Boy's machine."""
    return VideoFrameSource(path, expect_size=expect_size)
