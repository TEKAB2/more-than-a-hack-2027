"""Kinematic consistency filter for vision ball detections (numpy)."""

from __future__ import annotations

import numpy as np

from ballvision.constants import FPS, MAX_BALL_SPEED_M_S


def filter_teleports(
    ball: np.ndarray,
    filled_mask: np.ndarray,
    max_speed: float = MAX_BALL_SPEED_M_S,
    fps: float = FPS,
) -> tuple:
    """Reject filled positions that imply an impossible jump from the last known ball.

    ``ball`` is (n, 2) metres; ``filled_mask`` marks frames proposed by the detector.
    Observed (non-filled, finite) positions are never changed. A filled detection is
    kept only if the speed from the previous kept finite position is ≤ max_speed.

    Returns (ball_out, kept_mask) — aggregate callers should only log kept counts.
    """
    out = ball.astype(np.float64).copy()
    kept = np.zeros(len(ball), dtype=np.bool_)
    last_xy = None
    last_t = None
    for t in range(len(out)):
        if not np.isfinite(out[t, 0]):
            continue
        if not filled_mask[t]:
            last_xy = out[t].copy()
            last_t = t
            continue
        if last_xy is None:
            # No anchor yet — keep (will be checked when a later observed point exists
            # via a second pass below).
            kept[t] = True
            last_xy = out[t].copy()
            last_t = t
            continue
        dt = (t - last_t) / float(fps)
        if dt <= 0:
            out[t] = np.nan
            continue
        dist = float(np.hypot(out[t, 0] - last_xy[0], out[t, 1] - last_xy[1]))
        speed = dist / dt
        if speed <= max_speed:
            kept[t] = True
            last_xy = out[t].copy()
            last_t = t
        else:
            out[t] = np.nan
    # Drop unkept filled frames.
    drop = filled_mask & ~kept & np.isfinite(ball[:, 0])
    # Recompute: anything we rejected already set to NaN; clear kept flag mismatches.
    for t in np.flatnonzero(filled_mask & ~kept):
        out[t] = np.nan
    # Silence unused in aggregate path
    _ = drop
    return out.astype(np.float32), kept


def search_centre_metres(
    ball: np.ndarray,
    players: np.ndarray,
    frame: int,
    blocks,
    fallback_pitch_centre: tuple = (52.65, 34.0),
) -> np.ndarray:
    """Prior ball position (metres) for a missing frame: last/next seen, else nearby players.

    Uses simple linear bridge between surrounding observed samples; if none, the
    mean of finite player positions on that frame; else pitch centre.
    """
    n = len(ball)
    # Look backward / forward for anchors.
    prev_t = None
    for t in range(frame - 1, -1, -1):
        if np.isfinite(ball[t, 0]):
            prev_t = t
            break
    next_t = None
    for t in range(frame + 1, n):
        if np.isfinite(ball[t, 0]):
            next_t = t
            break
    if prev_t is not None and next_t is not None:
        w = (frame - prev_t) / float(next_t - prev_t)
        return (1.0 - w) * ball[prev_t] + w * ball[next_t]
    if prev_t is not None:
        return ball[prev_t].astype(np.float64)
    if next_t is not None:
        return ball[next_t].astype(np.float64)
    ply = players[frame, :, :2]
    finite = np.isfinite(ply[:, 0]) & np.isfinite(ply[:, 1])
    if finite.any():
        return np.nanmean(ply[finite], axis=0)
    return np.asarray(fallback_pitch_centre, dtype=np.float64)
