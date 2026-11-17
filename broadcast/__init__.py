"""Broadcast-camera alignment and ball fusion for the tactical timeline.

The TV feed is not synchronised to the tactical video (DATA.md). This package recovers a
per-segment frame offset from player positions alone, rejects replays / slow-motion, and
exposes an aligned ball track that `possession.atd` can fuse into missing tactical ball
observations.

Public entry points live in `broadcast.align`.
"""

from broadcast.align import (  # noqa: F401
    aggregate_metrics,
    align_arrays,
    align_and_cache,
    continuous_segments,
    fuse_ball,
    load_aligned,
)
