"""Organizer hard rules for ballvision (email clarification, 2026-11).

FORBIDDEN
  - training, fitting or calibrating on BAR-ATH ground-truth possession
  - training / fine-tuning / calibrating on the PROVIDED VIDEO
    (tactical or broadcast mp4) or on patches/labels derived from that match
  - uploading provided data to AI services; hand labels on the game

ALLOWED
  - using video / ATD / Smart Tagging as INPUT at inference
  - inspecting the video to understand possession (humans)
  - external datasets only for train / tune (or synthetic data)

The default provided game id is ``BAR-ATH`` (see ``games.GAME``).
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

# Match id(s) of the gated challenge bundle. Never train on these.
PROVIDED_GAME_IDS = frozenset({"BAR-ATH"})

TRAIN_FORBIDDEN_MSG = (
    "FORBIDDEN by organizer rules: cannot train/fine-tune/calibrate ballvision "
    "on the provided match (BAR-ATH) video, ATD-derived patches, or possession GT. "
    "Train on EXTERNAL tracking/video (or synthetic) only; use BAR-ATH at inference "
    "with frozen weights. See ballvision/policy.py and docs/TECHNICAL.md §2.1."
)


def is_provided_game(game: Optional[str]) -> bool:
    if game is None:
        return False
    return str(game).strip().upper() in PROVIDED_GAME_IDS


def refuse_train_on_provided_game(game: Optional[str]) -> None:
    """Raise SystemExit if ``game`` is the gated challenge match."""
    if is_provided_game(game):
        raise SystemExit(TRAIN_FORBIDDEN_MSG)


def refuse_train_cache_from_provided(cache_dir: str) -> None:
    """Refuse a cache dir whose index_meta.json names a provided game."""
    meta_path = os.path.join(cache_dir, "index_meta.json")
    if not os.path.isfile(meta_path):
        return
    with open(meta_path) as handle:
        meta = json.load(handle)
    source = meta.get("source_game") or meta.get("game")
    if is_provided_game(source):
        raise SystemExit(
            TRAIN_FORBIDDEN_MSG
            + " (index_meta.json source_game={!r})".format(source)
        )
    if meta.get("for_training") is False and meta.get("provided_match"):
        raise SystemExit(TRAIN_FORBIDDEN_MSG)


def mark_index_meta_training_allowed(
    meta: dict,
    *,
    source_game: str,
    for_training: bool,
) -> dict:
    """Annotate index metadata; ``for_training`` must be False for BAR-ATH."""
    out = dict(meta)
    out["source_game"] = source_game
    out["for_training"] = bool(for_training) and not is_provided_game(source_game)
    out["provided_match"] = is_provided_game(source_game)
    if is_provided_game(source_game):
        out["for_training"] = False
        out["policy_note"] = (
            "Provided match: inference input only. Do not train on these patches."
        )
    return out
