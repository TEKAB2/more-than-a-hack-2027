"""Smart Tagging -> per-frame priors.

Two uses, both as *evidence* fed to the decoder, never copied as labels:
  1. Set pieces are restarts: the ball is dead just before them, alive just after.
  2. Phases carry a Team tag: a weak prior on which team holds the ball during the phase.

The mapping from code to "who has the ball" below is a hypothesis about Metrica's vocabulary.
Verify it by WATCHING the video (allowed: watching to understand), not by comparing to the
public ground truth (that would be tuning on it).
"""

import difflib
import xml.etree.ElementTree as ET

import numpy as np

# +1: the tagged team is the one in possession.  -1: the tagged team is defending.
CODE_SIGN = {
    "build up": +1,
    "progression": +1,
    "ball in final third": +1,
    "creating chances": +1,
    "long ball": +1,
    "attacking transition": +1,
    "players in the box": +1,
    "ball in the box": +1,
    "defensive transition": -1,
    "defending in defensive third": -1,
    "defending in middle third": -1,
    "defending in attacking third": -1,
}
SET_PIECE_CODE = "set pieces"
GOAL_CODE = "goals"


def load(path, fps=25):
    out = []
    for instance in ET.parse(path).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        tags = {}
        for label in instance.findall("label"):
            group, text = label.findtext("group"), label.findtext("text")
            if group:
                tags[group.strip().lower()] = (text or "").strip()
        out.append({
            "code": (instance.findtext("code") or "").strip().lower(),
            "start": int(round(start_s * fps)),
            "end": int(round(end_s * fps)),
            "team": tags.get("team", ""),
            "type": tags.get("type", "").lower(),
        })
    return out


def side_of(team_tag, home_name, away_name):
    """0 home, 1 away, None for N/A or unrecognised. Fuzzy, because names differ by source."""
    tag = team_tag.lower()
    if not tag or tag in ("n/a", "na", "none"):
        return None
    home = difflib.SequenceMatcher(None, tag, home_name.lower()).ratio()
    away = difflib.SequenceMatcher(None, tag, away_name.lower()).ratio()
    if max(home, away) < 0.3:
        return None
    return 0 if home >= away else 1


def team_prior(events, n_frames, home_name, away_name):
    """float32 (n,): >0 favours home in possession, <0 away. Overlapping phases add up."""
    prior = np.zeros(n_frames, dtype=np.float32)
    for event in events:
        sign = CODE_SIGN.get(event["code"])
        side = side_of(event["team"], home_name, away_name)
        if sign is None or side is None:
            continue
        value = sign * (1.0 if side == 0 else -1.0)
        start, end = max(0, event["start"]), min(n_frames, event["end"])
        prior[start:end] += value
    return prior


def set_pieces(events, home_name, away_name):
    """[(start, end, type, side)] for set pieces, and goals as [(start, end, side)]."""
    pieces, goals = [], []
    for event in events:
        side = side_of(event["team"], home_name, away_name)
        if event["code"] == SET_PIECE_CODE:
            pieces.append((event["start"], event["end"], event["type"], side))
        elif event["code"] == GOAL_CODE:
            goals.append((event["start"], event["end"], side))
    return pieces, goals
