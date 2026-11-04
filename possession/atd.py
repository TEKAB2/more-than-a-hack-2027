"""Read the raw ATD (Metrica FIFA/EPTS export) into dense numpy arrays, with a cache.

Line format: "<frame_id>:<player>;<player>;...:<ball>", each group comma-separated
("x,y,speed" for players, at least "x,y,..." for the ball), NaN for missing values.
Coordinates are normalised 0-1; we convert to metres here so every later step is in metres.

Output (all indexed by VIDEO frame = frame_id - 1):
    players  float32 (n_frames, n_slots, 3)   x_m, y_m, speed_m_s
    ball     float32 (n_frames, 2)            x_m, y_m
    ball_extra float32 (n_frames, k)          any extra ball channels (z, flags...), NaN if absent
    team     int8    (n_slots,)               0 = home, 1 = away
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1


def team_of_each_track(meta_path):
    """0/1 per track slot. First <Team> in the file is home (FIFA/EPTS convention).

    The slot order of the raw file is the order of PlayerChannelRef in the DataFormatSpecification,
    which is not guaranteed to be the order of <Player> elements. Follow the refs when they exist
    (and say so if the two orders differ), fall back to <Player> order otherwise.
    """
    root = ET.parse(meta_path).getroot()
    tag = lambda e: e.tag.split("}")[-1]  # noqa: E731
    team_ids = [t.get("id") for t in root.iter() if tag(t) == "Team"]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    players = [p for p in root.iter() if tag(p) == "Player"]
    team_of_player = {p.get("id"): p.get("teamId") for p in players}
    by_player_order = [p.get("teamId") for p in players]

    channel_player = {c.get("id"): c.get("playerId") for c in root.iter()
                      if tag(c) == "PlayerChannel"}
    specs = [s for s in root.iter() if tag(s) == "DataFormatSpecification"]
    slots = by_player_order
    if len(specs) == 1:
        refs = [r.get("playerChannelId") for r in specs[0].iter() if tag(r) == "PlayerChannelRef"]
        xs = [channel_player.get(r) for r in refs if r and r.endswith("_x")]
        if xs and all(x in team_of_player for x in xs):
            slots = [team_of_player[x] for x in xs]
            if slots != by_player_order:
                print("note: channel order differs from <Player> order -- using channel order")
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def _floats(group):
    out = []
    for token in group.split(","):
        try:
            out.append(float(token))
        except ValueError:
            out.append(np.nan)
    return out


def parse_raw(raw_path, n_frames, n_slots, pitch):
    length, width = pitch
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    extras = {}
    n_extra = 0

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            groups = parts[1].split(";")
            for slot, group in enumerate(groups[:n_slots]):
                if group[:1] == "N":  # NaN,... -> slot not seen on this frame
                    continue
                values = _floats(group)
                players[frame, slot, :min(3, len(values))] = values[:3]
            values = _floats(parts[2])
            ball[frame, 0] = values[0] if values else np.nan
            ball[frame, 1] = values[1] if len(values) > 1 else np.nan
            if len(values) > 2:
                extras[frame] = values[2:]
                n_extra = max(n_extra, len(values) - 2)

    players[..., 0] *= length
    players[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width

    ball_extra = np.full((n_frames, n_extra), np.nan, dtype=np.float32)
    for frame, values in extras.items():
        ball_extra[frame, :len(values)] = values
    return players, ball, ball_extra


def load(meta_path, raw_path, n_frames, pitch, cache_path=None):
    """Arrays above, read from the cache when it is newer than the raw file."""
    if cache_path and os.path.exists(cache_path) and \
            os.path.getmtime(cache_path) >= os.path.getmtime(raw_path):
        data = np.load(cache_path)
        out = {key: data[key] for key in data.files}
        out["team"] = team_of_each_track(meta_path)  # cheap, and never stale
        return out

    team = team_of_each_track(meta_path)
    players, ball, ball_extra = parse_raw(raw_path, n_frames, len(team), pitch)
    out = {"players": players, "ball": ball, "ball_extra": ball_extra, "team": team}
    if cache_path:
        folder = os.path.dirname(cache_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        np.savez(cache_path, **out)
    return out
