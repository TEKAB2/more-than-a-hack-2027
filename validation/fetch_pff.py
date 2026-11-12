"""Download and convert the 64 FIFA World Cup 2022 matches (PFF FC Open Dataset) into compact
25 fps `.npz` archives compatible with `validation/external.py` and `validation/train_seq.py`.

Usage
-----
1. From a local PFF FC folder (`Metadata/`, `Rosters/`, `Event Data/`, `Tracking Data/`):
    python -m validation.fetch_pff --src /path/to/pff-fifa-2022 --matches 3812 10517

2. Direct download from PFF FC's public Google Drive release (no browser needed):
    python -m validation.fetch_pff --download --matches 3812          # 1 match
    python -m validation.fetch_pff --download                         # all 64 matches

Why these matches matter
------------------------
PFF FC released synchronized broadcast tracking (~29.97 fps, Kalman-smoothed player trajectories)
and human-annotated event data (with possession sequence indicators `home_ball`) for all 64 games
of the 2022 FIFA Men's World Cup. These are OTHER games (explicitly allowed by RULES.md for
training and validation when cited in `docs/TECHNICAL.md`).

Format alignment details
------------------------
* Frame rate: PFF tracking is recorded at 29.97 fps (`videoTimeMs` / `periodElapsedTime`), with
  duplicate `frameNum` rows when a single `game_event` spans multiple `possession_event`s. We
  deduplicate by `(period, frameNum)` (merging attached events) and resample onto a uniform
  25.0 fps grid (`dt = 0.04 s`) per regulation period (periods 1 and 2 by default; `--include-et`
  for extra time periods 3 and 4).
* Coordinates: PFF uses a centre-origin metric system `[-L/2, +L/2] x [-W/2, +W/2]` (`105 x 68 m`).
  We shift by `(+L/2, +W/2)` to match the pipeline's `[0, L] x [0, W]` convention.
* Ball coordinates: Since `homePlayersSmoothed` and `awayPlayersSmoothed` are Kalman-smoothed,
  `ballsSmoothed` must be used by default so the ball lies at the smoothed player's feet (`balls` is
  unsmoothed and has a ~2-4 m spatial offset from `PlayersSmoothed`). Note that `balls` When dead
   contains stationary out-of-play balls or secondary balls near the touchline, which would create
  artificial stationary balls if merged into `ballsSmoothed`; therefore `--ball-source smoothed`
  uses `ballsSmoothed` purely (`external.fill_ball()` and `external.degrade()` then synthesize
  ATD-like dead-ball visibility). (`--ball-source raw` extracts unsmoothed `homePlayers`/`awayPlayers`
  + `balls[0]`.)
* Labels (`alive` and `owner`): Each `game_event` carries `game_event_type` (`FIRSTKICKOFF`,
  `SECONDKICKOFF`, `OTB` -> alive; `OUT`, `END` -> dead) and `home_ball` (boolean indicating
  which team controls the possession sequence, already absorbing opponent deflections/blocks).
* Set-piece events: When `Event Data/<game_id>.json` is available, set pieces (`K`, `F`, `C`, `G`,
  `T`, `P`, `D`) are saved in the `.npz` so `external.fake_set_piece_tags()` can generate realistic
  Smart Tagging restart cues.

Output per match (`pff_<game_id>.npz` or `idsse_pff_<game_id>.npz` with `--prefix idsse_pff_`):
    players (n, n_players, 2) float32   NaN when the player is not tracked on the frame
    team    (n_players,) int8           0 home, 1 away
    ball    (n, 2) float32
    alive   (n,) bool                   ball in play
    owner   (n,) int8                   0 home, 1 away, -1 unknown
    period  (n,) int8                   1 = first half, 2 = second half
    pitch   (2,) float64                [length, width] in metres
"""

import argparse
import bz2
import json
import os
import re
import tempfile
import urllib.request

import numpy as np

FPS = 25.0
DT = 1.0 / FPS
OUT = os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "external_data"
)

# PFF FC public Google Drive folder IDs (linked from kloppy and PFF FC's release blog)
DRIVE_ROOT_ID = "1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa"
DRIVE_FOLDERS = {
    "Event Data": "1A7ZvTNzVZ715PDATIhaGA9fPb1KfFmPx",
    "Metadata": "1vTCH5W9RMlfWmucTHbrPJ1LrkKnpTJw0",
    "Rosters": "1Rp8tpK-MIJt3oSMjufgX6swa9VZZNRy7",
    "Tracking Data": "1-ug2vouBfYazPy8z5lZlgd9abAAJSA5q",
}

ALL_MATCHES = (
    "3812", "3813", "3814", "3815", "3816", "3817", "3818", "3819",
    "3820", "3821", "3822", "3823", "3824", "3825", "3826", "3827",
    "3828", "3829", "3830", "3831", "3832", "3833", "3834", "3835",
    "3836", "3837", "3838", "3839", "3840", "3841", "3842", "3843",
    "3844", "3845", "3846", "3847", "3848", "3849", "3850", "3851",
    "3852", "3853", "3854", "3855", "3856", "3857", "3858", "3859",
    "10502", "10503", "10504", "10505", "10506", "10507", "10508", "10509",
    "10510", "10511", "10512", "10513", "10514", "10515", "10516", "10517",
)

SETPIECE_SUBTYPES = {
    "K": "KICK OFF",
    "F": "FREE KICK",
    "C": "CORNER KICK",
    "G": "GOAL KICK",
    "T": "THROW IN",
    "P": "PENALTY",
    "D": "DROP BALL",
}


def list_drive_folder(folder_id):
    """Return {filename: file_id} for all files inside a public Google Drive folder."""
    url = "https://drive.google.com/embeddedfolderview?id={}#list".format(folder_id)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", "ignore")
    entries = re.findall(
        r'id="entry-([^"]+)".*?class="flip-entry-title">([^<]+)<', html, re.S
    )
    return {title.strip(): fid.strip() for fid, title in entries}


def build_drive_index():
    """Map each folder type to {filename: drive_file_id} across the 64 World Cup matches."""
    return {name: list_drive_folder(fid) for name, fid in DRIVE_FOLDERS.items()}


def download_drive_file(file_id, dest_path):
    """Download a single file from Google Drive (handles virus-scan confirmation for large files)."""
    url = (
        "https://drive.usercontent.google.com/download"
        "?id={}&export=download&confirm=t".format(file_id)
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _open_tracking(path):
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _parse_metadata_and_roster(meta_path, roster_path):
    meta_raw = _load_json(meta_path)
    meta = meta_raw[0] if isinstance(meta_raw, list) else meta_raw
    roster = _load_json(roster_path)

    home_id = str(meta["homeTeam"]["id"])
    away_id = str(meta["awayTeam"]["id"])
    pitches = (meta.get("stadium") or {}).get("pitches") or [{}]
    length = float(pitches[0].get("length") or 105.0)
    width = float(pitches[0].get("width") or 68.0)

    home_players = [r for r in roster if str(r["team"]["id"]) == home_id]
    away_players = [r for r in roster if str(r["team"]["id"]) == away_id]
    home_players.sort(key=lambda r: int(r["shirtNumber"]))
    away_players.sort(key=lambda r: int(r["shirtNumber"]))

    slot_home = {int(r["shirtNumber"]): i for i, r in enumerate(home_players)}
    slot_away = {
        int(r["shirtNumber"]): len(home_players) + i
        for i, r in enumerate(away_players)
    }
    team = np.array(
        [0] * len(home_players) + [1] * len(away_players), dtype=np.int8
    )
    return length, width, home_id, away_id, slot_home, slot_away, team


def _parse_event_setpieces(events_path, home_id):
    """Extract set-piece events from Event Data/<id>.json keyed by period and video time."""
    if not events_path or not os.path.exists(events_path):
        return []
    raw_events = _load_json(events_path)
    if isinstance(raw_events, dict):
        raw_events = raw_events.get("events", [])
    setpieces = []
    seen_ge = set()
    for ev in raw_events:
        gid = ev.get("gameEventId")
        if gid in seen_ge:
            continue
        ge = ev.get("gameEvents") or {}
        if ge.get("initialNonEvent"):
            continue
        sp = ge.get("setpieceType")
        ge_type = ge.get("gameEventType") or ""
        if sp in SETPIECE_SUBTYPES and sp != "O":
            seen_ge.add(gid)
            period = int(ge.get("period") or 0)
            video_time = float(ev.get("startTime") or 0.0)
            if "homeTeam" in ge and ge["homeTeam"] is not None:
                team_idx = 0 if ge["homeTeam"] else 1
            else:
                team_idx = 0 if str(ge.get("teamId")) == home_id else 1
            subtype = SETPIECE_SUBTYPES[sp]
            if "KICKOFF" in ge_type:
                subtype = "KICK OFF"
            setpieces.append({
                "period": period,
                "video_time": video_time,
                "team": team_idx,
                "type": "SET PIECE",
                "subtype": subtype,
            })
    return setpieces


def convert_match(
    game_id,
    meta_path,
    roster_path,
    tracking_path,
    events_path=None,
    out_dir=OUT,
    prefix="pff_",
    ball_source="smoothed",
    include_et=False,
):
    """Convert one PFF FC match into a 25 fps `.npz` file compatible with `validation/external.py`."""
    length, width, home_id, away_id, slot_home, slot_away, team = (
        _parse_metadata_and_roster(meta_path, roster_path)
    )
    n_players = len(team)
    max_period = 4 if include_et else 2

    # Pass 1: stream JSONL, deduplicate (period, frameNum), and track continuous state.
    by_period = {}
    alive_state = False
    owner_state = -1
    current_period = None

    with _open_tracking(tracking_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            p = row.get("period")
            if p is None or p < 1 or p > max_period:
                continue
            p = int(p)
            if p != current_period:
                alive_state = False
                owner_state = -1
                current_period = p

            ge = row.get("game_event")
            if ge:
                ge_type = ge.get("game_event_type")
                if ge_type in ("OUT", "END"):
                    alive_state = False
                elif ge_type in (
                    "FIRSTKICKOFF",
                    "SECONDKICKOFF",
                    "THIRDKICKOFF",
                    "FOURTHKICKOFF",
                    "OTB",
                ):
                    alive_state = True
                if ge.get("home_ball") is not None:
                    owner_state = 0 if ge["home_ball"] else 1
                elif ge.get("home_team") is not None and alive_state:
                    owner_state = 0 if ge["home_team"] else 1

            fnum = int(row["frameNum"])
            p_dict = by_period.setdefault(p, {})
            if fnum in p_dict:
                # Duplicate frameNum for multiple possession_events on same game_event:
                # update state only, keep existing coordinates.
                rec = p_dict[fnum]
                rec["alive"] = alive_state
                rec["owner"] = owner_state
                continue

            # Extract player coordinates (smoothed by default, or raw if --ball-source raw)
            pxy = np.full((n_players, 2), np.nan, dtype=np.float32)
            use_smoothed = ball_source != "raw"
            h_list = (
                row.get("homePlayersSmoothed") if use_smoothed else row.get("homePlayers")
            )
            if h_list is None:
                h_list = row.get("homePlayers") or []
            for pl in h_list:
                j = int(pl["jerseyNum"])
                if j in slot_home and pl.get("x") is not None and pl.get("y") is not None:
                    pxy[slot_home[j], 0] = float(pl["x"]) + length / 2.0
                    pxy[slot_home[j], 1] = float(pl["y"]) + width / 2.0

            a_list = (
                row.get("awayPlayersSmoothed") if use_smoothed else row.get("awayPlayers")
            )
            if a_list is None:
                a_list = row.get("awayPlayers") or []
            for pl in a_list:
                j = int(pl["jerseyNum"])
                if j in slot_away and pl.get("x") is not None and pl.get("y") is not None:
                    pxy[slot_away[j], 0] = float(pl["x"]) + length / 2.0
                    pxy[slot_away[j], 1] = float(pl["y"]) + width / 2.0

            # Extract ball coordinates
            bx, by = np.nan, np.nan
            if use_smoothed:
                bs = row.get("ballsSmoothed")
                if isinstance(bs, list):
                    bs = bs[0] if bs else None
                if bs and bs.get("x") is not None and bs.get("y") is not None:
                    bx = float(bs["x"]) + length / 2.0
                    by = float(bs["y"]) + width / 2.0
            else:
                br = row.get("balls")
                if br and len(br) > 0 and br[0].get("x") is not None and br[0].get("y") is not None:
                    bx = float(br[0]["x"]) + length / 2.0
                    by = float(br[0]["y"]) + width / 2.0

            p_dict[fnum] = {
                "t": float(row.get("periodElapsedTime") or 0.0),
                "v_ms": float(row.get("videoTimeMs") or 0.0),
                "players": pxy,
                "ball": (bx, by),
                "alive": alive_state,
                "owner": owner_state,
            }

    # Pass 2: resample each period from native ~29.97 fps onto a uniform 25.0 fps grid.
    all_players, all_ball, all_alive, all_owner, all_period = [], [], [], [], []
    period_vtime_map = {}
    offset = 0

    for p in sorted(by_period):
        fnums = sorted(by_period[p])
        recs = [by_period[p][fn] for fn in fnums]
        m = len(recs)
        if m < 2:
            continue

        t_src = np.array([r["t"] for r in recs], dtype=np.float64)
        v_src = np.array([r["v_ms"] / 1000.0 for r in recs], dtype=np.float64)
        # Ensure strictly non-decreasing timestamps
        t_src = np.maximum.accumulate(t_src)
        if t_src[-1] <= t_src[0]:
            t_src = np.arange(m, dtype=np.float64) / 29.97

        P_src = np.stack([r["players"] for r in recs], axis=0)  # (m, n_players, 2)
        B_src = np.array([r["ball"] for r in recs], dtype=np.float32)  # (m, 2)
        alive_src = np.array([r["alive"] for r in recs], dtype=bool)
        owner_src = np.array([r["owner"] for r in recs], dtype=np.int8)

        t_grid = np.arange(t_src[0], t_src[-1] + 1e-9, DT, dtype=np.float64)
        k = len(t_grid)

        # Nearest source frame index for each 25 fps grid point
        idx_right = np.searchsorted(t_src, t_grid, side="left")
        idx_right = np.clip(idx_right, 0, m - 1)
        idx_left = np.clip(idx_right - 1, 0, m - 1)
        choose_right = np.abs(t_src[idx_right] - t_grid) < np.abs(t_grid - t_src[idx_left])
        nearest_idx = np.where(choose_right, idx_right, idx_left)

        # Linearly interpolate continuous positions where adjacent frames are both valid (< 0.1 s)
        dt_span = t_src[idx_right] - t_src[idx_left]
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = np.where(dt_span > 1e-6, (t_grid - t_src[idx_left]) / dt_span, 0.0)
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        valid_span = (dt_span > 0) & (dt_span < 0.12)

        P_left, P_right = P_src[idx_left], P_src[idx_right]
        both_p = valid_span[:, None, None] & ~np.isnan(P_left) & ~np.isnan(P_right)
        P_25 = np.where(
            both_p,
            P_left + alpha[:, None, None] * (P_right - P_left),
            P_src[nearest_idx],
        ).astype(np.float32)

        B_left, B_right = B_src[idx_left], B_src[idx_right]
        both_b = valid_span[:, None] & ~np.isnan(B_left) & ~np.isnan(B_right)
        B_25 = np.where(
            both_b,
            B_left + alpha[:, None] * (B_right - B_left),
            B_src[nearest_idx],
        ).astype(np.float32)

        alive_25 = alive_src[nearest_idx]
        owner_25 = owner_src[nearest_idx]
        period_25 = np.full(k, p, dtype=np.int8)
        v_25 = np.interp(t_grid, t_src, v_src)

        period_vtime_map[p] = (offset, v_25)
        offset += k

        all_players.append(P_25)
        all_ball.append(B_25)
        all_alive.append(alive_25)
        all_owner.append(owner_25)
        all_period.append(period_25)

    P = np.concatenate(all_players, axis=0)
    B = np.concatenate(all_ball, axis=0)
    alive = np.concatenate(all_alive, axis=0)
    owner = np.concatenate(all_owner, axis=0)
    period = np.concatenate(all_period, axis=0)

    # Align set-piece events from Event Data/<id>.json to 25 fps frame indices
    raw_sp = _parse_event_setpieces(events_path, home_id)
    sp_frames, sp_teams, sp_subtypes = [], [], []
    for sp in raw_sp:
        p = sp["period"]
        if p not in period_vtime_map:
            continue
        base_idx, v_25 = period_vtime_map[p]
        local_idx = int(np.argmin(np.abs(v_25 - sp["video_time"])))
        frame_25 = base_idx + local_idx
        sp_frames.append(frame_25)
        sp_teams.append(sp["team"])
        sp_subtypes.append(sp["subtype"])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "{}{}.npz".format(prefix, game_id))
    save_kwargs = dict(
        players=P,
        team=team,
        ball=B,
        alive=alive,
        owner=owner,
        period=period,
        pitch=np.array([length, width], dtype=np.float64),
    )
    if sp_frames:
        save_kwargs["setpiece_frames"] = np.array(sp_frames, dtype=np.int32)
        save_kwargs["setpiece_teams"] = np.array(sp_teams, dtype=np.int8)
        save_kwargs["setpiece_subtypes"] = np.array(sp_subtypes, dtype="<U16")

    np.savez_compressed(out_path, **save_kwargs)
    n = len(B)
    alive_share = float(alive.mean()) if n else 0.0
    home_share = float((owner[alive] == 0).mean()) if alive.any() else 0.0
    seen_share = float((~np.isnan(B[:, 0])).mean()) if n else 0.0
    print(
        "pff_{}  {} frames (25 fps)  alive {:.0%}  home owns {:.0%} of alive  "
        "ball seen {:.0%}  setpieces {}  -> {}".format(
            game_id,
            n,
            alive_share,
            home_share,
            seen_share,
            len(sp_frames),
            os.path.relpath(out_path),
        )
    )
    return out_path


def process_one(
    game_id,
    src_dir=None,
    drive_index=None,
    out_dir=OUT,
    prefix="pff_",
    ball_source="smoothed",
    include_et=False,
):
    out_path = os.path.join(out_dir, "{}{}.npz".format(prefix, game_id))
    if os.path.exists(out_path):
        print(game_id, "already done ->", out_path)
        return out_path

    if src_dir:
        meta_path = os.path.join(src_dir, "Metadata", "{}.json".format(game_id))
        roster_path = os.path.join(src_dir, "Rosters", "{}.json".format(game_id))
        events_path = os.path.join(src_dir, "Event Data", "{}.json".format(game_id))
        track_bz2 = os.path.join(
            src_dir, "Tracking Data", "{}.jsonl.bz2".format(game_id)
        )
        track_raw = os.path.join(src_dir, "Tracking Data", "{}.jsonl".format(game_id))
        tracking_path = track_bz2 if os.path.exists(track_bz2) else track_raw
        return convert_match(
            game_id,
            meta_path,
            roster_path,
            tracking_path,
            events_path=events_path if os.path.exists(events_path) else None,
            out_dir=out_dir,
            prefix=prefix,
            ball_source=ball_source,
            include_et=include_et,
        )

    if drive_index is None:
        raise ValueError("Either --src or --download is required")

    meta_fid = drive_index["Metadata"].get("{}.json".format(game_id))
    roster_fid = drive_index["Rosters"].get("{}.json".format(game_id))
    events_fid = drive_index["Event Data"].get("{}.json".format(game_id))
    track_fid = drive_index["Tracking Data"].get("{}.jsonl.bz2".format(game_id))
    if not (meta_fid and roster_fid and track_fid):
        raise RuntimeError("Missing files on Google Drive for match {}".format(game_id))

    with tempfile.TemporaryDirectory(prefix="pff_{}_".format(game_id)) as tmp:
        meta_path = os.path.join(tmp, "{}_meta.json".format(game_id))
        roster_path = os.path.join(tmp, "{}_roster.json".format(game_id))
        events_path = os.path.join(tmp, "{}_events.json".format(game_id))
        tracking_path = os.path.join(tmp, "{}.jsonl.bz2".format(game_id))

        download_drive_file(meta_fid, meta_path)
        download_drive_file(roster_fid, roster_path)
        if events_fid:
            download_drive_file(events_fid, events_path)
        else:
            events_path = None
        download_drive_file(track_fid, tracking_path)

        return convert_match(
            game_id,
            meta_path,
            roster_path,
            tracking_path,
            events_path=events_path,
            out_dir=out_dir,
            prefix=prefix,
            ball_source=ball_source,
            include_et=include_et,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert PFF FC World Cup 2022 matches to 25 fps .npz archives."
    )
    parser.add_argument(
        "--matches",
        nargs="*",
        default=list(ALL_MATCHES),
        help="PFF game IDs to convert (default: all 64 matches)",
    )
    parser.add_argument(
        "--src",
        default=os.environ.get("MTAH_PFF_RAW_DIR"),
        help="Local folder holding Metadata/, Rosters/, Event Data/, Tracking Data/",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Stream/download directly from PFF FC's public Google Drive folder",
    )
    parser.add_argument(
        "--out",
        default=OUT,
        help="Output folder for .npz files (default: validation/external_data)",
    )
    parser.add_argument(
        "--prefix",
        default="pff_",
        help="Filename prefix (default: 'pff_'; use 'idsse_pff_' to auto-include in external.py)",
    )
    parser.add_argument(
        "--ball-source",
        choices=("smoothed", "raw"),
        default="smoothed",
        help="Coherent player+ball coordinate pair to extract (default: smoothed)",
    )
    parser.add_argument(
        "--include-et",
        action="store_true",
        help="Include extra-time periods 3 and 4 when present (default: regulation 1 & 2 only)",
    )
    args = parser.parse_args()

    if not args.src and not args.download:
        parser.error("Provide --src /path/to/pff-fifa-2022 or pass --download")

    drive_index = build_drive_index() if args.download and not args.src else None
    for game_id in args.matches:
        process_one(
            str(game_id),
            src_dir=args.src,
            drive_index=drive_index,
            out_dir=args.out,
            prefix=args.prefix,
            ball_source=args.ball_source,
            include_et=args.include_et,
        )


if __name__ == "__main__":
    main()
