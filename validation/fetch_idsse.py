"""Download the 7 open Bundesliga matches (IDSSE / Sportec Open Data) and save compact arrays.

    pip install kloppy==3.19.1
    python -m validation.fetch_idsse                 # -> validation/external_data/idsse_*.npz

Why these matches matter
------------------------
The challenge ground truth is "derived from corrected tracking data: a frame is D where the
corrected tracking marks the ball dead, otherwise the team the tracking names as owning it"
(CHALLENGE.md). The DFL position data of these open matches carries exactly those two fields on
every frame -- BallStatus and BallPossession -- from the league's own corrected tracking. So they
are much closer to the challenge's labels than labels re-derived from event data, and there are
seven full matches instead of three.

Source and licence: Bassek et al., "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y; data on figshare
(doi:10.6084/m9.figshare.28196177), CC BY 4.0, distributed through kloppy. These are OTHER games,
which RULES.md allows for training and validation; the technical note must cite them.

Output per match (all arrays indexed by frame, 25 fps, metres, pitch origin at a corner):
    players (n, n_players, 2) float32   NaN when the player is not on the pitch
    team    (n_players,) int8           0 home, 1 away
    ball    (n, 2) float32
    alive   (n,) bool                   BallStatus
    owner   (n,) int8                   0 home, 1 away, -1 unknown   (BallPossession)
    period  (n,) int8
"""

import argparse
import os

import numpy as np

MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_data")


def convert(match_id, out_dir):
    from kloppy import sportec

    ds = sportec.load_open_tracking_data(match_id=match_id, only_alive=False)
    meta = ds.metadata
    dims = meta.pitch_dimensions
    length = float(getattr(dims, "pitch_length", None) or 105.0)
    width = float(getattr(dims, "pitch_width", None) or 68.0)
    home, away = meta.teams[0], meta.teams[1]
    players = [p for t in (home, away) for p in t.players]
    slot = {p.player_id: i for i, p in enumerate(players)}
    team = np.array([0] * len(home.players) + [1] * len(away.players), dtype=np.int8)

    # Coordinates: kloppy's default system is normalised 0-1 on both axes. Rescale with the
    # real pitch size; check the range so a change of default cannot go unnoticed.
    n = len(ds.records)
    P = np.full((n, len(players), 2), np.nan, dtype=np.float32)
    B = np.full((n, 2), np.nan, dtype=np.float32)
    alive = np.zeros(n, dtype=bool)
    owner = np.full(n, -1, dtype=np.int8)
    period = np.zeros(n, dtype=np.int8)
    for i, fr in enumerate(ds.records):
        period[i] = fr.period.id
        if fr.ball_coordinates is not None:
            B[i] = (fr.ball_coordinates.x, fr.ball_coordinates.y)
        alive[i] = fr.ball_state is not None and fr.ball_state.value == "alive"
        if fr.ball_owning_team is not None:
            owner[i] = 0 if fr.ball_owning_team.team_id == home.team_id else 1
        for player, data in fr.players_data.items():
            if data.coordinates is not None and player.player_id in slot:
                P[i, slot[player.player_id]] = (data.coordinates.x, data.coordinates.y)

    span = np.nanmax(np.abs(np.concatenate([P[..., 0].ravel(), B[:, 0]])))
    if span <= 1.5:                       # normalised -> metres
        P[..., 0] *= length
        P[..., 1] *= width
        B[:, 0] *= length
        B[:, 1] *= width
    else:
        raise SystemExit("unexpected coordinate range ({}); tell Claude".format(span))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "idsse_{}.npz".format(match_id))
    np.savez_compressed(path, players=P, team=team, ball=B, alive=alive, owner=owner,
                        period=period, pitch=np.array([length, width]))
    print("{}  {} frames  alive {:.0%}  home owns {:.0%} of alive  ball seen {:.0%}  -> {}"
          .format(match_id, n, alive.mean(), (owner[alive] == 0).mean(),
                  (~np.isnan(B[:, 0])).mean(), os.path.relpath(path)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", nargs="*", default=list(MATCHES))
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    for match_id in args.matches:
        if os.path.exists(os.path.join(args.out, "idsse_{}.npz".format(match_id))):
            print(match_id, "already done")
            continue
        convert(match_id, args.out)


if __name__ == "__main__":
    main()
