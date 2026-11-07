"""Build the data pack for tools/viewer.html: tracking projected onto the video + model evidence.

    python -m tools.export_viewer                 # -> cache/viewer_pack.bin (~40 MB)

Then open tools/viewer.html in Chrome/Edge (double-click), and load:
  * the tactical video   data/tactical/BAR-ATH_tactical.mp4
  * this pack            cache/viewer_pack.bin
  * one or more label CSVs (your submission, an older version, the public ground truth)

Everything stays on your computer: the page has no network code, and the pack is written to cache/
which is gitignored. Never commit it, never upload it anywhere -- it is derived from the licensed
data (frame-level positions) and the terms forbid redistributing that.

Pack layout: 8-byte magic "MTAHVW01", uint32 little-endian JSON length, the JSON header, padding
to 4 bytes, then the arrays described in header["arrays"] (name, dtype, shape, byte offset from the
start of the data section).
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, require  # noqa: E402
from possession import model, pipeline, tagging  # noqa: E402

MISSING = -32768


def homographies(path, n):
    """(n, 3, 3) float64 from the homography CSV, NaN where invalid. numpy only."""
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    frames = table["frame"].astype(int)
    cols = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    H = np.full((n, 3, 3), np.nan)
    M = np.column_stack([table[c] for c in cols]).reshape(-1, 3, 3)
    valid = table["valid"].astype(bool)
    keep = (frames >= 0) & (frames < n)
    H[frames[keep]] = np.where(valid[keep, None, None], M[keep], np.nan)
    return H


def project(H, x, y):
    """Per-frame projection of normalised pitch points. H (n,3,3), x/y (n,k) -> pixels (n,k)."""
    den = H[:, 2, 0, None] * x + H[:, 2, 1, None] * y + H[:, 2, 2, None]
    u = (H[:, 0, 0, None] * x + H[:, 0, 1, None] * y + H[:, 0, 2, None]) / den
    v = (H[:, 1, 0, None] * x + H[:, 1, 1, None] * y + H[:, 1, 2, None]) / den
    return u, v


def to_i16(a):
    out = np.where(np.isfinite(a), np.clip(np.round(a), -30000, 30000), MISSING)
    return out.astype(np.int16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=os.path.join(ROOT, "cache", "viewer_pack.bin"))
    parser.add_argument("--v1", action="store_true")
    args = parser.parse_args()

    weights = None if args.v1 or not os.path.exists(model.WEIGHTS) else model.WEIGHTS
    res = pipeline.infer(args.data_dir, weights_path=weights)
    n, pitch, atd = res["n"], res["pitch"], res["atd"]
    hom_path, tags_path = require(["homography", "smart_tagging"], GAME, args.data_dir)

    print("projecting tracking through the homography ...")
    H = homographies(hom_path, n)
    px = np.concatenate([atd["players"][:, :, 0], atd["ball"][:, :1]], axis=1) / pitch[0]
    py = np.concatenate([atd["players"][:, :, 1], atd["ball"][:, 1:2]], axis=1) / pitch[1]
    u, v = project(H, px.astype(np.float64), py.astype(np.float64))
    xy = np.stack([to_i16(u), to_i16(v)], axis=2)          # (n, slots + 1, 2), last = ball

    sig = lambda s: 1.0 / (1.0 + np.exp(-np.clip(s.astype(np.float64), -30, 30)))  # noqa: E731
    arrays = [
        ("xy", xy),
        ("p_home", np.round(sig(res["s_team"]) * 255).astype(np.uint8)),
        ("p_dead", np.round(sig(res["s_dead"]) * 255).astype(np.uint8)),
        ("ball_seen", res["features"]["ball_seen"].astype(np.uint8)),
    ]

    events = tagging.load(tags_path)
    header = {
        "game": GAME, "n_frames": int(n), "fps": 25, "n_slots": int(atd["players"].shape[1]),
        "team": atd["team"].astype(int).tolist(), "home": res["home"], "away": res["away"],
        "blocks": [list(map(int, b)) for b in res["blocks"]],
        "restarts": [[int(f), k] for f, k in res["restarts"]],
        "tags": [[e["start"], e["end"], e["code"], e["type"], e["team"]] for e in events],
        "emissions": "learned" if weights else "v1",
        "arrays": [],
    }
    offset = 0
    for name, arr in arrays:
        header["arrays"].append({"name": name, "dtype": str(arr.dtype), "shape": list(arr.shape),
                                 "offset": offset})
        offset += arr.nbytes
        offset += (-offset) % 4

    blob = json.dumps(header).encode("utf-8")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as handle:
        handle.write(b"MTAHVW01")
        handle.write(struct.pack("<I", len(blob)))
        handle.write(blob)
        handle.write(b"\0" * ((-(12 + len(blob))) % 4))
        for name, arr in arrays:
            raw = np.ascontiguousarray(arr).tobytes()
            handle.write(raw)
            handle.write(b"\0" * ((-len(raw)) % 4))
    print("-> {} ({:.1f} MB)".format(os.path.relpath(args.out, ROOT),
                                     os.path.getsize(args.out) / 1e6))
    print("open tools/viewer.html in Chrome or Edge and load the video, this pack and your CSVs")


if __name__ == "__main__":
    main()
