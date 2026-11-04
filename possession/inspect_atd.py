"""Describe the STRUCTURE of the ATD without printing any of its values.

    python -m possession.inspect_atd

Answers one question that matters a lot: does the automatic tracking carry extra ball channels
(height, ball state, ball owner...)? CHALLENGE.md says the ground truth comes from *corrected*
tracking's own "dead" and "owning" fields; if the automatic ATD has uncorrected versions of those
fields, they are the strongest feature available.

Output is counts, channel names and category sizes only, so it is safe to paste into a chat.
Never paste rows of the data itself anywhere (terms of use: no extracts to hosted LLMs).
"""

import collections
import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import require  # noqa: E402


def strip(tag):
    return tag.split("}")[-1]


def main():
    meta, raw = require(["atd_meta", "atd_raw"])
    root = ET.parse(meta).getroot()

    print("== XML structure (tags and attribute names, no values) ==")
    seen = collections.Counter()
    for elem in root.iter():
        seen[(strip(elem.tag), tuple(sorted(elem.attrib)))] += 1
    for (tag, attrs), count in sorted(seen.items()):
        print("  {:<28} x{:<5} attrs={}".format(tag, count, list(attrs)))

    print("\n== Channel / sensor ids (names only) ==")
    for elem in root.iter():
        if strip(elem.tag) in ("Channel", "Sensor", "BallChannel", "BallChannelRef",
                               "PlayerChannelRef", "DataSpec", "StringRegister", "SplitRegister"):
            name = elem.attrib.get("id") or elem.attrib.get("channelId") or \
                elem.attrib.get("playerChannelId") or elem.attrib.get("separator")
            print("  {:<18} {}".format(strip(elem.tag), name))

    print("\n== Raw file shape ==")
    group_sizes, ball_sizes = collections.Counter(), collections.Counter()
    ball_extra_kinds = collections.defaultdict(collections.Counter)
    n_lines = 0
    with open(raw) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            n_lines += 1
            if len(parts) != 3:
                group_sizes["bad line, parts={}".format(len(parts))] += 1
                continue
            group_sizes[len(parts[1].split(";"))] += 1
            ball = parts[2].split(",")
            ball_sizes[len(ball)] += 1
            for i, token in enumerate(ball[2:], start=2):
                try:
                    value = float(token)
                    kind = "nan" if value != value else (
                        "int" if value == int(value) else "float")
                except ValueError:
                    kind = "text"
                ball_extra_kinds[i][kind] += 1
    print("  lines:", n_lines)
    print("  player groups per line:", dict(group_sizes))
    print("  ball fields per line:", dict(ball_sizes))
    for i, kinds in sorted(ball_extra_kinds.items()):
        print("  ball field #{} value kinds: {}".format(i, dict(kinds)))
    print("\nIf a ball field is mostly 'int' with few distinct values, it is probably a state or an"
          "\nowner flag. Run `python -m possession.inspect_atd --distinct` to count distinct values.")

    if "--distinct" in sys.argv:
        distinct = collections.defaultdict(set)
        with open(raw) as handle:
            for line in handle:
                parts = line.rstrip("\n").split(":")
                if len(parts) == 3:
                    for i, token in enumerate(parts[2].split(",")[2:], start=2):
                        if len(distinct[i]) <= 20:
                            distinct[i].add(token)
        for i, values in sorted(distinct.items()):
            print("  ball field #{}: {} distinct{}".format(
                i, len(values), "+" if len(values) > 20 else ""))


if __name__ == "__main__":
    main()
