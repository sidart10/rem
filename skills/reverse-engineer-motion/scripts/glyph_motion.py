#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "opencv-python-headless>=4.9", "scipy>=1.11"]
# ///
"""Kinetic-type measuring for reverse-engineer-motion: letters keyed locally, per-letter moves, beats and claims.

  glyph_motion.py key <effect> --rule "luma>110" [--rule ..] [--name type] [--min-area 30] [--close 0] [--box x0,y0,x1,y1]
  glyph_motion.py alpha <effect> --rgba <render.webm|png dir> [--name rebuild] [--min-area 30] [--close 0] [--scale 0.5]
  glyph_motion.py moves <effect> [--still 0.04] [--min-travel 0.15] [--min-rows 8]
  glyph_motion.py arrivals <effect> [--tol 0.2]          # letters landing in a word: when each slot settles
  glyph_motion.py rows <effect> --rule "sat>40" [--box ..]  # stacked echoes: rows per frame, height, pitch
  glyph_motion.py streaks <effect> --cap 25                # flight trails keyed apart (sat>45): per flight
  glyph_motion.py compare-streaks <his effect> <our effect>
  glyph_motion.py compare <his effect> <our effect>

Between key/alpha and moves, run `motion_re.py run <effect>` (measure, link, sheet), so letters get identities with
the same linker as every other effect. Free and local: nothing is sent anywhere. See motionre/glyphs.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

from motionre import glyphs  # noqa: E402
from motionre.study import Effect  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("key")
    k.add_argument("effect")
    k.add_argument("--rule", action="append", required=True)
    k.add_argument("--name", default="type")
    k.add_argument("--min-area", type=int, default=30)
    k.add_argument("--close", type=int, default=0)
    k.add_argument("--box")
    a = sub.add_parser("alpha")
    a.add_argument("effect")
    a.add_argument("--rgba", required=True)
    a.add_argument("--name", default="rebuild")
    a.add_argument("--min-area", type=int, default=30)
    a.add_argument("--close", type=int, default=0)
    a.add_argument("--scale", type=float)
    m = sub.add_parser("moves")
    m.add_argument("effect")
    m.add_argument("--still", type=float, default=0.04)
    m.add_argument("--min-travel", type=float, default=0.15)
    m.add_argument("--min-rows", type=int, default=8)
    ar = sub.add_parser("arrivals")
    ar.add_argument("effect")
    ar.add_argument("--tol", type=float, default=0.2)
    ro = sub.add_parser("rows")
    ro.add_argument("effect")
    ro.add_argument("--rule", action="append", required=True)
    ro.add_argument("--min-px", type=int, default=40)
    ro.add_argument("--box")
    sk = sub.add_parser("streaks")
    sk.add_argument("effect")
    sk.add_argument("--cap", type=float, required=True, help="cap height in px (from the letters' arrivals)")
    cs = sub.add_parser("compare-streaks")
    cs.add_argument("his")
    cs.add_argument("ours")
    c = sub.add_parser("compare")
    c.add_argument("his")
    c.add_argument("ours")
    c.add_argument("--claim", action="append")
    x = ap.parse_args(argv)
    if x.cmd == "key":
        box = [float(v) for v in x.box.split(",")] if x.box else None
        r = glyphs.key(Effect(Path(x.effect)), x.rule, x.name, x.min_area, x.close, box)
    elif x.cmd == "alpha":
        r = glyphs.alpha(Effect(Path(x.effect)), Path(x.rgba), x.name, x.min_area, x.close, x.scale)
    elif x.cmd == "moves":
        r = glyphs.moves(Effect(Path(x.effect)), x.still, x.min_travel, x.min_rows)
    elif x.cmd == "arrivals":
        r = glyphs.arrivals(Effect(Path(x.effect)), x.tol)
    elif x.cmd == "rows":
        r = glyphs.rows(Effect(Path(x.effect)), x.rule, x.min_px, box=[float(v) for v in x.box.split(",")] if x.box else None)
    elif x.cmd == "streaks":
        r = glyphs.streaks(Effect(Path(x.effect)), x.cap)
    elif x.cmd == "compare-streaks":
        r = glyphs.compare_streaks(Effect(Path(x.his)), Effect(Path(x.ours)))
    else:
        r = glyphs.compare(Effect(Path(x.his)), Effect(Path(x.ours)), x.claim)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
