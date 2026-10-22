"""Measured quantities over time, one store per effect, and the record of how each was measured.

series.json holds every quantity the later stages fit and claim on:

  {"fps": 23.976, "quantities": {"camera.scale": {"t": [...], "v": [...], "unit": "x", "basis": "derived",
                                                   "step": "camera"}, ...}}

`t` is seconds from the window's first frame (so his clip and a rebuild line up), `v` holds None where a frame has no
reading. Every quantity names the pipeline step that made it.

effect.json `pipeline` lists the measuring steps in order, each with the arguments it ran with. `replay` runs the same
steps on a rebuild's effect folder, so his clip and ours are always measured by the same code with the same settings
(the rule the Johnny Harris loop proved: a rebuild measured any other way proves nothing).
"""
from __future__ import annotations

import numpy as np

from .study import Effect

TRACK_FIELDS = ("cx", "cy", "w", "h", "area", "rot")


def load(effect: Effect) -> dict:
    return effect.read("series.json", {"fps": effect.meta().get("fps"), "quantities": {}})


def put(effect: Effect, name: str, t, v, unit: str, basis: str, step: str, **extra):
    s = load(effect)
    s["fps"] = effect.meta().get("fps")
    s["quantities"][name] = {"t": [round(float(x), 4) for x in t],
                             "v": [None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)
                                   for x in v],
                             "unit": unit, "basis": basis, "step": step, **extra}
    effect.write("series.json", s)


def get(effect: Effect, name: str):
    q = load(effect)["quantities"].get(name)
    if q is None:
        raise SystemExit(f"{effect.root.name}: no quantity {name!r} in series.json (have: "
                         f"{', '.join(sorted(load(effect)['quantities'])) or 'none'})")
    t = np.array(q["t"], float)
    v = np.array([np.nan if x is None else x for x in q["v"]], float)
    return t, v, q


def rel_times(effect: Effect) -> list:
    fr = effect.frames()
    return [r["t"] - fr[0]["t"] for r in fr]


def record_step(effect: Effect, cmd: str, name: str, args: dict):
    """Add or replace (same cmd and name) one measuring step in effect.json, in run order."""
    meta = effect.meta()
    steps = [s for s in meta.get("pipeline", []) if not (s["cmd"] == cmd and s.get("name") == name)]
    steps.append({"cmd": cmd, "name": name, "args": {k: v for k, v in args.items() if v not in (None, [], False)}})
    meta["pipeline"] = steps
    effect.save_meta(meta)


def tracks_near(effect: Effect, x: float, y: float, within: float = 90.0, max_size: float = 1e9) -> dict:
    """Rows {frame: row} of every track whose median centre lies within `within` px of (x, y), merged.

    Selecting by place instead of track id is what lets a replay find the same element in a rebuild (Johnny Harris
    effect A: the badge tracks were picked this way, and SAM 3 split one badge into several tracklets)."""
    out = {}
    for tr in effect.read("tracks.json", {}).get("tracks", []):
        rows = tr["rows"]
        if len(rows) < 3 or tr.get("flicker"):
            continue
        if np.median([r["w"] for r in rows]) > max_size or np.median([r["h"] for r in rows]) > max_size:
            continue
        c = np.median([[r["cx"], r["cy"]] for r in rows], axis=0)
        if np.hypot(c[0] - x, c[1] - y) < within:
            for r in rows:
                out.setdefault(r["f"], r)
    return out


def from_track(effect: Effect, name: str, x: float, y: float, within: float, fields, max_size: float = 1e9) -> dict:
    rows = tracks_near(effect, x, y, within, max_size)
    if not rows:
        raise SystemExit(f"no track within {within:g} px of ({x:g}, {y:g}); check tracks-sheet.jpg")
    t = rel_times(effect)
    made = []
    for f_ in fields:
        if f_ not in TRACK_FIELDS:
            raise SystemExit(f"field {f_!r}: pick from {', '.join(TRACK_FIELDS)}")
        v = [rows[i][f_] if i in rows and rows[i].get(f_) is not None else None for i in range(len(t))]
        put(effect, f"{name}.{f_}", t, v, "deg" if f_ == "rot" else "px", "observed", "track",
            near=[x, y], frames=len(rows))
        made.append(f"{name}.{f_}")
    return {"quantities": made, "frames": len(rows), "of": len(t)}
