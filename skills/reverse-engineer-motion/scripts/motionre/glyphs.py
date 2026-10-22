"""Kinetic type: letters as elements. Local, free masks for type on a flat field, and per-letter move statistics.

Built for kinetic-type looks (letters on black, letters assembling into a word). SAM 3
is a poor fit for single letters (dozens of tiny elements per frame, no prompt that means "each glyph"), and type on a
flat background separates cleanly by pixel rule, so letters are keyed locally, at no cost.

key      masks from a pixel rule on his frames (pixels.py rules, ANDed: "luma>110", or "luma<150" for dark type on a
         light field, "sat>60" for coloured echoes), split into connected parts, written in the same format as the
         sam3 and alpha providers, so `measure`, `link` and `sheet` run unchanged.
alpha    the same split on the alpha channel of our RGBA render, with the same --close and --min-area, so his letters
         and ours are cut into elements by identical code (the rule this skill lives by).
moves    tracks.json -> glyph-moves.json. Each letter track is split into holds and moves (a move is a run of frames
         whose step is over `still` cap heights, padded by one frame each side). Per move, the visible checkpoints
         read off the data along the move's own direction: t10/t50/t90, travel (px and cap heights), dx, dy, height
         change, rotation change, overshoot. Moves that overlap in time across letters form a beat; per beat: the
         spread of t50 (stagger), the order letters move in, and the beat's span. Between beats: hold length and
         the cycle period. The ease is fitted per family on every move's progress at once (one shared shape,
         per-move start and end), ranked by BIC; it is named only when the win is clear (fit.EASE_GAP).
compare  his glyph-moves.json against a rebuild's: per statistic, the delta and pass or fail, unknown never passing.

Every number carries its basis: observed (read off pixels), derived (computed from observed values).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import fit as fitmod
from . import pixels
from .masks import source_dir
from .study import Effect


def _split(binary: np.ndarray, out: Path, i: int, source: str, min_area: int, close: int) -> int:
    b = binary.astype(np.uint8)
    if close > 0:
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    k, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    rec = {"frame": i, "source": source, "prompt": None, "scores": [], "masks": []}
    for c in range(1, k):
        if st[c, cv2.CC_STAT_AREA] < min_area:
            continue
        m = f"f{i:04d}--m{len(rec['masks'])}.png"
        cv2.imwrite(str(out / m), ((lab == c) * 255).astype(np.uint8))
        rec["masks"].append(m)
        rec["scores"].append(1.0)
    (out / f"f{i:04d}.json").write_text(json.dumps(rec))
    return len(rec["masks"])


def key(effect: Effect, rules, name: str = "type", min_area: int = 30, close: int = 0, box=None) -> dict:
    """Local masks from pixel rules on the effect's own frames. box (x0,y0,x1,y1) limits where letters are looked for."""
    parsed = [pixels.parse_rule(r) for r in rules]
    out = source_dir(effect, f"key:{name}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    counts = []
    for fr in effect.frames():
        img = cv2.imread(str(effect.frame_png(fr["i"])))
        ch = pixels.channels(img)
        b = np.ones(img.shape[:2], bool)
        for d, c, op, x in parsed:
            if d:
                raise SystemExit("key rules read this frame only; difference rules (d...) belong to `pixels`")
            b &= pixels._cmp(ch[c], op, x)
        if box:
            x0, y0, x1, y1 = [int(v) for v in box]
            keep = np.zeros_like(b)
            keep[y0:y1, x0:x1] = True
            b &= keep
        counts.append(_split(b, out, fr["i"], f"key:{name}", min_area, close))
    meta = effect.meta()
    meta.setdefault("glyph_masks", {})[f"key:{name}"] = {"rules": list(rules), "min_area": min_area, "close": close,
                                                         "box": box}
    effect.save_meta(meta)
    return {"source": f"key:{name}", "frames": len(counts), "parts_per_frame": [min(counts), int(np.median(counts)),
                                                                                  max(counts)], "usd": 0}


def alpha(effect: Effect, rgba: Path, name: str = "rebuild", min_area: int = 30, close: int = 0,
          scale: float | None = None) -> dict:
    """Split our RGBA render's alpha into parts with the same rule as `key`. scale resizes first (match his size)."""
    out = source_dir(effect, f"alpha:{name}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    n = len(effect.frames())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if rgba.is_dir():
            pngs = sorted(rgba.glob("*.png"))
        else:
            vf = "alphaextract" if not scale else f"scale=iw*{scale}:ih*{scale}:flags=area,alphaextract"
            dec = ["-c:v", "libvpx-vp9"] if rgba.suffix == ".webm" else []  # ffmpeg's default vp9 decoder drops alpha
            subprocess.run(["ffmpeg", "-v", "error", "-y", *dec, "-i", str(rgba), "-frames:v", str(n),
                            "-vf", vf, "-start_number", "0", str(tmp / "a%04d.png")], check=True)
            pngs = sorted(tmp.glob("a*.png"))
        counts = []
        for i, p in enumerate(pngs[:n]):
            a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if a.ndim == 3:
                a = a[..., 3] if a.shape[2] == 4 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            counts.append(_split(a > 127, out, i, f"alpha:{name}", min_area, close))
    return {"source": f"alpha:{name}", "frames": len(counts), "parts_per_frame": [min(counts), int(np.median(counts)),
                                                                                    max(counts)], "usd": 0}


# ---------------------------------------------------------------- moves

def _cross(t, prog, q):
    return fitmod._crossing(np.asarray(t), np.asarray(prog), q)


def _segments(moving: np.ndarray, gap: int = 1):
    segs, i, n = [], 0, len(moving)
    while i < n:
        if not moving[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (moving[j + 1] or (j + 1 + gap < n and moving[j + 1: j + 2 + gap].any())):
            j += 1
        segs.append((i, j))
        i = j + 1
    return segs


def moves(effect: Effect, still: float = 0.04, min_travel: float = 0.15, min_rows: int = 8, area_min: int = 0) -> dict:
    tracks = [t for t in effect.read("tracks.json", {}).get("tracks", [])
              if len(t["rows"]) >= min_rows and not t.get("flicker")]
    if not tracks:
        raise SystemExit(f"{effect.root.name}: no tracks; run `motion_re.py run` first")
    fr = effect.frames()
    t_all = np.array([r["t"] - fr[0]["t"] for r in fr])
    fps = effect.meta().get("fps") or 30.0
    cap = float(np.median([np.median([r["h"] for r in t["rows"]]) for t in tracks]))
    out_moves, letters = [], []
    for k, tr in enumerate(tracks):
        rows = sorted(tr["rows"], key=lambda r: r["f"])
        if area_min and np.median([r["area"] for r in rows]) < area_min:
            continue
        f = np.array([r["f"] for r in rows])
        xy = np.array([[r["cx"], r["cy"]] for r in rows], float)
        h = np.array([r["h"] for r in rows], float)
        ang = np.array([(r.get("rot") or [0, 0, 0, 0, 0])[4] for r in rows], float)
        t = t_all[f]
        step = np.r_[0, np.hypot(*np.diff(xy, axis=0).T)] / cap
        dstep = np.r_[0, np.abs(np.diff(h))] / cap
        mov = (step > still) | (dstep > still)
        letters.append({"track": k, "cx": round(float(np.median(xy[:, 0])), 1), "cy": round(float(np.median(xy[:, 1])), 1),
                        "h": round(float(np.median(h)), 1), "frames": [int(f[0]), int(f[-1])]})
        for a, b in _segments(mov):
            s, e = max(0, a - 1), min(len(f) - 1, b)
            if e - s < 2:
                continue
            d = xy[e] - xy[s]
            travel = float(np.hypot(*d))
            dh = float(h[e] - h[s])
            if travel / cap < min_travel and abs(dh) / cap < min_travel:
                continue
            if travel / cap >= min_travel:
                u = d / travel
                prog = ((xy[s: e + 1] - xy[s]) @ u) / travel
                axis = "position"
            else:
                prog = (h[s: e + 1] - h[s]) / dh
                axis = "height"
            tt = t[s: e + 1]
            t10, t50, t90 = (_cross(tt, prog, q) for q in (0.1, 0.5, 0.9))
            out_moves.append({
                "track": k, "letter_at": letters[-1]["cx"], "line_at": letters[-1]["cy"], "axis": axis,
                "frames": [int(f[s]), int(f[e])], "t_start": round(float(tt[0]), 4), "t_end": round(float(tt[-1]), 4),
                "t10": None if t10 is None else round(t10, 4), "t50": None if t50 is None else round(t50, 4),
                "t90": None if t90 is None else round(t90, 4),
                "dur_10_90": None if None in (t10, t90) else round(t90 - t10, 4),
                "travel_px": round(travel, 1), "travel_cap": round(travel / cap, 3),
                "dx_cap": round(float(d[0]) / cap, 3), "dy_cap": round(float(d[1]) / cap, 3),
                "dh_cap": round(dh / cap, 3), "drot_deg": round(float(ang[e] - ang[s]), 1),
                "overshoot": round(float(max(0.0, prog.max() - 1.0)), 3),
                "_t": [float(x) for x in tt], "_p": [float(x) for x in prog]})
    beats = _beats(out_moves, fps)
    ease = _pooled_ease(out_moves, fps)
    for m in out_moves:
        m.pop("_t"), m.pop("_p")
    res = {"effect": effect.root.name, "fps": fps, "cap_px": round(cap, 1), "params": {"still": still,
           "min_travel": min_travel, "min_rows": min_rows}, "letters": len(letters), "moves": out_moves,
           "beats": beats, "ease": ease, "summary": _summary(out_moves, beats, ease, fps),
           "basis": {"t10/t50/t90, travel, dx, dy, dh": "observed (mask centroids and boxes, per frame)",
                     "beats, stagger, hold, period, ease": "derived"}}
    effect.write("glyph-moves.json", res)
    return res["summary"]


def _beats(ms, fps):
    pts = sorted([m for m in ms if m["t50"] is not None], key=lambda m: m["t50"])
    groups, cur = [], []
    for m in pts:
        if cur and m["t_start"] > max(c["t_end"] for c in cur):
            groups.append(cur)
            cur = []
        cur.append(m)
    if cur:
        groups.append(cur)
    beats = []
    for g in groups:
        t50 = np.array([m["t50"] for m in g])
        order = sorted(g, key=lambda m: m["t50"])
        xs = np.array([m["letter_at"] for m in order])
        ys = np.array([m["line_at"] for m in order])
        rho_x = float(np.corrcoef(np.arange(len(xs)), xs)[0, 1]) if len(g) > 2 and xs.std() > 0 else None
        rho_y = float(np.corrcoef(np.arange(len(ys)), ys)[0, 1]) if len(g) > 2 and ys.std() > 0 else None
        diffs = np.diff(np.sort(t50))
        beats.append({"letters": len(g), "t_start": round(min(m["t_start"] for m in g), 4),
                      "t_end": round(max(m["t_end"] for m in g), 4),
                      "t50_first": round(float(t50.min()), 4), "t50_last": round(float(t50.max()), 4),
                      "t50_spread": round(float(t50.max() - t50.min()), 4),
                      "stagger_median": round(float(np.median(diffs)), 4) if len(diffs) else None,
                      "dur_10_90_median": round(float(np.median([m["dur_10_90"] for m in g if m["dur_10_90"]] or [np.nan])), 4),
                      "travel_cap_median": round(float(np.median([m["travel_cap"] for m in g])), 3),
                      "order_vs_x": None if rho_x is None else round(rho_x, 2),
                      "order_vs_y": None if rho_y is None else round(rho_y, 2)})
    return beats


def _pooled_ease(ms, fps):
    usable = [m for m in ms if len(m["_t"]) >= 4]
    if len(usable) < 3:
        return {"ease": "unknown", "why": "fewer than 3 moves with 4+ frames"}
    step = 0.5 / fps
    tot = {}
    n = sum(len(m["_t"]) for m in usable)
    for fam, f in fitmod.EASES.items():
        rss_sum = 0.0
        for m in usable:
            t, p = np.array(m["_t"]), np.array(m["_p"])
            best = np.inf
            for t0 in np.arange(t[0] - 2 * step, t[-1], step):
                t1s = np.arange(t0 + 2 / fps, t[-1] + 4 * step, step)
                if not len(t1s):
                    continue
                u = np.clip((t[None, :] - t0) / (t1s[:, None] - t0), 0, 1)
                best = min(best, float(((f(u) - p[None, :]) ** 2).sum(axis=1).min()))
            rss_sum += best
        tot[fam] = round(n * np.log(max(rss_sum, 1e-12) / n), 1)
    ranked = sorted(tot, key=tot.get)
    gap = tot[ranked[1]] - tot[ranked[0]]
    return {"ease": ranked[0] if gap >= fitmod.EASE_GAP else "unknown", "best": ranked[0], "next": ranked[1],
            "bic_gap": round(gap, 1), "moves": len(usable), "points": n,
            "top5": {k: tot[k] for k in ranked[:5]}, "basis": "derived (pooled fit, shared shape per family)"}


def _summary(ms, beats, ease, fps):
    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.median(xs)), 4) if xs else None
    c = [b["t50_first"] for b in beats if b["letters"] >= 3]
    holds = [beats[i + 1]["t_start"] - beats[i]["t_end"] for i in range(len(beats) - 1)
             if beats[i]["letters"] >= 3 and beats[i + 1]["letters"] >= 3]
    return {"moves": len(ms), "beats": len(beats), "frame_s": round(1 / fps, 4),
            "move_dur_10_90_s": med([m["dur_10_90"] for m in ms]),
            "move_travel_cap": med([m["travel_cap"] for m in ms]),
            "move_abs_dy_cap": med([abs(m["dy_cap"]) for m in ms]),
            "move_overshoot": med([m["overshoot"] for m in ms]),
            "beat_t50_spread_s": med([b["t50_spread"] for b in beats if b["letters"] >= 3]),
            "beat_stagger_s": med([b["stagger_median"] for b in beats if b["letters"] >= 3]),
            "hold_s": med(holds), "period_s": med(list(np.diff(c))) if len(c) > 1 else None,
            "ease": ease.get("ease"), "ease_best": ease.get("best"), "ease_bic_gap": ease.get("bic_gap")}


CLAIM_TOL = {"move_dur_10_90_s": ("time", 1.5), "beat_t50_spread_s": ("time", 1.5), "beat_stagger_s": ("time", 1.0),
             "hold_s": ("time", 1.5), "period_s": ("time", 1.5), "move_travel_cap": ("ratio", 0.25),
             "move_abs_dy_cap": ("ratio", 0.25), "move_overshoot": ("abs", 0.05)}


def compare(his: Effect, ours: Effect, keys=None) -> dict:
    a, b = his.read("glyph-moves.json"), ours.read("glyph-moves.json")
    if not a or not b:
        raise SystemExit("run `glyph_motion.py moves` on both effects first")
    fps = a["fps"]
    rows, ok = [], True
    for k, (kind, tol) in CLAIM_TOL.items():
        if keys and k not in keys:
            continue
        x, y = a["summary"].get(k), b["summary"].get(k)
        if x is None or y is None:
            rows.append({"claim": k, "his": x, "ours": y, "result": "unknown"})
            ok = False
            continue
        if kind == "time":
            delta = round((y - x) * fps, 2)
            res = abs(delta) <= tol
            rows.append({"claim": k, "his": x, "ours": y, "delta_frames": delta, "tol_frames": tol,
                         "result": "pass" if res else "fail"})
        elif kind == "ratio":
            r = y / x if x else None
            res = r is not None and abs(r - 1) <= tol
            rows.append({"claim": k, "his": x, "ours": y, "ratio": None if r is None else round(r, 3),
                         "result": "pass" if res else "fail"})
        else:
            res = abs(y - x) <= tol
            rows.append({"claim": k, "his": x, "ours": y, "result": "pass" if res else "fail"})
        ok &= res
    if a["summary"].get("ease") not in (None, "unknown") and (keys is None or "ease" in keys):
        e = b["summary"].get("ease_best")
        res = e == a["summary"]["ease"]
        rows.append({"claim": "ease", "his": a["summary"]["ease"], "ours": e, "result": "pass" if res else "fail"})
        ok &= res
    out = {"his": his.root.name, "ours": ours.root.name, "ready": bool(ok), "claims": rows}
    ours.write("glyph-compare.json", out)
    return out


# ---------------------------------------------------------------- arrivals (letters assembling into a word)

def arrivals(effect: Effect, tol: float = 0.2, last: int = 3) -> dict:
    """When each letter lands in its final slot. Slots are the parts of the last `last` frames; a slot is settled from
    the first frame after which a part of similar height stays within `tol` cap heights of it to the end. Robust where
    the linker loses a letter in flight (motion blur): only the landing needs to be seen. For each landed letter the
    track ending in the slot gives its flight path back as far as the linker held it (travel, direction, t50)."""
    det = effect.read("raw/detections.json")
    if not det:
        raise SystemExit("run `motion_re.py run` first")
    frs = det["frames"]
    t0 = frs[0]["t"]
    slots = [d for d in frs[-1]["dets"]]
    cap = float(np.median([d["h"] for d in slots]))
    tracks = effect.read("tracks.json", {}).get("tracks", [])
    out = []
    for s in sorted(slots, key=lambda d: d["cx"]):
        ok = []
        for fr in frs:
            hit = any(np.hypot(d["cx"] - s["cx"], d["cy"] - s["cy"]) < tol * cap and
                      abs(np.log(max(d["h"], 1) / max(s["h"], 1))) < 0.35 for d in fr["dets"])
            ok.append(hit)
        k = len(ok)
        while k > 0 and ok[k - 1]:
            k -= 1
        row = {"slot_cx": round(s["cx"], 1), "slot_cy": round(s["cy"], 1), "h": s["h"],
               "settled_frame": int(frs[k]["i"]) if k < len(frs) else None,
               "settled_t": round(frs[k]["t"] - t0, 4) if k < len(frs) else None}
        tr = next((t for t in tracks if t["rows"] and t["rows"][-1]["f"] == frs[-1]["i"] and
                   np.hypot(t["rows"][-1]["cx"] - s["cx"], t["rows"][-1]["cy"] - s["cy"]) < 2), None)
        if tr and len(tr["rows"]) > 3:
            r0 = tr["rows"][0]
            row.update({"track_from_frame": r0["f"], "from_cx": r0["cx"], "from_cy": r0["cy"],
                        "travel_cap_seen": round(float(np.hypot(r0["cx"] - s["cx"], r0["cy"] - s["cy"])) / cap, 2),
                        "h_from": r0["h"]})
        out.append(row)
    st = [r["settled_t"] for r in out if r["settled_t"] is not None]
    order = sorted([r for r in out if r["settled_t"] is not None], key=lambda r: r["settled_t"])
    rho = float(np.corrcoef(range(len(order)), [r["slot_cx"] for r in order])[0, 1]) if len(order) > 2 else None
    res = {"cap_px": round(cap, 1), "tol_cap": tol, "letters": out,
           "first_land_s": min(st) if st else None, "last_land_s": max(st) if st else None,
           "land_spread_s": round(max(st) - min(st), 4) if st else None,
           "land_gap_median_s": round(float(np.median(np.diff(sorted(st)))), 4) if len(st) > 1 else None,
           "order_vs_x": None if rho is None else round(rho, 2),
           "basis": "observed (part positions per frame); spread, gaps and order derived"}
    effect.write("glyph-arrivals.json", res)
    return {k: v for k, v in res.items() if k != "letters"}


# ---------------------------------------------------------------- rows (stacked echoes of a line)

def rows(effect: Effect, rules, min_px: int = 40, min_gap: int = 3, box=None) -> dict:
    """Per frame, the text rows that pass the pixel rules: a horizontal projection split at empty bands. Gives each
    row's centre, height and ink, so a word that repeats as a stack of echoes (Gemini "Brainstorm Ideas") is read as
    rows appearing (count over time), growing (height) and spreading (pitch between neighbours)."""
    parsed = [pixels.parse_rule(r) for r in rules]
    fr0 = effect.frames()
    out = []
    for fr in fr0:
        img = cv2.imread(str(effect.frame_png(fr["i"])))
        ch = pixels.channels(img)
        b = np.ones(img.shape[:2], bool)
        for d, c, op, x in parsed:
            b &= pixels._cmp(ch[c], op, x)
        if box:
            x0, y0, x1, y1 = [int(v) for v in box]
            k = np.zeros_like(b)
            k[y0:y1, x0:x1] = True
            b &= k
        prof = b.sum(1)
        on = prof > 2
        rs, y = [], 0
        while y < len(on):
            if not on[y]:
                y += 1
                continue
            e = y
            while e + 1 < len(on) and (on[e + 1] or on[e + 1: e + 1 + min_gap].any()):
                e += 1
            ink = int(prof[y:e + 1].sum())
            if ink >= min_px:
                yy = np.arange(y, e + 1)
                rs.append({"cy": round(float((yy * prof[y:e + 1]).sum() / ink), 1), "h": int(e - y + 1), "ink": ink})
            y = e + 1
        cys = [r["cy"] for r in rs]
        out.append({"i": fr["i"], "t": round(fr["t"] - fr0[0]["t"], 4), "n": len(rs), "rows": rs,
                    "pitch_median": round(float(np.median(np.diff(cys))), 1) if len(cys) > 1 else None})
    effect.write("glyph-rows.json", {"rules": list(rules), "frames": out,
                                     "basis": "observed (pixel projection per frame)"})
    return {"frames": len(out), "n_rows": [f["n"] for f in out], "pitch": [f["pitch_median"] for f in out]}


# ---------------------------------------------------------------- streaks (motion trails of flying letters)

def streaks(effect: Effect, cap: float, min_rows: int = 3) -> dict:
    """Trails keyed separately (e.g. sat>45: Gemini's coloured trails; letters are grey or white): each track of 3+
    frames is one flight. Per flight: frames visible, start, longest extent along its axis in cap heights."""
    tr = [t for t in effect.read("tracks.json", {}).get("tracks", []) if len(t["rows"]) >= min_rows]
    fr = effect.frames()
    t0 = fr[0]["t"]
    fps = effect.meta().get("fps") or 30.0
    fl = []
    for t in tr:
        r = t["rows"]
        dx, dy = abs(r[-1]["cx"] - r[0]["cx"]), abs(r[-1]["cy"] - r[0]["cy"])
        ext = max(max(x["w"] for x in r), max(x["h"] for x in r))
        fl.append({"from_frame": r[0]["f"], "frames": len(r), "start_t": round(fr[r[0]["f"]]["t"] - t0, 4),
                   "axis": "x" if dx >= dy else "y", "max_len_cap": round(ext / cap, 2)})
    st = sorted(f["start_t"] for f in fl)
    # link-free: trails attached to fast letters jump more than their own length per frame, so the linker splits
    # or merges them (2026-10-21: default gates gave 1-frame tracks on our render, relaxed gates merged his flights).
    # These read straight off each frame's parts and are the claims; linked flights are kept as notes.
    det = effect.read("raw/detections.json", {}).get("frames", [])
    ext, on = [], []
    for f in det:
        if f["dets"]:
            on.append(f["t"] - t0)
            ext += [max(d["w"], d["h"]) / cap for d in f["dets"]]
    res = {"cap_px": cap, "flights": fl, "summary": {
        "trail_first_s": round(on[0], 4) if on else None,
        "trail_window_s": round(on[-1] - on[0], 4) if on else None,
        "trail_frames": len(on),
        "trail_len_cap_p50": round(float(np.percentile(ext, 50)), 2) if ext else None,
        "trail_len_cap_p90": round(float(np.percentile(ext, 90)), 2) if ext else None,
        "linked_flights": len(fl),
        "flights": len(fl),
        "streak_frames": float(np.median([f["frames"] for f in fl])) if fl else None,
        "streak_s": round(float(np.median([f["frames"] for f in fl])) / fps, 4) if fl else None,
        "trail_len_cap": round(float(np.median([f["max_len_cap"] for f in fl])), 2) if fl else None,
        "flights_spread_s": round(st[-1] - st[0], 4) if st else None,
        "flight_gap_s": round(float(np.median(np.diff(st))), 4) if len(st) > 1 else None},
        "basis": "observed (keyed trail parts per frame, linked); medians derived"}
    effect.write("glyph-streaks.json", res)
    return res["summary"]


STREAK_TOL = {"trail_window_s": ("time", 3.0), "trail_len_cap_p50": ("ratio", 0.3),
              "trail_len_cap_p90": ("ratio", 0.3)}


def compare_streaks(his: Effect, ours: Effect) -> dict:
    a, b = his.read("glyph-streaks.json"), ours.read("glyph-streaks.json")
    if not a or not b:
        raise SystemExit("run `glyph_motion.py streaks` on both effects first")
    fps = his.meta().get("fps") or 30.0
    rows, ok = [], True
    for k, (kind, tol) in STREAK_TOL.items():
        x, y = a["summary"].get(k), b["summary"].get(k)
        if x is None or y is None:
            rows.append({"claim": k, "his": x, "ours": y, "result": "unknown"})
            ok = False
            continue
        if kind == "time":
            d = round((y - x) * fps, 2)
            res = abs(d) <= tol
            rows.append({"claim": k, "his": x, "ours": y, "delta_frames": d, "tol_frames": tol, "result": "pass" if res else "fail"})
        else:
            r = y / x if x else None
            res = r is not None and abs(r - 1) <= tol
            rows.append({"claim": k, "his": x, "ours": y, "ratio": None if r is None else round(r, 3), "result": "pass" if res else "fail"})
        ok &= res
    out = {"his": his.root.name, "ours": ours.root.name, "ready": bool(ok), "claims": rows}
    ours.write("glyph-compare-streaks.json", out)
    return out
