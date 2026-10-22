"""Claims a rebuild must pass, the per-claim comparison, and the motion recipe builders read.

claims   fits.json -> claims.json. Every claim is a visible quantity with a tolerance:
           time   t10/t50/t90/t95 of a move; peak time, half-way up and down of a pulse   (± TOL_FRAMES frames)
           value  start and end levels, overshoot, a pulse's base and peak               (± VALUE_TOL of the travel)
           ease   the ease family, only when its BIC win is clear (fit.EASE_GAP); otherwise it is listed under
                  not_claimed, never claimed as a guess
compare  his claims against a rebuild's (measured by `replay`, so by the same steps): per claim the delta (in frames
         for times) and pass or fail. A move's t10-t95 are timed at his levels on the rebuild's readings (when ours
         reaches the value his reached at that point), so its end level is judged once, by its own claim. A claim the rebuild can't answer is unknown, and unknown never passes. Ready
         only when every claim passes. The result is written to the rebuild's compare.json and listed in his
         effect.json `compares`.
recipe   the build-facing summary of an effect: each move's ease (or its nearest family, marked unproven), fitted
         start and settle times, levels and checkpoints; each pulse's levels, times and simplified keyframes; the
         camera's rate; and a `manual` block for numbers measured by hand (bead spacing) that rewrites keep. A
         builder reads recipe.json by path (and can pin its sha256), so
         numbers are never copied by hand and a changed recipe shows up at validation.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

import numpy as np

from . import fit as fitmod
from . import series
from .study import Effect

TOL_FRAMES = 1.5
VALUE_TOL = 0.02


def claims(effect: Effect, tol_frames: float = TOL_FRAMES, value_tol: float = VALUE_TOL) -> dict:
    fits = effect.read("fits.json", {})
    if not fits:
        raise SystemExit(f"{effect.root.name}: no fits yet; run `fit` on each quantity first")
    fps = effect.meta().get("fps") or 30.0
    tt = round(tol_frames / fps, 4)
    out, skipped = [], []

    def add(q, what, kind, value, tol, basis):
        if value is None:
            skipped.append({"id": f"{q}.{what}", "why": "not measured (no crossing in the window)"})
            return
        out.append({"id": f"{q}.{what}", "quantity": q, "what": what, "kind": kind, "value": value, "tol": tol,
                    "basis": basis})

    for q, f in fits.items():
        if f["kind"] == "move":
            travel = abs(f["end"] - f["start"])
            for k in ("t10", "t50", "t90", "t95"):
                add(q, k, "time", f[k], tt, "observed")
            add(q, "start", "value", f["start"], round(max(value_tol * travel, 1e-4), 4), "observed")
            add(q, "end", "value", f["end"], round(max(value_tol * travel, 1e-4), 4), "observed")
            add(q, "overshoot", "value", f["overshoot"], 0.02, "derived")
            if f["ease"] != "unknown":
                add(q, "ease", "ease", f["ease"], None, "derived")
            else:
                skipped.append({"id": f"{q}.ease", "why": f"best {f['best']} beat {f['next']} by only {f['bic_gap']} "
                                                         f"BIC (needs {fitmod.EASE_GAP:g})"})
        else:
            amp = abs(f["peak"] - f["base"])
            for k in ("peak_t", "rise_t50", "fall_t50"):
                add(q, k, "time", f[k], tt, "observed")
            for k in ("base", "peak"):
                add(q, k, "value", f[k], round(max(0.05 * amp, 1e-4), 4), "observed")
    res = {"effect": effect.root.name, "made": datetime.now().isoformat(timespec="seconds"), "fps": fps,
           "tol_frames": tol_frames, "value_tol": value_tol, "claims": out, "not_claimed": skipped}
    effect.write("claims.json", res)
    return {"claims": len(out), "not_claimed": skipped}


def compare(his: Effect, reb: Effect) -> dict:
    hc, rc = his.read("claims.json"), reb.read("claims.json")
    if not hc:
        raise SystemExit(f"{his.root.name}: no claims.json; run `claims` on it first")
    if not rc:
        raise SystemExit(f"{reb.root.name}: no claims.json; run `replay {his.root} --on {reb.root}` first")
    fps = hc["fps"]
    rmap = {c["id"]: c for c in rc["claims"]}
    hfits = his.read("fits.json", {})
    rows, passed = [], 0
    for c in hc["claims"]:
        r = rmap.get(c["id"])
        f = hfits.get(c["quantity"], {})
        if c["kind"] == "time" and f.get("kind") == "move" and c["what"] in ("t10", "t50", "t90", "t95"):
            # a move's checkpoint is timed at HIS level (start + q x his travel), on the rebuild's own readings: when
            # does ours reach the pixel his reached at q? Normalising each clip by its own end let a few px of
            # difference at the end (a rope hidden under a badge 16 px early) drag every later checkpoint by frames,
            # one look difference reported as four timing failures (effect A, 2026-10-21). The end is its own claim.
            try:
                t_, v_, _ = series.get(reb, c["quantity"])
                ok_ = ~np.isnan(v_)
                lvl = int(c["what"][1:]) / 100
                when = fitmod._crossing(t_[ok_], (v_[ok_] - f["start"]) / (f["end"] - f["start"]), lvl)
                r = {"value": None if when is None else round(when, 4)}
            except SystemExit:
                r = None
        row = {"id": c["id"], "kind": c["kind"], "his": c["value"], "ours": None if r is None else r["value"]}
        if r is None or r["value"] is None:
            row.update(result="unknown", why="the rebuild has no reading for this claim")
        elif c["kind"] == "ease":
            row["result"] = "pass" if r["value"] == c["value"] else "fail"
        else:
            d = float(r["value"]) - float(c["value"])
            row["delta"] = round(d, 4)
            if c["kind"] == "time":
                row["delta_frames"] = round(d * fps, 2)
            row["result"] = "pass" if abs(d) <= c["tol"] + 1e-9 else "fail"
        passed += row["result"] == "pass"
        rows.append(row)
    for s in rc.get("not_claimed", []):
        if s["id"] in {c["id"] for c in hc["claims"]}:
            next(r for r in rows if r["id"] == s["id"]).setdefault("why", s["why"])
    res = {"his": his.root.name, "rebuild": reb.root.name, "at": datetime.now().isoformat(timespec="seconds"),
           "passed": passed, "of": len(rows), "ready": passed == len(rows) and len(rows) > 0, "rows": rows}
    reb.write("compare.json", res)
    meta = his.meta()
    meta["compares"] = [c for c in meta.get("compares", []) if c["rebuild"] != reb.root.name] + [
        {k: res[k] for k in ("rebuild", "at", "passed", "of", "ready")}]
    his.save_meta(meta)
    return {k: res[k] for k in ("rebuild", "passed", "of", "ready")} | {
        "failed": [{k: r.get(k) for k in ("id", "his", "ours", "delta_frames", "delta", "why")}
                   for r in rows if r["result"] != "pass"]}


def _rdp(pts, eps):
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    nrm = np.hypot(*ab) or 1.0
    d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / nrm
    i = int(np.argmax(d))
    if d[i] <= eps:
        return np.array([a, b])
    return np.vstack([_rdp(pts[: i + 1], eps)[:-1], _rdp(pts[i:], eps)])


def recipe(effect: Effect, sets=()) -> dict:
    fits = effect.read("fits.json", {})
    old = effect.read("recipe.json", {})
    manual = dict(old.get("manual", {}))
    for kv in sets:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass
        manual[k] = v
    moves, pulses = [], []
    steps = effect.meta().get("pipeline") or []

    def measured_by(q):  # the step that made the quantity, so a builder knows what its units mean
        base = q.split(".")[0]
        s_ = next((s for s in steps if s["cmd"] in ("pixels", "track", "camera") and
                   (s.get("name") == base or (s["cmd"] == "camera" and base == "camera"))), None)
        return None if s_ is None else " ".join([s_["cmd"]] + s_.get("argv", []))

    for q, f in fits.items():
        if f["kind"] == "move":
            b = f["fits"][f["best"]]
            moves.append({"quantity": q, "ease": f["ease"], "nearest_ease": f["best"], "bic_gap": f["bic_gap"],
                          "start_s": b["t0"], "settle_s": b["t1"], "from": f["start"], "to": f["end"],
                          "unit": f["unit"], **{k: f[k] for k in ("t10", "t50", "t90", "t95", "overshoot")},
                          "measured_by": measured_by(q)})
        else:
            t, v, _ = series.get(effect, q)
            ok = ~np.isnan(v)
            amp = (f["peak"] - f["base"]) or 1.0
            keys = _rdp(np.c_[t[ok], (v[ok] - f["base"]) / amp], 0.04)
            pulses.append({"quantity": q, "measured_by": measured_by(q),
                           **{k: f[k] for k in ("base", "peak", "peak_t", "rise_t50", "fall_t50")},
                           "keys": [[round(float(a), 4), round(float(b), 3)] for a, b in keys]})
    cam = (effect.read("camera.json") or {}).get("summary")
    cmp_ = (effect.meta().get("compares") or [])
    claims_p = effect.root / "claims.json"
    res = {"effect": effect.root.name, "source": effect.meta().get("source", {}).get("public_url"),
           "made": datetime.now().isoformat(timespec="seconds"), "fps": effect.meta().get("fps"),
           "times": "seconds from the effect window's first frame",
           "ready": any(c["ready"] for c in cmp_), "compares": cmp_,
           "claims_sha256": hashlib.sha256(claims_p.read_bytes()).hexdigest() if claims_p.exists() else None,
           "camera": cam, "moves": moves, "pulses": pulses, "manual": manual}
    effect.write("recipe.json", res)
    return {"recipe": str(effect.root / "recipe.json"), "moves": len(moves), "pulses": len(pulses),
            "camera": bool(cam), "ready": res["ready"], "manual": sorted(manual)}
