"""Fit a measured quantity: which GSAP ease it moves on, when, and its visible checkpoints.

move   the quantity goes from one level to another. Its progress (0 at the start level, 1 at the end level) is
       fitted with every ease family below, over a grid of start and end times at half-frame steps, and the families
       are ranked by BIC (n ln(RSS/n) + 2 ln n). The ease is claimable only when the best beats the next by at least
       EASE_GAP (Kass and Raftery's "very strong"; effect A's lines won by 42 and 70). Whatever the ease, the visible
       checkpoints are read straight off the data: t10/t50/t90/t95 (first crossing, interpolated), the start and end
       levels, and any overshoot past the end level.
pulse  the quantity rises from a base and falls back (a glow): base, peak, peak time, half-way up and half-way down.

Results go to fits.json under the quantity's name; `claims` turns them into the claim file.
"""
from __future__ import annotations

import numpy as np

from . import series
from .study import Effect

EASE_GAP = 10.0
PI = np.pi


def _in(p):
    return lambda u: u ** p


def _out(p):
    return lambda u: 1 - (1 - u) ** p


def _inout(p):
    return lambda u: np.where(u < 0.5, 2 ** (p - 1) * u ** p, 1 - (-2 * u + 2) ** p / 2)


EASES = {"none": lambda u: u}
for k, p in (("power1", 2), ("power2", 3), ("power3", 4), ("power4", 5)):
    EASES[f"{k}.in"], EASES[f"{k}.out"], EASES[f"{k}.inOut"] = _in(p), _out(p), _inout(p)
EASES.update({
    "sine.in": lambda u: 1 - np.cos(u * PI / 2), "sine.out": lambda u: np.sin(u * PI / 2),
    "sine.inOut": lambda u: -(np.cos(PI * u) - 1) / 2,
    "expo.in": lambda u: np.where(u == 0, 0, 2 ** (10 * u - 10)),
    "expo.out": lambda u: np.where(u == 1, 1, 1 - 2 ** (-10 * u)),
    "expo.inOut": lambda u: np.where(u < 0.5, 2 ** (20 * u - 10) / 2, (2 - 2 ** (-20 * u + 10)) / 2),
    "circ.in": lambda u: 1 - np.sqrt(1 - u ** 2), "circ.out": lambda u: np.sqrt(1 - (u - 1) ** 2),
})


def _level(v, idx):
    vals = v[idx]
    vals = vals[~np.isnan(vals)]
    return float(np.median(vals)) if vals.size else np.nan


def _crossing(t, prog, q):
    for i in range(1, len(prog)):
        if prog[i] >= q and prog[i - 1] < q:
            a, b = prog[i - 1], prog[i]
            return float(t[i - 1] + (q - a) / (b - a) * (t[i] - t[i - 1])) if b != a else float(t[i])
    return float(t[0]) if len(prog) and prog[0] >= q else None


def fit_move(effect: Effect, name: str, t_from=None, t_to=None, start=None, end=None, edge_frames=3) -> dict:
    t, v, q = series.get(effect, name)
    fps = effect.meta().get("fps") or 30.0
    sel = np.ones(len(t), bool)
    if t_from is not None:
        sel &= t >= t_from
    if t_to is not None:
        sel &= t <= t_to
    t, v = t[sel], v[sel]
    ok = ~np.isnan(v)
    t, v = t[ok], v[ok]
    if len(t) < 6:
        raise SystemExit(f"{name}: only {len(t)} readings in the window; widen it or fix the measure")
    lo = start if start is not None else _level(v, slice(0, edge_frames))
    hi = end if end is not None else _level(v, slice(-edge_frames, None))
    if hi == lo:
        raise SystemExit(f"{name}: the start and end levels are equal ({lo:g}); it doesn't move in this window")
    prog = (v - lo) / (hi - lo)
    n, step = len(t), 0.5 / fps
    T0 = np.arange(t[0] - step, t[-1], step)
    fits = {}
    for fam, f in EASES.items():
        best = None
        for t0 in T0:
            t1s = np.arange(t0 + 2 / fps, t[-1] + 10 * step, step)
            if not len(t1s):
                continue
            u = np.clip((t[None, :] - t0) / (t1s[:, None] - t0), 0, 1)
            rss = ((f(u) - prog[None, :]) ** 2).sum(axis=1)
            j = int(np.argmin(rss))
            if best is None or rss[j] < best[0]:
                best = (float(rss[j]), float(t0), float(t1s[j]))
        rss, t0, t1 = best
        fits[fam] = {"bic": round(n * np.log(max(rss, 1e-12) / n) + 2 * np.log(n), 1), "t0": round(t0, 4),
                     "t1": round(t1, 4)}
    ranked = sorted(fits, key=lambda k: fits[k]["bic"])
    gap = fits[ranked[1]]["bic"] - fits[ranked[0]]["bic"]
    past = prog[np.argmax(prog >= 0.95):] if (prog >= 0.95).any() else np.array([])
    res = {"kind": "move", "quantity": name, "unit": q["unit"], "n": n, "fps": fps,
           "window": [round(float(t[0]), 4), round(float(t[-1]), 4)], "start": round(lo, 4), "end": round(hi, 4),
           "first_change": _crossing(t, prog, 0.02),
           **{f"t{int(p * 100)}": _crossing(t, prog, p) for p in (0.1, 0.5, 0.9, 0.95)},
           "overshoot": round(float(max(0.0, past.max() - 1.0)) if past.size else 0.0, 4),
           "ease": ranked[0] if gap >= EASE_GAP else "unknown", "best": ranked[0], "next": ranked[1],
           "bic_gap": round(gap, 1), "fits": fits}
    for k in ("first_change", "t10", "t50", "t90", "t95"):
        res[k] = None if res[k] is None else round(res[k], 4)
    _save(effect, name, res)
    return {k: res[k] for k in ("quantity", "ease", "best", "bic_gap", "t10", "t50", "t90", "t95", "start", "end",
                                "overshoot")} | {"fit": fits[ranked[0]]}


def fit_pulse(effect: Effect, name: str, t_from=None, t_to=None, base_frames=4) -> dict:
    t, v, q = series.get(effect, name)
    sel = np.ones(len(t), bool)
    if t_from is not None:
        sel &= t >= t_from
    if t_to is not None:
        sel &= t <= t_to
    t, v = t[sel], v[sel]
    ok = ~np.isnan(v)
    t, v = t[ok], v[ok]
    base = _level(v, slice(0, base_frames))
    k = int(np.argmax(v))
    peak = float(v[k])
    half = base + 0.5 * (peak - base)
    up = _crossing(t[: k + 1], (v[: k + 1] - base) / max(1e-9, peak - base), 0.5)
    down = next((float(t[i]) for i in range(k, len(t)) if v[i] <= half), None)
    res = {"kind": "pulse", "quantity": name, "unit": q["unit"], "base": round(base, 3), "peak": round(peak, 3),
           "peak_t": round(float(t[k]), 4), "rise_t50": None if up is None else round(up, 4),
           "fall_t50": None if down is None else round(down, 4), "fps": effect.meta().get("fps")}
    _save(effect, name, res)
    return res


def _save(effect: Effect, name: str, res: dict):
    fits = effect.read("fits.json", {})
    fits[name] = res
    effect.write("fits.json", fits)
