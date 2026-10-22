"""Local pixel measures for what masks can't see: thin or half-drawn lines, glows, reveals, typing heads, parts.

Every frame is first warped back into frame-0 coordinates with camera.json (when there is one), so a push or pan
doesn't read as motion. A measure is a set of pixel rules, ANDed, on these channels:

  luma, r, g, b   0-255          red   r - g (a red glow)          sat   HSV saturation 0-255
  d<channel>      the channel minus frame 0's (dluma<-40: pixels that got at least 40 darker since frame 0)

restricted to a region: --box (frame-0 px), --ring cx,cy,r0,r1, --within-source (union of a mask source's last
frames, dilated: the finished line's area) and --not-source (union of a source over every frame, dilated: the badges
the line leaves from). Outputs per frame, as series quantities named <name>.<what>:

  area, top, bottom, left, right, cx, cy     of the pixels that pass (per-part geometry: Johnny Harris effect E,
                                             where one scale for the whole badge was wrong and each part's own edges
                                             were right)
  mean                                       with --value <channel>: the channel's mean over the region (a pulse)
  drawn, drawn_frac                          with --path or --path-auto: how far a line has drawn along its own path,
                                             walking out from the start and bridging gaps up to --bridge px (beads
                                             leave gaps), gaps forgiven inside the first --skip px (under a badge's
                                             label and glow), counted from --origin px along the path (the badge
                                             edge; not from --skip, or an ease-out's fast start is thrown away).
                                             Effect A: measure along the path, not as straight-line
                                             distance; on a curved line the two disagree near the end.

--path-auto ax,ay --toward tx,ty builds the path from the last frame's passing pixels on the side facing (tx, ty):
a nearest-neighbour chain out from (ax, ay), smoothed, saved to paths/<name>.json.
"""
from __future__ import annotations

import json
import re

import cv2
import numpy as np

from . import camera, series
from .study import Effect

RULE = re.compile(r"^(d?)(luma|r|g|b|red|sat)\s*(<=|>=|<|>)\s*(-?[\d.]+)$")


def channels(bgr):
    b, g, r = (bgr[..., k].astype(np.int16) for k in range(3))
    return {"luma": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16), "r": r, "g": g, "b": b, "red": r - g,
            "sat": cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].astype(np.int16)}


def parse_rule(text: str):
    m = RULE.match(text.replace(" ", ""))
    if not m:
        raise SystemExit(f"rule {text!r}: write it like luma<60, red>25 or dluma<-40")
    return m.group(1) == "d", m.group(2), m.group(3), float(m.group(4))


def _cmp(a, op, x):
    return {"<": a < x, ">": a > x, "<=": a <= x, ">=": a >= x}[op]


def _src_masks(effect: Effect, source: str, frame: int, shape):
    d = effect.masks_dir / source
    p = d / f"f{frame:04d}.json"
    m = np.zeros(shape[:2], np.uint8)
    if p.exists():
        for name in json.loads(p.read_text())["masks"]:
            mk = cv2.imread(str(d / name), cv2.IMREAD_GRAYSCALE)
            if mk is not None:
                m |= (cv2.resize(mk, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    return m


def _unwarp(img, M, W, H, nearest=False):
    Minv = cv2.invertAffineTransform(M.astype(np.float64))
    return cv2.warpAffine(img, Minv, (W, H), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _chain(points, anchor, max_jump=45.0):
    """Order scattered line pixels into a path: nearest-neighbour steps out from the anchor, never doubling back."""
    cells = {}
    for x, y in points:
        cells.setdefault((int(x) // 4, int(y) // 4), []).append((x, y))
    pts = np.array([np.mean(v, axis=0) for v in cells.values()], float)
    if len(pts) < 3:
        return None
    a = np.array(anchor, float)
    r = np.hypot(*(pts - a).T)
    cur = pts[int(np.argmin(r))]
    used = np.zeros(len(pts), bool)
    used[int(np.argmin(r))] = True
    path = [a, cur]
    while True:
        d = np.hypot(*(pts - cur).T)
        d[used] = np.inf
        d[r < np.hypot(*(cur - a)) - 10] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]) or d[j] > max_jump:
            break
        used[d < 6] = True
        used[j] = True
        cur = pts[j]
        path.append(cur)
    p = np.array(path)
    if len(p) < 4:
        return p
    # A thick line (a rope with a glow and beads) gives a band of pixels several px wide, and the chain zig-zags
    # across it; summed, the wiggle made his path 10% longer than the same line drawn thin (effect A, 2026-10-21).
    # Resample at 4 px and smooth along the path (Gaussian, sigma 12 px), keeping both ends where they are.
    q, _, _ = _resample(p, 4.0)
    q = np.vstack([q, p[-1]])
    if len(q) > 8:
        k = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)
        k /= k.sum()
        pad = np.vstack([np.repeat(q[:1], 9, axis=0), q, np.repeat(q[-1:], 9, axis=0)])
        sm = np.c_[np.convolve(pad[:, 0], k, "valid"), np.convolve(pad[:, 1], k, "valid")]
        sm[0], sm[-1] = q[0], q[-1]
        q = sm
    return q


def _resample(p, step=2.0):
    seg = np.diff(p, axis=0)
    L = np.r_[0, np.cumsum(np.hypot(*seg.T))]
    s = np.arange(0, L[-1], step)
    return np.c_[np.interp(s, L, p[:, 0]), np.interp(s, L, p[:, 1])], s, float(L[-1])


def measure(effect: Effect, name: str, rules=(), value=None, box=None, ring=None, within=None, within_dilate=41,
            within_last=8, not_source=None, not_pad=12, open_px=0, path=None, path_auto=None, toward=None,
            tube=6.0, bridge=40.0, skip=0.0, origin=0.0) -> dict:
    fr = effect.frames()
    mats = camera.matrices(effect)
    img0 = cv2.imread(str(effect.frame_png(fr[0]["i"])))
    H, W = img0.shape[:2]
    ch0 = channels(img0)
    parsed = [parse_rule(r) for r in rules]
    region = np.ones((H, W), bool)
    if box:
        x0, y0, x1, y1 = box
        b_ = np.zeros((H, W), bool)
        b_[int(y0):int(y1), int(x0):int(x1)] = True
        region &= b_
    if ring:
        cx, cy, r0, r1 = ring
        yy, xx = np.ogrid[:H, :W]
        d = np.hypot(xx - cx, yy - cy)
        region &= (d >= r0) & (d <= r1)
    if within:
        u = np.zeros((H, W), np.uint8)
        for i in range(max(0, len(fr) - within_last), len(fr)):
            u |= _unwarp(_src_masks(effect, within, i, (H, W)), mats[i], W, H, nearest=True)
        if not u.any():
            raise SystemExit(f"--within-source {within}: no masks on the last {within_last} frames")
        region &= cv2.dilate(u, np.ones((within_dilate, within_dilate), np.uint8)) > 0
    if not_source:
        u = np.zeros((H, W), np.uint8)
        for i in range(len(fr)):
            u |= _unwarp(_src_masks(effect, not_source, i, (H, W)), mats[i], W, H, nearest=True)
        region &= ~(cv2.dilate(u, np.ones((2 * not_pad + 1, 2 * not_pad + 1), np.uint8)) > 0)

    def passing(i):
        img = _unwarp(cv2.imread(str(effect.frame_png(fr[i]["i"]))), mats[i], W, H)
        ch = channels(img)
        m = region.copy()
        for rel, c, op, x in parsed:
            m &= _cmp(ch[c] - ch0[c] if rel else ch[c], op, x)
        if open_px > 1:
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8)) > 0
        return m, ch

    # the path, when the measure draws along one
    pth = None
    if path_auto:
        m_last, _ = passing(len(fr) - 1)
        ys, xs = np.nonzero(m_last)
        a = np.array(path_auto, float)
        keep = np.ones(len(xs), bool)
        if toward:
            v = np.array(toward, float) - a
            keep = ((xs - a[0]) * v[0] + (ys - a[1]) * v[1]) > 0
        pth = _chain(np.c_[xs[keep], ys[keep]], a)
        if pth is None:
            raise SystemExit(f"{name}: too few passing pixels on the last frame to build a path")
        (effect.root / "paths").mkdir(exist_ok=True)
        (effect.root / "paths" / f"{name}.json").write_text(json.dumps({"points": pth.round(1).tolist(),
                                                                         "from": list(path_auto), "toward": toward}))
    elif path:
        pth = np.array(json.loads(open(path).read())["points"], float)
    if pth is not None:
        samples, s_along, L = _resample(pth)
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * tube + 1), int(2 * tube + 1)))
        sx = np.clip(samples[:, 0].round().astype(int), 0, W - 1)
        sy = np.clip(samples[:, 1].round().astype(int), 0, H - 1)

    t = series.rel_times(effect)
    out = {k: [] for k in ("area", "top", "bottom", "left", "right", "cx", "cy", "mean", "drawn", "drawn_frac")}
    for i in range(len(fr)):
        m, ch = passing(i)
        if value:
            sel = m if parsed else region
            out["mean"].append(float(ch[value][sel].mean()) if sel.any() else None)
            continue
        ys, xs = np.nonzero(m)
        n = len(xs)
        out["area"].append(n)
        for k, v in (("top", ys.min() if n else None), ("bottom", ys.max() if n else None),
                     ("left", xs.min() if n else None), ("right", xs.max() if n else None),
                     ("cx", xs.mean() if n else None), ("cy", ys.mean() if n else None)):
            out[k].append(None if v is None else float(v))
        if pth is not None:
            cov = cv2.dilate(m.astype(np.uint8), disk)[sy, sx] > 0
            last, gap, step = None, 0.0, 2.0
            for k, s in enumerate(s_along):
                if cov[k]:
                    last, gap = s, 0.0
                else:
                    gap += step
                    if s >= skip and gap > bridge:
                        break
            d = 0.0 if last is None else max(0.0, float(last) - origin)
            out["drawn"].append(d)
            out["drawn_frac"].append(d / max(1.0, L - origin))
    made = []
    keys = ["mean"] if value else ["area", "top", "bottom", "left", "right", "cx", "cy"] + (
        ["drawn", "drawn_frac"] if pth is not None else [])
    for k in keys:
        unit = "x" if k == "drawn_frac" else ("px²" if k == "area" else ("level" if k == "mean" else "px"))
        series.put(effect, f"{name}.{k}", t, out[k], unit, "observed" if k != "drawn_frac" else "derived", "pixels",
                   rules=list(rules), value=value)
        made.append(f"{name}.{k}")
    res = {"quantities": made, "frames": len(fr)}
    if pth is not None:
        res["path_px"] = round(L, 1)
    return res
