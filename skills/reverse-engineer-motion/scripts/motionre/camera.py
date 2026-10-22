"""The camera: a similarity (scale, rotation, shift) mapping frame 0 onto every frame, so later stages can measure in
frame-0 coordinates and a push or pan is not mistaken for an element's own motion.

Two methods, both registering each frame straight to frame 0, never chained frame to frame (Johnny Harris effect B:
a chain drifted about 2% over 90 frames and invented a pan he never made):

  features  ORB features on the background, matched to frame 0 with RANSAC; the previous frame's result only filters
            the matches. Moving elements are masked out with --exclude-source (mask sources, dilated) and
            --exclude-box. Use when the background has texture (a map, a page).
  anchors   two elements fixed to the background (two badges on a map), found by place in tracks.json; each centre
            is fitted linearly in time and the similarity comes from the two points (effect A). Use when SAM 3 holds
            two fixed elements better than features hold the background, or the push is known to be steady.

Writes camera.json (per frame: the 2x3 matrix, scale, rotation, shift, inliers) and the series camera.scale,
camera.tx, camera.ty, camera.rot, plus a summary with the zoom rate and how linear it is.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from . import series
from .study import Effect


def _sim_from_points(a0, b0, a1, b1):
    v0, v1 = b0 - a0, b1 - a1
    s = np.linalg.norm(v1) / np.linalg.norm(v0)
    ang = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
    c, si = s * np.cos(ang), s * np.sin(ang)
    R = np.array([[c, -si], [si, c]])
    t = a1 - R @ a0
    return np.array([[c, -si, t[0]], [si, c, t[1]]])


def _source_union(effect: Effect, source: str, frame: int, shape, dilate: int):
    d = effect.masks_dir / source
    rec_p = d / f"f{frame:04d}.json"
    m = np.zeros(shape[:2], np.uint8)
    if rec_p.exists():
        for name in json.loads(rec_p.read_text())["masks"]:
            mk = cv2.imread(str(d / name), cv2.IMREAD_GRAYSCALE)
            if mk is not None:
                m |= (cv2.resize(mk, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    return cv2.dilate(m, np.ones((dilate, dilate), np.uint8)) if dilate > 1 and m.any() else m


def features(effect: Effect, exclude_sources=(), exclude_boxes=(), dilate=31, n_features=3000):
    fr = effect.frames()
    img = lambda i: cv2.imread(str(effect.frame_png(i)))  # noqa: E731
    first = img(fr[0]["i"])
    H, W = first.shape[:2]
    orb = cv2.ORB_create(n_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    base = np.full((H, W), 255, np.uint8)
    for x0, y0, x1, y1 in exclude_boxes:
        base[int(y0):int(y1), int(x0):int(x1)] = 0

    def feats(i, bgr):
        m = base.copy()
        for s in exclude_sources:
            m[_source_union(effect, s, i, bgr.shape, dilate) > 0] = 0
        return orb.detectAndCompute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), m)

    f0 = feats(fr[0]["i"], first)
    mats, inl = [np.eye(3)[:2]], [None]
    for r in fr[1:]:
        k1 = feats(r["i"], img(r["i"]))
        ms = bf.match(f0[1], k1[1]) if f0[1] is not None and k1[1] is not None else []
        p0 = np.float32([f0[0][m.queryIdx].pt for m in ms])
        p1 = np.float32([k1[0][m.trainIdx].pt for m in ms])
        guess = mats[-1]
        if len(p0):
            ok = np.hypot(*((np.c_[p0, np.ones(len(p0))] @ guess.T) - p1).T) < 40
            p0, p1 = p0[ok], p1[ok]
        if len(p0) < 6:
            mats.append(guess)
            inl.append(0)
            continue
        M, keep = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=1.5)
        mats.append(M if M is not None else guess)
        inl.append(int(keep.sum()) if keep is not None else 0)
    return mats, inl


def anchors(effect: Effect, a, b, within=90.0, max_size=220.0):
    fr = effect.frames()
    t = np.array(series.rel_times(effect))
    pts = []
    for x, y in (a, b):
        rows = series.tracks_near(effect, x, y, within, max_size)
        if len(rows) < 3:
            raise SystemExit(f"anchor ({x:g}, {y:g}): fewer than 3 tracked frames; check tracks-sheet.jpg")
        fs = sorted(rows)
        kx = np.polyfit(t[fs], [rows[f]["cx"] for f in fs], 1)
        ky = np.polyfit(t[fs], [rows[f]["cy"] for f in fs], 1)
        pts.append((kx, ky, len(fs)))
    at = lambda k, tt: np.array([np.polyval(k[0], tt), np.polyval(k[1], tt)])  # noqa: E731
    a0, b0 = at(pts[0], t[0]), at(pts[1], t[0])
    mats = [_sim_from_points(a0, b0, at(pts[0], tt), at(pts[1], tt)) for tt in t]
    return mats, [min(pts[0][2], pts[1][2])] * len(fr)


def measure(effect: Effect, method: str = "features", **kw) -> dict:
    mats, inl = features(effect, **kw) if method == "features" else anchors(effect, **kw)
    t = series.rel_times(effect)
    rows = []
    for i, (M, n) in enumerate(zip(mats, inl)):
        s = float(np.hypot(M[0][0], M[1][0]))
        rows.append({"f": i, "t": round(t[i], 4), "M": [[round(float(x), 6) for x in r] for r in M], "scale": round(s, 6),
                     "rot_deg": round(float(np.degrees(np.arctan2(M[1][0], M[0][0]))), 3),
                     "tx": round(float(M[0][2]), 2), "ty": round(float(M[1][2]), 2), "inliers": n})
    sc = np.array([r["scale"] for r in rows])
    k = np.polyfit(t, sc, 1)
    W = effect.meta().get("source", {}).get("width") or 1920
    H = effect.meta().get("source", {}).get("height") or 1080
    centre = np.array([[r["M"][0][0] * W / 2 + r["M"][0][1] * H / 2 + r["M"][0][2],
                        r["M"][1][0] * W / 2 + r["M"][1][1] * H / 2 + r["M"][1][2]] for r in rows])
    summary = {"method": method, "zoom_total": round(float(sc[-1] / sc[0]), 5), "zoom_per_s": round(float(k[0]), 5),
               "linear_max_err": round(float(np.max(np.abs(np.polyval(k, t) - sc))), 5),
               "centre_shift_px": [round(float(v), 1) for v in centre[-1] - centre[0]],
               "min_inliers": min((r["inliers"] for r in rows[1:] if r["inliers"] is not None), default=None)}
    effect.write("camera.json", {"summary": summary, "rows": rows})
    for key, unit in (("scale", "x"), ("tx", "px"), ("ty", "px"), ("rot_deg", "deg")):
        series.put(effect, f"camera.{key.replace('_deg', '')}", t, [r[key] for r in rows], unit, "derived", "camera")
    return summary


def matrices(effect: Effect):
    """Per-frame 2x3 frame-0 -> frame matrices, or identities when no camera was measured."""
    cam = effect.read("camera.json")
    n = len(effect.frames())
    if not cam:
        return [np.eye(3)[:2] for _ in range(n)]
    return [np.array(r["M"], float) for r in cam["rows"]]
