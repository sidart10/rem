"""Measure every mask: geometry for the motion stages, appearance for the linker. No pixels leave raw/.

Per mask (frame pixels, origin top-left):
  box [x0, y0, x1, y1], cx/cy (centroid of the mask, not the box), area,
  rot [cx, cy, w, h, angle] (minimum-area rotated box),
  corners [[x, y] x4] tl, tr, br, bl (a 4-point fit of the outline; the perspective stage reads flips and tilts
          from these: a card turning edge-on keeps its height and loses its width),
  quad_fit: how much of the mask the 4-point fit covers (1.0 = a clean quadrilateral, low = not a card),
  edge: true when the mask touches the frame border (its size is cut off, so size claims there are unknown),
  lab [L, a, b] mean colour, hist (HSV 8x4x4, sums to 1), thumb (24x24 grey of the masked crop, hex).
Masks of one frame that cover the same pixels (IoU over 0.8, from one prompt or several) are one detection, keeping
the higher score and listing every source that found it.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from .study import Effect

THUMB = 24
DEDUPE_IOU = 0.8


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(-1, 2).astype(np.float32)
    s, d = pts.sum(1), pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def quad(mask: np.ndarray):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    q = None
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        a = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a) == 4:
            q = a
            break
    if q is None:
        q = cv2.boxPoints(cv2.minAreaRect(c))
    q = order_corners(q)
    poly = np.zeros_like(mask, np.uint8)
    cv2.fillConvexPoly(poly, q.astype(np.int32), 1)
    inter = float((poly & mask).sum())
    fit = inter / max(1.0, float((poly | mask).sum()))
    return q, round(fit, 3), cv2.minAreaRect(c)


def appearance(frame: np.ndarray, mask: np.ndarray, box) -> dict:
    x0, y0, x1, y1 = box
    crop, m = frame[y0:y1 + 1, x0:x1 + 1], mask[y0:y1 + 1, x0:x1 + 1].astype(bool)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[m].mean(0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], m.astype(np.uint8), [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    h = h / max(1.0, h.sum())
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g[~m] = g[m].mean()
    t = cv2.resize(g, (THUMB, THUMB), interpolation=cv2.INTER_AREA).clip(0, 255).astype(np.uint8)
    return {"lab": [round(float(v), 1) for v in lab], "hist": [round(float(v), 4) for v in h], "thumb": t.tobytes().hex()}


def one(frame: np.ndarray, mask: np.ndarray, score, source: str) -> dict | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    H, W = mask.shape
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    q, fit, rr = quad(mask)
    (rx, ry), (rw, rh), ang = rr
    return {"box": box, "cx": round(float(xs.mean()), 2), "cy": round(float(ys.mean()), 2), "area": int(len(xs)),
            "w": box[2] - box[0] + 1, "h": box[3] - box[1] + 1,
            "rot": [round(rx, 1), round(ry, 1), round(rw, 1), round(rh, 1), round(ang, 2)],
            "corners": [[round(float(x), 1), round(float(y), 1)] for x, y in q], "quad_fit": fit,
            "edge": bool(box[0] <= 1 or box[1] <= 1 or box[2] >= W - 2 or box[3] >= H - 2),
            "score": None if score is None else round(float(score), 3), "sources": [source],
            **appearance(frame, mask, box)}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    i = np.logical_and(a, b).sum()
    return float(i) / max(1.0, float(np.logical_or(a, b).sum()))


def measure(effect: Effect, sources: list | None = None, min_score: float = 0.5) -> dict:
    frames = effect.frames()
    dirs = [d for d in sorted(effect.masks_dir.iterdir()) if d.is_dir()] if effect.masks_dir.exists() else []
    if sources:
        dirs = [d for d in dirs if d.name in sources]
    if not dirs:
        raise SystemExit(f"no masks under {effect.masks_dir}; run `motion_re.py mask` first")
    out, dropped = [], 0
    for fr in frames:
        i = fr["i"]
        img = cv2.imread(str(effect.frame_png(i)))
        kept = []  # (mask, det)
        for d in dirs:
            j = d / f"f{i:04d}.json"
            if not j.exists():
                continue
            rec = json.loads(j.read_text())
            for k, name in enumerate(rec["masks"]):
                score = rec["scores"][k] if k < len(rec["scores"]) else None
                if score is not None and score < min_score:
                    dropped += 1
                    continue
                m = cv2.imread(str(d / name), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                if m.shape != img.shape[:2]:
                    m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                m = m > 127
                det = one(img, m, score, rec["source"])
                if det is None:
                    continue
                # by position: list.index() would compare the numpy masks element-wise and raise
                at = next((n for n, x in enumerate(kept) if iou(x[0], m) > DEDUPE_IOU), None)
                if at is None:
                    kept.append((m, det))
                else:
                    dup = kept[at][1]
                    dup["sources"] = sorted(set(dup["sources"] + det["sources"]))
                    if (det["score"] or 0) > (dup["score"] or 0):
                        det["sources"] = dup["sources"]
                        kept[at] = (m, det)
        out.append({"i": i, "t": fr["t"], "dup": fr["dup"], "dets": [d for _, d in kept]})
    data = {"sources": [d.name for d in dirs], "min_score": min_score, "dropped_low_score": dropped, "frames": out}
    effect.write("raw/detections.json", data)
    return {"frames": len(out), "detections": sum(len(f["dets"]) for f in out), "dropped_low_score": dropped}
