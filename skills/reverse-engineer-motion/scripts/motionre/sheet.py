"""Contact sheet of an effect window: every frame, each track's mask outline in its own colour with its id.

The quick look that tells a wrong link at a glance (two cards sharing a colour, a card changing colour mid-flip).
Written to raw/ (the creator's pixels stay local and out of git).
"""
from __future__ import annotations

import colorsys
import json

import cv2
import numpy as np

from .study import Effect


def colours(n):
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((k * 0.618) % 1, 0.85, 1.0))[::-1] for k in range(n)]


def sheet(effect: Effect, cols: int = 8, w: int = 216) -> str:
    tr = effect.read("tracks.json")
    det = effect.read("raw/detections.json")
    if not tr or not det:
        raise SystemExit("run `motion_re.py measure` and `link` first")
    tracks = tr["tracks"]
    col = dict(zip((t["id"] for t in tracks), colours(len(tracks))))
    by_frame = {}
    for t in tracks:
        for r in t["rows"]:
            by_frame.setdefault(r["f"], []).append((t["id"], r, t.get("flicker")))
    tiles = []
    for fr in det["frames"]:
        i = fr["i"]
        img = cv2.imread(str(effect.frame_png(i)))
        sc = w / img.shape[1]
        for tid, r, flick in by_frame.get(i, []):
            c = col[tid]
            pts = (np.array(r["corners"]) * 1).astype(np.int32)
            cv2.polylines(img, [pts], True, c, 6 if not flick else 2)
            x0, y0 = r["box"][:2]
            cv2.putText(img, tid, (x0 + 8, max(40, y0 + 50)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 10)
            cv2.putText(img, tid, (x0 + 8, max(40, y0 + 50)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, c, 4)
        tile = cv2.resize(img, (w, int(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        label = f"f{i} {fr['t']:.3f}s" + (" dup" if fr["dup"] else "")
        cv2.rectangle(tile, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    th = tiles[0].shape[0]
    rows = []
    for k in range(0, len(tiles), cols):
        row = tiles[k:k + cols]
        row += [np.zeros_like(tiles[0])] * (cols - len(row))
        rows.append(np.hstack(row))
    out = effect.raw / "tracks-sheet.jpg"
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return str(out)
