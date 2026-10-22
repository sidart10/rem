"""Every frame of one effect window at the source's own rate, with its real time and a duplicate flag.

The 2 fps pass and single stills hid the motion that matters (the deck flips in place in one or two frames), so an
effect is always read at native rate. Times come from the decoder (showinfo pts), not from index / fps, so a
variable-frame-rate phone upload stays honest. A frame that repeats the previous one (pulldown, a platform
re-encode, a held frame) is flagged `dup` so later stages don't read it as the element holding still.
"""
from __future__ import annotations

import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

DUP_MEAN = 0.35   # mean abs grey difference (0-255) at 270 px wide under which a frame repeats its predecessor
DUP_MAX = 12      # and no pixel changes more than this


def probe(src: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate,codec_name:format=duration",
                          "-of", "json", str(src)], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    return {"width": s["width"], "height": s["height"], "codec": s.get("codec_name"),
            "fps": float(Fraction(s["r_frame_rate"])), "avg_fps": float(Fraction(s["avg_frame_rate"] or "0/1") or 0),
            "duration_s": float(d["format"].get("duration", 0))}


def extract(src: Path, t0: float, t1: float, out_dir: Path) -> list:
    """Decode [t0, t1] to out_dir/f0000.png...; returns [{i, t, dup}] with t in source seconds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("f*.png"):
        old.unlink()
    # -copyts keeps the decoder's own timestamps, so a frame's time is the source frame's time, not "seek + n / fps"
    r = subprocess.run(["ffmpeg", "-hide_banner", "-y", "-copyts", "-ss", f"{t0:.4f}", "-to", f"{t1:.4f}", "-i", str(src),
                        "-fps_mode", "passthrough", "-vf", "showinfo", "-start_number", "0", str(out_dir / "f%04d.png")],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"ffmpeg could not read {src}: {r.stderr.strip()[-400:]}")
    start = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=start_time",
                            "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout.strip()
    start = float(start) if re.match(r"^-?[\d.]+$", start or "") else 0.0
    pts = [float(m) - start for m in re.findall(r"pts_time:\s*(-?[\d.]+)", r.stderr)]
    pngs = sorted(out_dir.glob("f*.png"))
    if len(pts) != len(pngs):  # showinfo and the muxer disagree: fall back to the nominal rate
        fps = probe(src)["fps"]
        pts = [t0 + i / fps for i in range(len(pngs))]
    rows, prev = [], None
    for i, p in enumerate(pngs):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(g, (270, int(270 * g.shape[0] / g.shape[1])), interpolation=cv2.INTER_AREA).astype(np.int16)
        dup = False
        if prev is not None:
            d = np.abs(small - prev)
            dup = bool(d.mean() < DUP_MEAN and d.max() <= DUP_MAX)
        rows.append({"i": i, "t": round(pts[i], 4), "dup": dup})
        prev = small
    return rows


def ensure(effect) -> bool:
    """Re-extract the window's frames when they were cleaned away. Frames are a cache, not a record: effect.json keeps
    the source file and the window, frames.json the times, and every measure reads frames through this. Returns True
    when it had to extract."""
    want = len(effect.frames())
    have = len(list(effect.frames_dir.glob("f*.png"))) if effect.frames_dir.exists() else 0
    if want and have >= want:
        return False
    meta = effect.meta()
    src = Path(meta.get("source", {}).get("path", ""))
    if not src.exists():
        raise SystemExit(f"{effect.root.name}: frames are gone and the source {src} is missing; restore it or rerun "
                         "`motion_re.py window`")
    w = meta["window"]
    rows = extract(src, w["t0"], w["t1"], effect.frames_dir)
    old = effect.frames()
    if old and [r["i"] for r in rows] != [r["i"] for r in old]:
        raise SystemExit(f"{effect.root.name}: re-extracted {len(rows)} frames but frames.json has {len(old)}; the "
                         "source changed, so start the window again")
    return True


CACHE_DIRS = ("raw/frames", "raw/annotated")


def clean(path: Path, dry_run: bool = False) -> dict:
    """Delete the frame caches (raw/frames, raw/annotated) of one effect, or of every effect in a study. Masks,
    detections, sheets, series, fits, claims and recipes stay: they are paid for or hand-checked, frames are not."""
    import shutil

    path = Path(path)
    roots = [path] if (path / "effect.json").exists() else sorted(p.parent for p in path.glob("motion/*/effect.json"))
    roots += [] if roots else sorted(p.parent.parent for p in path.glob("motion/*/raw"))
    freed, removed = 0, []
    for r in roots:
        for d in CACHE_DIRS:
            q = r / d
            if q.is_dir():
                freed += sum(f.stat().st_size for f in q.rglob("*") if f.is_file())
                removed.append(str(q))
                if not dry_run:
                    shutil.rmtree(q)
    return {"removed": removed, "freed_mb": round(freed / 1e6, 1), "dry_run": dry_run}
