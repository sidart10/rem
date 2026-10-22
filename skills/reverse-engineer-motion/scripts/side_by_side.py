#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "opencv-python-headless>=4.9", "scipy>=1.11"]
# ///
"""Side-by-side video of an effect and its rebuilds, frame-locked, with labels and a frame counter.

  side_by_side.py <his effect> <rebuild effect> [<rebuild effect> ..] --out cmp.mp4 [--gif cmp.gif]
                  [--labels "Original" "Rebuild v1" ..] [--layout stack|row] [--width 960] [--speeds 1 0.5]
                  [--hold 0.4] [--crop x,y,w,h] [--gif-width 400] [--gif-fps 10] [--stills 48 ..] [--no-counter]

Each input is an effect folder (made by `motion_re.py window`): its source file and window come from effect.json, so
the original and every rebuild play the same window frame by frame (frame k of each row is the frame `measure`
read). A plain video file plays from its start. Rows shorter than the first hold their last frame. `--speeds 1 0.5`
plays the window at full speed, then at half speed (each frame shown twice), with `--hold` seconds on the last frame
between passes. `--crop` cuts the same box out of every row (in the original's pixels; rebuilds of another size are
scaled to match first). The GIF is the first pass only, small enough for a README; `--stills` saves those frames
as PNGs beside the video (<out>-f048.png). This is the side-by-side a person watches before an effect counts as learned; it is local and free.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from motionre.frames import probe  # noqa: E402

BAR = 34
COLOURS = [(64, 150, 255), (255, 170, 90), (120, 220, 140), (200, 140, 255)]  # BGR: orange, blue, green, violet


def source(item: str):
    """(path, t0, n_frames or None, fps, default label) for an effect folder or a video file."""
    p = Path(item)
    if (p / "effect.json").exists():
        m = json.loads((p / "effect.json").read_text())
        fr = json.loads((p / "frames.json").read_text()) if (p / "frames.json").exists() else {}
        n = len(fr.get("frames", [])) or None
        src = Path(m["source"]["path"])
        if not src.exists():
            raise SystemExit(f"{item}: its source file is gone ({src}); pass the video file itself instead")
        return src, float(m["window"]["t0"]), n, float(m["fps"]), m.get("name") or p.name
    if p.is_file():
        info = probe(p)
        return p, 0.0, None, float(info["fps"]), p.stem
    raise SystemExit(f"{item}: not an effect folder (no effect.json) or a video file")


def decode(path: Path, t0: float, n, w: int, h: int, crop=None, ref=None) -> list:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-i", str(path)]
    if n:
        cmd += ["-frames:v", str(n)]
    vf = f"scale={w}:{h}:flags=lanczos"
    if crop:
        x, y, cw, ch = crop
        vf = f"scale={ref[0]}:{ref[1]},crop={cw}:{ch}:{x}:{y}," + vf
    cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    if not len(frames):
        raise SystemExit(f"{path}: no frames decoded from {t0:.3f} s")
    return list(frames)


def label(img, text, colour, right=""):
    cv2.rectangle(img, (0, 0), (img.shape[1], BAR), (24, 24, 24), -1)
    cv2.putText(img, text, (10, BAR - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 1, cv2.LINE_AA)
    if right:
        (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(img, right, (img.shape[1] - tw - 10, BAR - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gif")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--layout", choices=["stack", "row"], default="stack")
    ap.add_argument("--width", type=int, default=960, help="width of each panel")
    ap.add_argument("--speeds", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--hold", type=float, default=0.4)
    ap.add_argument("--crop", help="x,y,w,h in the original's pixels")
    ap.add_argument("--gif-width", type=int, default=400)
    ap.add_argument("--gif-fps", type=int, default=10)
    ap.add_argument("--stills", type=int, nargs="+", default=[])
    ap.add_argument("--no-counter", action="store_true")
    a = ap.parse_args()

    srcs = [source(i) for i in a.inputs]
    labels = a.labels or [s[4] for s in srcs]
    if len(labels) != len(srcs):
        raise SystemExit("--labels needs one label per input")
    path0, _, n0, fps, _ = srcs[0]
    info = probe(path0)
    crop = [int(v) for v in a.crop.split(",")] if a.crop else None
    ref = (info["width"], info["height"])
    aspect = crop[3] / crop[2] if crop else info["height"] / info["width"]
    w = a.width - a.width % 2
    h = int(round(w * aspect)) // 2 * 2
    rows = [decode(p, t0, n or n0, w, h, crop, ref) for p, t0, n, _, _ in srcs]
    n = len(rows[0])

    def frame(k: int):
        panels = []
        for r, (row, text) in enumerate(zip(rows, labels)):
            img = np.vstack([np.zeros((BAR, w, 3), np.uint8), row[min(k, len(row) - 1)].copy()])
            right = "" if a.no_counter or r else f"frame {k}  +{k / fps:.2f} s"
            label(img, text, COLOURS[r % len(COLOURS)], right)
            panels.append(img)
        return np.vstack(panels) if a.layout == "stack" else np.hstack(panels)

    seq = []
    for s in a.speeds:
        rep = max(1, round(1 / s))
        seq += [k for k in range(n) for _ in range(rep)] + [n - 1] * int(round(a.hold * fps))
    first = frame(0)
    H, W = first.shape[:2]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    cache = {}
    for k in seq:
        if k not in cache:
            cache[k] = frame(k).tobytes()
        enc.stdin.write(cache[k])
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("ffmpeg failed while encoding the side-by-side")
    res = {"out": str(out), "size": f"{W}x{H}", "frames": n, "fps": round(fps, 3), "seconds": round(len(seq) / fps, 2)}
    if a.gif:
        gw = a.gif_width if a.layout == "stack" else a.gif_width * len(rows)
        first = (n * max(1, round(1 / a.speeds[0]))) / fps
        vf = (f"fps={a.gif_fps},scale={gw}:-1:flags=lanczos,split[x][y];"
              f"[x]palettegen=max_colors=128:stats_mode=diff[p];[y][p]paletteuse=dither=none")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out), "-t", f"{first:.3f}", "-vf", vf, a.gif], check=True)
        res["gif"] = a.gif
        res["gif_mb"] = round(Path(a.gif).stat().st_size / 1e6, 2)
    for k in a.stills:
        png = out.with_name(f"{out.stem}-f{k:03d}.png")
        cv2.imwrite(str(png), frame(min(k, n - 1)))
        res.setdefault("stills", []).append(str(png))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
