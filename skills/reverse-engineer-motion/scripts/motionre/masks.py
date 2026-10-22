"""Mask providers. Each writes raw/masks/<source>/f0000.json (+ f0000--m0.png ...) per frame, same format:

  {"frame": 7, "source": "sam3:card", "prompt": "card", "scores": [0.89, ...], "masks": ["f0007--m0.png", ...]}

sam3   fal-ai/sam-3/image, one call per frame and prompt (cloud, billed, public footage only). Proven on Justyn's
       deck 2026-10-19: masks hug every card through edge-on frames and a hand in front. The video endpoint
       (fal-ai/sam-3-1/video-rle) did not propagate (3 of 41 frames) and is not used. Frames already cached are never
       sent again, so a rerun costs nothing.
alpha  local and free: the alpha channel of an RGBA render (our rebuild rendered with a transparent background, or
       any RGBA video or PNG sequence), split into connected parts. This is how our own renders get measured
       without sending anyone's face anywhere, and how the tracker is calibrated against motion we authored.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from . import budget
from .study import Effect, cloud_refusal

SAM3 = "fal-ai/sam-3/image"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "x"


def source_dir(effect: Effect, source: str) -> Path:
    return effect.masks_dir / slug(source.replace(":", "-"))


def sam3(effect: Effect, prompt: str, frames: list, workers: int = 4, dry_run: bool = False,
         max_masks: int = 10) -> dict:
    """Mask `frames` (indices) with SAM 3 for one text prompt. Returns a summary with the cost."""
    why = cloud_refusal(effect.meta())
    if why:
        raise SystemExit(f"Refused, nothing sent: this effect may not go to a cloud model ({why}). "
                         "Use the alpha provider for our own renders.")
    out = source_dir(effect, f"sam3:{prompt}")
    todo = [i for i in frames if not (out / f"f{i:04d}.json").exists()]
    each = budget.price(SAM3)
    planned = len(todo) * each
    state = budget.check(effect, planned) if todo else {"planned_usd": 0}
    summary = {"source": f"sam3:{prompt}", "frames": len(frames), "cached": len(frames) - len(todo),
               "calls": len(todo), "usd": round(planned, 4), "budget": state}
    if dry_run or not todo:
        return summary
    from . import fal_client as fal_api  # needs FAL_KEY

    out.mkdir(parents=True, exist_ok=True)

    def one(i):
        args = {"image_url": fal_api.data_uri(str(effect.frame_png(i))), "prompt": prompt,
                "return_multiple_masks": True, "max_masks": max_masks, "include_scores": True,
                "include_boxes": True, "apply_mask": False}
        res = fal_api.run(SAM3, args, timeout=300)
        budget.log(effect, SAM3, each, frame=i, prompt=prompt)
        rec = {"frame": i, "source": f"sam3:{prompt}", "prompt": prompt, "scores": res.get("scores") or [],
               "boxes": res.get("boxes") or [], "masks": []}
        for k, m in enumerate(res.get("masks") or []):
            name = f"f{i:04d}--m{k}.png"
            fal_api.download(m["url"], str(out / name))
            rec["masks"].append(name)
        (out / f"f{i:04d}.json").write_text(json.dumps(rec))
        return i, len(rec["masks"])

    done, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, i): i for i in todo}
        for f in as_completed(futs):
            try:
                done.append(f.result())
            except Exception as e:  # a failed call is not billed by fal; keep going, report it
                errors.append({"frame": futs[f], "error": str(e)[:300]})
    summary.update({"sent": len(done), "errors": errors,
                    "masks_per_frame": dict(sorted((str(i), n) for i, n in done))})
    return summary


def alpha(effect: Effect, rgba: Path, name: str = "alpha", min_area: int = 400, offset_s: float = 0.0) -> dict:
    """Local masks from the alpha channel of an RGBA video (or a folder of RGBA PNGs) covering the same window."""
    meta, frames = effect.meta(), effect.frames()
    out = source_dir(effect, f"alpha:{name}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if rgba.is_dir():
            pngs = sorted(rgba.glob("*.png"))
        else:
            t0 = meta["window"]["t0"] + offset_s
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.4f}", "-i", str(rgba),
                            "-frames:v", str(len(frames)), "-fps_mode", "passthrough", "-vf", "alphaextract",
                            "-start_number", "0", str(tmp / "a%04d.png")], check=True)
            pngs = sorted(tmp.glob("a*.png"))
        n = 0
        for i, p in enumerate(pngs[: len(frames)]):
            a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            a = a[..., 3] if a.ndim == 3 and a.shape[2] == 4 else (a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY))
            k, lab, st, _ = cv2.connectedComponentsWithStats((a > 127).astype(np.uint8))
            rec = {"frame": i, "source": f"alpha:{name}", "prompt": None, "scores": [], "masks": []}
            for c in range(1, k):
                if st[c, cv2.CC_STAT_AREA] < min_area:
                    continue
                m = f"f{i:04d}--m{len(rec['masks'])}.png"
                cv2.imwrite(str(out / m), ((lab == c) * 255).astype(np.uint8))
                rec["masks"].append(m)
                rec["scores"].append(1.0)
            (out / f"f{i:04d}.json").write_text(json.dumps(rec))
            n += 1
    return {"source": f"alpha:{name}", "frames": n, "usd": 0}


def import_cache(effect: Effect, src: Path, prompt: str, usd_spent: float = 0.0, note: str = "") -> dict:
    """Bring in masks made before this skill existed (the 2026-10-19 spike: fNN.json + fNN--mK.png per frame)."""
    out = source_dir(effect, f"sam3:{prompt}")
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for j in sorted(src.glob("f*.json")):
        rec = json.loads(j.read_text())
        i = int(re.sub(r"\D", "", j.stem))
        names = []
        for k, m in enumerate(rec.get("masks") or []):
            name = f"f{i:04d}--m{k}.png"
            shutil.copy(src / m, out / name)
            names.append(name)
        (out / f"f{i:04d}.json").write_text(json.dumps({"frame": i, "source": f"sam3:{prompt}", "prompt": prompt,
                                                        "scores": rec.get("scores") or [], "boxes": rec.get("boxes") or [],
                                                        "masks": names, "imported_from": str(src)}))
        n += 1
    if usd_spent:
        budget.log(effect, SAM3, usd_spent, frames=n, prompt=prompt, note=note or f"spent earlier, imported from {src}")
    return {"source": f"sam3:{prompt}", "imported_frames": n, "usd_logged": usd_spent}
