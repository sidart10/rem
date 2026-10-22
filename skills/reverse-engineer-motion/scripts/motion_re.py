#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "opencv-python-headless>=4.9", "scipy>=1.11"]
# ///
"""Reverse-engineer how elements move in a finished video, and prove a rebuild moves the same way.

  motion_re.py window <url|file> --study <study> --effect <slug> --t0 S --t1 S [--name ".."] [--public-url URL] [--family]
  motion_re.py mask <effect> --prompt card [--prompt ..] [--frames all|a-b|a,b,c] [--every N] [--dry-run]
  motion_re.py mask <effect> --alpha <rgba.webm|png dir> [--name rebuild]        # local, free
  motion_re.py mask <effect> --import <dir> --prompt card [--import-usd 0.24]  # masks made before this skill
  motion_re.py measure <effect> [--min-score 0.5]
  motion_re.py link <effect> [--param young_app_gate=0.35 ...]
  motion_re.py sheet <effect>
  motion_re.py run <effect>                       # measure + link + sheet
  motion_re.py spend [--study <study>]

Measuring (each step is recorded in effect.json `pipeline`, so `replay` can run the same steps on a rebuild):
  motion_re.py camera <effect> [--exclude-source SRC ..] [--exclude-box x0,y0,x1,y1 ..]        # features to frame 0
  motion_re.py camera <effect> --anchors x,y x,y [--within 90]                                  # two fixed elements
  motion_re.py track <effect> --name badge --near x,y [--within 90] [--fields cx,cy,w,h]        # a track, by place
  motion_re.py pixels <effect> --name line --rule luma<60 --rule dluma<-40 [--within-source SRC] [--not-source SRC]
               [--box ..] [--ring cx,cy,r0,r1] [--open 3] [--value red] [--path-auto x,y --toward x,y] [--skip 138 --origin 74]
  motion_re.py fit <effect> <quantity> [--pulse] [--from S --to S] [--start V --end V]
  motion_re.py claims <effect>                    # fits -> claims.json (visible quantities with tolerances)

Proving a rebuild (window the render as its own effect, mask it with --alpha, then):
  motion_re.py replay <his effect> --on <rebuild effect> [--map his-source=our-source ..] [--keep-frames]
  motion_re.py compare <his effect> <rebuild effect>
  motion_re.py recipe <effect> [--set beads.gap_px=24 ..]   # recipe.json, what a graphic's builder reads
  motion_re.py clean <effect|study> [--dry-run]               # delete frame caches; they re-extract when needed
  motion_re.py --self-test

<study> is any folder for one creator or video (for example studies/2026-10-<creator>); the effect
lives in <study>/motion/<slug>/ (layout in motionre/study.py). A URL is fetched with yt-dlp into <study>/raw/media/
(git-ignored) and recorded as public. A local file is public only with --public-url (the post it was downloaded from);
files under ~/Pictures, ~/Movies, /Volumes or MOTION_RE_PRIVATE_DIRS never go to the cloud. SAM 3 runs cost $0.005
per frame and prompt, are priced whole before sending, and are refused past the caps in references/budget.json
($10 per study, $40 per month by default). They need FAL_KEY. Try a prompt on two or three frames first.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

from motionre import budget, frames as fr, masks, measure, sheet  # noqa: E402
from motionre import camera, claims as claimsmod, fit as fitmod, pixels, series  # noqa: E402
from motionre import link as linker  # noqa: E402
from motionre.study import Effect, ensure_ignored, sha12, is_private_path, cloud_refusal  # noqa: E402



def show(x):
    print(json.dumps(x, indent=1))


def frame_list(spec: str, every: int, n: int) -> list:
    if spec == "all":
        ks = list(range(n))
    elif "-" in spec and "," not in spec:
        a, b = spec.split("-")
        ks = list(range(int(a), int(b) + 1))
    else:
        ks = [int(x) for x in spec.split(",")]
    return [k for k in ks if 0 <= k < n][:: max(1, every)]


def cmd_window(a):
    study = Path(a.study)
    ensure_ignored(study)
    eff = Effect(study / "motion" / a.effect)
    src, public = a.source, a.public_url
    if src.startswith(("http://", "https://")):
        from motionre import fetch as sr
        media = study / "raw/media"
        path, info = sr.fetch(src, media)
        public = public or info.get("webpage_url") or src
    else:
        path = Path(src).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"no such file: {path}")
    path = Path(path)
    probe = fr.probe(path)
    rows = fr.extract(path, a.t0, a.t1, eff.frames_dir)
    meta = {"id": a.effect, "name": a.name or a.effect, "created": datetime.now().isoformat(timespec="seconds"),
            "source": {"path": str(path), "public_url": public, "sha12": sha12(path), **probe},
            "window": {"t0": a.t0, "t1": a.t1}, "fps": probe["fps"], "family": a.family}
    eff.save_meta(meta)
    eff.write("frames.json", {"fps": probe["fps"], "frames": rows})
    why = cloud_refusal(meta)
    show({"effect": str(eff.root), "frames": len(rows), "dup_frames": [r["i"] for r in rows if r["dup"]],
          "fps": probe["fps"], "cloud": "allowed" if not why else f"refused: {why}"})


def cmd_mask(a):
    eff = Effect(Path(a.effect))
    n = len(eff.frames())
    if a.alpha:
        show(masks.alpha(eff, Path(a.alpha), a.name or "alpha"))
        return
    if not a.prompt:
        raise SystemExit("give --prompt (SAM 3) or --alpha (local)")
    if a.import_dir:
        show(masks.import_cache(eff, Path(a.import_dir), a.prompt[0], a.import_usd, a.note or ""))
        return
    fr.ensure(eff)
    ks = frame_list(a.frames, a.every, n)
    for p in a.prompt:
        show(masks.sam3(eff, p, ks, workers=a.workers, dry_run=a.dry_run))


def cmd_measure(a):
    fr.ensure(Effect(Path(a.effect)))
    show(measure.measure(Effect(Path(a.effect)), a.source or None, a.min_score))


def parse_params(items):
    out = {}
    for kv in items or []:
        k, v = kv.split("=", 1)
        out[k] = float(v) if "." in v else int(v)
    return out


def cmd_link(a):
    show(linker.link(Effect(Path(a.effect)), parse_params(a.param)))


def cmd_sheet(a):
    fr.ensure(Effect(Path(a.effect)))
    print(sheet.sheet(Effect(Path(a.effect))))


def cmd_run(a):
    eff = Effect(Path(a.effect))
    fr.ensure(eff)
    show(measure.measure(eff, None, a.min_score))
    show(linker.link(eff, parse_params(a.param)))
    print(sheet.sheet(eff))


def cmd_spend(a):
    show(budget.report(Path(a.study) if a.study else None))


# ---------------------------------------------------------------- measuring, fitting, proving

def _xy(text):
    return [float(v) for v in text.split(",")]


def cmd_camera(a):
    eff = Effect(Path(a.effect))
    fr.ensure(eff)
    if a.anchors:
        show(camera.measure(eff, "anchors", a=_xy(a.anchors[0]), b=_xy(a.anchors[1]), within=a.within))
    else:
        show(camera.measure(eff, "features", exclude_sources=a.exclude_source or (),
                            exclude_boxes=[_xy(b) for b in a.exclude_box or ()]))


def cmd_track(a):
    eff = Effect(Path(a.effect))
    x, y = _xy(a.near)
    show(series.from_track(eff, a.name, x, y, a.within, a.fields.split(","), a.max_size))


def cmd_pixels(a):
    eff = Effect(Path(a.effect))
    fr.ensure(eff)
    show(pixels.measure(eff, a.name, rules=a.rule or (), value=a.value, box=_xy(a.box) if a.box else None,
                        ring=_xy(a.ring) if a.ring else None, within=a.within_source, within_dilate=a.within_dilate,
                        not_source=a.not_source, not_pad=a.not_pad, open_px=a.open,
                        path=a.path, path_auto=_xy(a.path_auto) if a.path_auto else None,
                        toward=_xy(a.toward) if a.toward else None, tube=a.tube, bridge=a.bridge, skip=a.skip,
                        origin=a.origin))


def cmd_fit(a):
    eff = Effect(Path(a.effect))
    if a.pulse:
        show(fitmod.fit_pulse(eff, a.quantity, a.t_from, a.t_to))
    else:
        show(fitmod.fit_move(eff, a.quantity, a.t_from, a.t_to, a.start, a.end))


def cmd_claims(a):
    show(claimsmod.claims(Effect(Path(a.effect)), a.tol_frames))


def cmd_compare(a):
    show(claimsmod.compare(Effect(Path(a.his)), Effect(Path(a.rebuild))))


def cmd_recipe(a):
    show(claimsmod.recipe(Effect(Path(a.effect)), a.set or ()))


def cmd_clean(a):
    show(fr.clean(Path(a.path), a.dry_run))


def cmd_replay(a):
    his, reb = Effect(Path(a.his)), Effect(Path(a.on))
    steps = his.meta().get("pipeline") or []
    if not steps:
        raise SystemExit(f"{his.root.name}: no recorded steps; measure it with camera/track/pixels/fit/claims first")
    mapping = dict(m.split("=", 1) for m in a.map or ())
    meta = reb.meta()
    meta["pipeline"] = []
    reb.save_meta(meta)
    for s in steps:
        argv = [s["cmd"], str(reb.root)] + [mapping.get(x, x) for x in s["argv"]]
        print("replay:", " ".join(argv[:1] + argv[2:]), file=sys.stderr)
        run(argv)
    out = {"replayed": len(steps), "on": str(reb.root)}
    if not a.keep_frames:
        out["clean"] = fr.clean(reb.root)
    show(out)


RECORDED = {"camera", "track", "pixels", "fit", "claims"}


# ---------------------------------------------------------------- self-test

def self_test():
    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        os.environ["MOTION_RE_LEDGER"] = str(tmp / "ledger.jsonl")
        study = tmp / "2026-10-test"
        eff = Effect(study / "motion/cards")
        eff.frames_dir.mkdir(parents=True)
        W, H, N = 540, 960, 30
        rng = np.random.default_rng(1)
        bg = (rng.integers(40, 70, (H, W, 3))).astype(np.uint8)
        src = eff.masks_dir / "alpha-synthetic"
        src.mkdir(parents=True)
        rows = []
        for i in range(N):
            img = bg.copy()
            parts = []
            # A: blue card sliding right fast (power3.out over frames 2-12), hidden at frame 8 (a hand in front)
            if i >= 2:
                p = min(1, (i - 2) / 10)
                x = int(40 + 300 * (1 - (1 - p) ** 4))
                if i != 8:
                    parts.append(((x, 200, x + 120, 380), (200, 90, 40)))
            # C: orange card flipping in at frame 5: edge-on (12 px wide) then full, travelling 70 px per frame
            if i >= 5:
                wdt = 12 if i == 5 else 110
                x = min(60 + 70 * (i - 5), 340)
                parts.append(((x, 560, x + wdt, 740), (40, 120, 230)))
            # D: a bigger white card with a black bar, landing on A's resting spot at frame 20 (A is gone by then)
            if i >= 20:
                parts.append(((330, 180, 480, 400), (235, 235, 235)))
            ms = []
            for (x0, y0, x1, y1), c in parts:
                cv2.rectangle(img, (x0, y0), (x1, y1), c, -1)
                if c == (235, 235, 235):
                    cv2.rectangle(img, (x0 + 20, y0 + 90), (x1 - 20, y0 + 120), (10, 10, 10), -1)
                m = np.zeros((H, W), np.uint8)
                cv2.rectangle(m, (x0, y0), (x1, y1), 255, -1)
                ms.append(m)
            if i >= 20:  # A leaves when D lands
                ms = [m for m, (b, c) in zip(ms, parts) if c != (200, 90, 40)]
                img2 = bg.copy()
                for (x0, y0, x1, y1), c in parts:
                    if c != (200, 90, 40):
                        cv2.rectangle(img2, (x0, y0), (x1, y1), c, -1)
                        if c == (235, 235, 235):
                            cv2.rectangle(img2, (x0 + 20, y0 + 90), (x1 - 20, y0 + 120), (10, 10, 10), -1)
                img = img2
            cv2.imwrite(str(eff.frame_png(i)), img)
            rec = {"frame": i, "source": "alpha:synthetic", "prompt": None, "scores": [], "masks": []}
            for k, m in enumerate(ms):
                cv2.imwrite(str(src / f"f{i:04d}--m{k}.png"), m)
                rec["masks"].append(f"f{i:04d}--m{k}.png")
                rec["scores"].append(1.0)
            (src / f"f{i:04d}.json").write_text(json.dumps(rec))
            rows.append({"i": i, "t": round(i / 30, 4), "dup": False})
        eff.save_meta({"id": "cards", "source": {"path": str(tmp / "x.mp4"), "public_url": "https://example.com/p"},
                       "window": {"t0": 0, "t1": 1}, "fps": 30})
        eff.write("frames.json", {"fps": 30, "frames": rows})
        measure.measure(eff)
        res = linker.link(eff)
        tracks = [t for t in eff.read("tracks.json")["tracks"] if not t["flicker"]]
        spans = sorted((t["first"], t["last"], t["n"]) for t in tracks)
        assert len(tracks) == 3, res
        assert spans[0] == (2, 19, 17), spans          # A: through the hidden frame 8, ends when D lands
        assert spans[1] == (5, 29, 25), spans          # C: includes its edge-on first frame
        assert spans[2] == (20, 29, 10), spans         # D: its own track, not A's continuation
        sheet.sheet(eff)
        # budget: refused whole before sending; spend is logged to the effect and the month ledger
        budget.log(eff, masks.SAM3, 9.9, frame=0)
        try:
            budget.check(eff, 0.2)
            raise AssertionError("over-cap run was not refused")
        except budget.OverBudget:
            pass
        assert budget.study_spend(study) == 9.9 and budget.month_spend() >= 9.9
        # privacy: no public URL, or a private path, means no cloud
        assert cloud_refusal({"source": {"path": str(tmp / "x.mp4")}})
        assert cloud_refusal({"source": {"path": str(Path.home() / "Movies/x.mp4"), "public_url": "https://x"}})
        assert is_private_path(study.parent) is False
        self_test_proof(tmp)
    print("self-test ok")


def self_test_proof(tmp: Path):
    """The proof loop on a clip whose truth is known: a curved line drawing on sine.out (0.30 -> 1.20 s) under a 3%/s
    camera push. Measure it, then prove an identical rebuild ready and a linear one not."""
    import subprocess

    import cv2
    import numpy as np

    W, H, N, FPS = 640, 360, 45, 30
    rng = np.random.default_rng(7)
    ground = np.full((H, W, 3), 175, np.uint8)
    for _ in range(260):  # texture the camera can lock onto
        c = tuple(int(v) for v in rng.integers(120, 235, 3))
        cv2.circle(ground, (int(rng.integers(0, W)), int(rng.integers(0, H))), int(rng.integers(3, 14)), c, -1)
    arc = np.array([(160 + 320 * u, 250 - 130 * np.sin(u * np.pi / 2)) for u in np.linspace(0, 1, 400)])
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(arc, axis=0).T))]

    def clip(name, ease):
        d = tmp / f"src-{name}"
        d.mkdir()
        for i in range(N):
            t = i / FPS
            u = min(1, max(0, (t - 0.3) / 0.9))
            img = ground.copy()
            cv2.circle(img, (160, 250), 16, (250, 250, 250), -1)
            k = int(np.searchsorted(seg, ease(u) * seg[-1]))
            if k > 1:
                cv2.polylines(img, [arc[:k].astype(np.int32)], False, (25, 25, 25), 5)
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1 + 0.03 * t)
            cv2.imwrite(str(d / f"f{i:04d}.png"), cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT))
        mp4 = tmp / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(d / "f%04d.png"), "-c:v",
                        "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(mp4)], check=True)
        return mp4

    study = tmp / "2026-10-proof"
    sine = lambda u: np.sin(u * np.pi / 2)  # noqa: E731
    his, same, lin = clip("his", sine), clip("same", sine), clip("linear", lambda u: u)
    E = study / "motion"
    with open(os.devnull, "w") as quiet:
        old, sys.stdout = sys.stdout, quiet
        try:
            for name, src in (("his", his), ("same", same), ("linear", lin)):
                run(["window", str(src), "--study", str(study), "--effect", name, "--t0", "0", "--t1", str(N / FPS)])
            run(["camera", str(E / "his"), "--exclude-box", "140,100,520,280"])
            run(["pixels", str(E / "his"), "--name", "line", "--rule", "luma<70", "--rule", "dluma<-60", "--open", "3",
                 "--path-auto", "160,250", "--toward", "480,120", "--origin", "18", "--skip", "18", "--bridge", "20"])
            run(["fit", str(E / "his"), "line.drawn"])
            run(["claims", str(E / "his")])
            run(["replay", str(E / "his"), "--on", str(E / "same")])
            run(["replay", str(E / "his"), "--on", str(E / "linear")])
        finally:
            sys.stdout = old
    cam = json.loads((E / "his/camera.json").read_text())["summary"]
    assert abs(cam["zoom_per_s"] - 0.03) < 0.004, cam
    f = json.loads((E / "his/fits.json").read_text())["line.drawn"]
    assert f["ease"] == "sine.out", (f["best"], f["bic_gap"])
    b = f["fits"]["sine.out"]
    assert abs(b["t0"] - 0.3) <= 1.5 / FPS and abs(b["t1"] - 1.2) <= 1.5 / FPS, b
    ok_same = claimsmod.compare(Effect(E / "his"), Effect(E / "same"))
    ok_lin = claimsmod.compare(Effect(E / "his"), Effect(E / "linear"))
    assert ok_same["ready"], ok_same
    assert not ok_lin["ready"] and any(x["id"] == "line.drawn.ease" for x in ok_lin["failed"]), ok_lin
    r = claimsmod.recipe(Effect(E / "his"), ["beads.gap_px=24"])
    rec = json.loads((E / "his/recipe.json").read_text())
    assert r["ready"] and rec["manual"]["beads.gap_px"] == 24 and rec["moves"][0]["measured_by"].startswith("pixels")
    # frames are a cache: cleaned away, then re-extracted from the source when a measure needs them
    assert fr.clean(E / "his")["freed_mb"] > 0 and not list((E / "his/raw/frames").glob("*.png"))
    assert fr.ensure(Effect(E / "his")) and len(list((E / "his/raw/frames").glob("*.png"))) == N


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    w = sub.add_parser("window")
    w.add_argument("source")
    w.add_argument("--study", required=True)
    w.add_argument("--effect", required=True)
    w.add_argument("--t0", type=float, required=True)
    w.add_argument("--t1", type=float, required=True)
    w.add_argument("--name")
    w.add_argument("--public-url")
    w.add_argument("--family", action="store_true")
    m = sub.add_parser("mask")
    m.add_argument("effect")
    m.add_argument("--prompt", action="append")
    m.add_argument("--frames", default="all")
    m.add_argument("--every", type=int, default=1)
    m.add_argument("--workers", type=int, default=4)
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--alpha")
    m.add_argument("--name")
    m.add_argument("--import", dest="import_dir")
    m.add_argument("--import-usd", type=float, default=0.0)
    m.add_argument("--note")
    me = sub.add_parser("measure")
    me.add_argument("effect")
    me.add_argument("--source", action="append", help="mask source folder name under raw/masks (default: all)")
    me.add_argument("--min-score", type=float, default=0.5)
    li = sub.add_parser("link")
    li.add_argument("effect")
    li.add_argument("--param", action="append")
    sh = sub.add_parser("sheet")
    sh.add_argument("effect")
    r = sub.add_parser("run")
    r.add_argument("effect")
    r.add_argument("--min-score", type=float, default=0.5)
    r.add_argument("--param", action="append")
    sp = sub.add_parser("spend")
    sp.add_argument("--study")
    c = sub.add_parser("camera")
    c.add_argument("effect")
    c.add_argument("--anchors", nargs=2, metavar="X,Y")
    c.add_argument("--within", type=float, default=90.0)
    c.add_argument("--exclude-source", action="append")
    c.add_argument("--exclude-box", action="append")
    t = sub.add_parser("track")
    t.add_argument("effect")
    t.add_argument("--name", required=True)
    t.add_argument("--near", required=True, metavar="X,Y")
    t.add_argument("--within", type=float, default=90.0)
    t.add_argument("--fields", default="cx,cy,w,h")
    t.add_argument("--max-size", type=float, default=1e9)
    px = sub.add_parser("pixels")
    px.add_argument("effect")
    px.add_argument("--name", required=True)
    px.add_argument("--rule", action="append")
    px.add_argument("--value", choices=["luma", "r", "g", "b", "red", "sat"])
    px.add_argument("--box")
    px.add_argument("--ring")
    px.add_argument("--within-source")
    px.add_argument("--within-dilate", type=int, default=41)
    px.add_argument("--not-source")
    px.add_argument("--not-pad", type=int, default=12)
    px.add_argument("--open", type=int, default=0)
    px.add_argument("--path")
    px.add_argument("--path-auto", metavar="X,Y")
    px.add_argument("--toward", metavar="X,Y")
    px.add_argument("--tube", type=float, default=6.0)
    px.add_argument("--bridge", type=float, default=40.0)
    px.add_argument("--skip", type=float, default=0.0)
    px.add_argument("--origin", type=float, default=0.0, help="px along the path where drawn = 0 (a badge edge)")
    fi = sub.add_parser("fit")
    fi.add_argument("effect")
    fi.add_argument("quantity")
    fi.add_argument("--pulse", action="store_true")
    fi.add_argument("--from", dest="t_from", type=float)
    fi.add_argument("--to", dest="t_to", type=float)
    fi.add_argument("--start", type=float)
    fi.add_argument("--end", type=float)
    cl = sub.add_parser("claims")
    cl.add_argument("effect")
    cl.add_argument("--tol-frames", type=float, default=claimsmod.TOL_FRAMES)
    rp = sub.add_parser("replay")
    rp.add_argument("his")
    rp.add_argument("--on", required=True)
    rp.add_argument("--map", action="append")
    rp.add_argument("--keep-frames", action="store_true")
    co = sub.add_parser("compare")
    co.add_argument("his")
    co.add_argument("rebuild")
    re_ = sub.add_parser("recipe")
    re_.add_argument("effect")
    re_.add_argument("--set", action="append")
    cn = sub.add_parser("clean")
    cn.add_argument("path")
    cn.add_argument("--dry-run", action="store_true")
    return ap


CMDS = {"window": cmd_window, "mask": cmd_mask, "measure": cmd_measure, "link": cmd_link, "sheet": cmd_sheet,
        "run": cmd_run, "spend": cmd_spend, "camera": cmd_camera, "track": cmd_track, "pixels": cmd_pixels,
        "fit": cmd_fit, "claims": cmd_claims, "replay": cmd_replay, "compare": cmd_compare, "recipe": cmd_recipe,
        "clean": cmd_clean}


def run(argv):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.cmd not in CMDS:
        return ap.print_help()
    CMDS[a.cmd](a)
    if a.cmd in RECORDED:  # the step, with every argument after the effect, so replay runs it the same way
        name = getattr(a, "name", None) or getattr(a, "quantity", None) or ""
        series.record_step(Effect(Path(a.effect)), a.cmd, name, {})
        meta = Effect(Path(a.effect)).meta()
        meta["pipeline"][-1]["argv"] = argv[2:]
        Effect(Path(a.effect)).save_meta(meta)


def main():
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
