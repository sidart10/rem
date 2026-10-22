#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "opencv-python-headless>=4.9", "scipy>=1.11", "matplotlib>=3.8"]
# ///
"""Explainer figures drawn from a study's own numbers, for a write-up, a README or a post.

  explain_figures.py curve <his effect> <rebuild effect> [..] --quantity line-right.drawn --labels "Original" "v1" ..
                     --out curve.png [--title ..] [--subtitle ..]
  explain_figures.py eases <effect> --quantity line-right.drawn --out eases.png [--title ..]
  explain_figures.py scorecard <rebuild effect> --out scorecard.png [--title ..] [--group line-left="Left line" ..]
  explain_figures.py tracking <effect> --track e4 --frames 9-13 --at 20 --out tracking.png [--title ..]
  explain_figures.py camera-drift <effect> --out drift.png [--exclude-box x0,y0,x1,y1 ..] [--title ..]

Every figure reads the files the pipeline wrote (series.json, fits.json, claims.json, compare.json, tracks.json) or
measures the frames again (camera-drift registers every frame both ways: chained frame to frame, and straight to
frame 0). Nothing is typed in by hand, so a figure can be rebuilt after a remeasure. `tracking` draws boxes, not
pixels, so it can be shared without the creator's frames. Dark, 1600 px wide, Inter when installed.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

from motionre import fit as fitmod  # noqa: E402
from motionre import series  # noqa: E402
from motionre.study import Effect  # noqa: E402

BG, PANEL, INK, MUTED, GRID = "#0E0E0E", "#16161A", "#EDEDED", "#8A8A8A", "#2A2A30"
ROWS = ["#FF9640", "#5AAAFF", "#8CDC78", "#C88CFF"]   # the side-by-side's row colours: original first
PASS, FAIL, UNKNOWN = "#3DDC84", "#FF5A5F", "#6B6B73"


def style():
    for f in [Path.home() / "Library/Fonts/Inter-Variable.ttf", Path("/Library/Fonts/Inter-Variable.ttf")]:
        if f.exists():
            font_manager.fontManager.addfont(str(f))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(f)).get_name()
            break
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 13,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": BG})


def header(fig, title, subtitle=None):
    fig.text(0.04, 0.955, title, fontsize=24, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.04, 0.895, subtitle, fontsize=14, color=MUTED, va="top")


def save(fig, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return {"png": str(out)}


def progress(effect: Effect, quantity: str, lo: float, hi: float):
    t, v, _ = series.get(effect, quantity)
    ok = ~np.isnan(v)
    return t[ok], (v[ok] - lo) / (hi - lo)


# ---------------------------------------------------------------- curve: his checkpoints against each rebuild

def curve(a):
    his = Effect(Path(a.effects[0]))
    f = his.read("fits.json")[a.quantity]
    claims = {c["what"]: c for c in his.read("claims.json")["claims"] if c["quantity"] == a.quantity}
    fps = his.meta()["fps"]
    labels = a.labels or [Path(e).name for e in a.effects]
    fig = plt.figure(figsize=(16, 9))
    header(fig, a.title or f"{a.quantity}: progress over time",
           a.subtitle or "Bands: the ±1.5-frame pass zone around each of the original's checkpoints")
    ax = fig.add_axes([0.07, 0.1, 0.62, 0.72])
    levels = [0.1, 0.5, 0.9, 0.95]
    for q in levels:
        c = claims.get(f"t{int(q * 100)}")
        if c:
            ax.add_patch(Rectangle((c["value"] - c["tol"], q - 0.022), 2 * c["tol"], 0.044, color=ROWS[0], alpha=0.38,
                                   lw=0))
    notes = []
    for k, e in enumerate(a.effects):
        t, p = progress(Effect(Path(e)), a.quantity, f["start"], f["end"])
        col = ROWS[k % len(ROWS)]
        ax.plot(t, p, color=col, lw=3 if k == 0 else 2.2, marker="o" if k == 0 else None, ms=4, label=labels[k],
                zorder=3 if k == 0 else 4)
        if k == 0:
            continue
        t50 = fitmod._crossing(t, p, 0.5)
        if t50 is not None and claims.get("t50"):
            d = (t50 - claims["t50"]["value"]) * fps
            notes.append((labels[k], d, col))
            ax.plot([t50], [0.5], "o", color=col, ms=9, zorder=5, mec=BG, mew=1.5)
    if claims.get("t50"):
        ax.plot([claims["t50"]["value"]], [0.5], "o", color=ROWS[0], ms=11, zorder=6, mec=BG, mew=1.5)
    ax.set_xlabel("seconds from the start of the window")
    ax.set_ylabel("share of the original's travel drawn")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK, fontsize=14)
    side = fig.add_axes([0.73, 0.1, 0.24, 0.72])
    side.axis("off")
    side.set_facecolor(BG)
    y = 0.95
    side.text(0, y, "Halfway point", fontsize=17, fontweight="bold", color=INK, va="top")
    y -= 0.1
    if claims.get("t50"):
        side.text(0, y, f"{labels[0]}: {claims['t50']['value']:.2f} s", fontsize=15, color=ROWS[0], va="top")
        y -= 0.08
    for lab, d, col in notes:
        verdict = "pass" if abs(d) <= claims["t50"]["tol"] * fps + 1e-9 else "fail"
        side.text(0, y, f"{lab}: {d:+.1f} frames ({verdict})", fontsize=15, color=col, va="top")
        y -= 0.08
    nice = lambda e: "none (constant speed)" if e == "none" else e  # noqa: E731
    eases = [f"{labels[0]}: {nice(f['ease'])}"]
    for k, e in enumerate(a.effects[1:], 1):
        rf = Effect(Path(e)).read("fits.json", {}).get(a.quantity)
        if rf:
            eases.append(f"{labels[k]}: {nice(rf['ease'])}")
    y -= 0.05
    side.text(0, y, "Fitted curve", fontsize=17, fontweight="bold", color=INK, va="top")
    for k, s in enumerate(eases):
        side.text(0, y - 0.1 - k * 0.08, s, fontsize=15, color=ROWS[k % len(ROWS)], va="top")
    return save(fig, a.out)


# ---------------------------------------------------------------- eases: all 21 fits, ranked

def eases(a):
    eff = Effect(Path(a.effects[0]))
    f = eff.read("fits.json")[a.quantity]
    t, p = progress(eff, a.quantity, f["start"], f["end"])
    ranked = sorted(f["fits"], key=lambda k: f["fits"][k]["bic"])
    best = f["fits"][ranked[0]]["bic"]
    fig = plt.figure(figsize=(16, 10))
    gap = f["fits"][ranked[1]]["bic"] - best
    header(fig, a.title or f"21 curves fitted to {a.quantity}",
           a.subtitle or (f"Ranked by BIC (lower is better). A curve is named only when it beats the next by 10 or "
                          f"more. Here {ranked[0]} wins by {gap:.0f}, so the claim is {f['ease']}."))
    cols, rows = 7, 3
    for i, fam in enumerate(ranked):
        r, c = divmod(i, cols)
        ax = fig.add_axes([0.04 + c * 0.136, 0.6 - r * 0.265, 0.12, 0.2])
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        fi = f["fits"][fam]
        u = np.clip((t - fi["t0"]) / max(1e-9, fi["t1"] - fi["t0"]), 0, 1)
        ax.scatter(t, p, s=6, color=MUTED, zorder=2)
        win = i == 0
        ax.plot(t, fitmod.EASES[fam](u), color=PASS if win else ROWS[1], lw=2.6 if win else 1.6, zorder=3)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(PASS if win else GRID)
            s.set_linewidth(2.5 if win else 0.8)
        ax.set_title(f"{fam}", fontsize=13, color=INK if win else MUTED, loc="left", pad=4)
        ax.text(0.97, 0.06, "best" if win else f"+{fi['bic'] - best:.0f}", transform=ax.transAxes, ha="right",
                fontsize=12, color=PASS if win else MUTED)
    return save(fig, a.out)


# ---------------------------------------------------------------- scorecard: every claim of one comparison

def scorecard(a):
    reb = Effect(Path(a.effects[0]))
    cmp_ = reb.read("compare.json")
    groups = dict(g.split("=", 1) for g in a.group or ())
    his = Effect(reb.root.parent / cmp_["his"])

    def unit(q):
        try:
            u = series.get(his, q)[2].get("unit", "")
        except SystemExit:
            return ""
        return "" if u in ("", None, "x", "ratio") else f" {u}"
    by = {}
    for r in cmp_["rows"]:
        q, what = r["id"].rsplit(".", 1)
        by.setdefault(q, []).append((what, r))
    fig = plt.figure(figsize=(16, 9))
    header(fig, a.title or f"{cmp_['passed']} of {cmp_['of']} claims pass",
           a.subtitle or f"{cmp_['rebuild']} against {cmp_['his']}. Unknown never passes.")
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.78])
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, len(by) + 0.2)
    for i, (q, items) in enumerate(by.items()):
        y = len(by) - i - 0.9
        name = next((v for k, v in groups.items() if q.startswith(k)), q)
        ax.text(0, y + 0.42, name, fontsize=16, fontweight="bold", color=INK, va="center")
        for j, (what, r) in enumerate(items):
            col = {"pass": PASS, "fail": FAIL}.get(r["result"], UNKNOWN)
            x = 2.2 + j * 0.98
            ax.add_patch(FancyBboxPatch((x, y + 0.08), 0.88, 0.7, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc=col, ec="none", alpha=0.9 if r["result"] != "pass" else 0.75))
            ax.text(x + 0.44, y + 0.56, what, ha="center", va="center", fontsize=13, color=BG, fontweight="bold")
            if r.get("delta_frames") is not None:
                d = f"{r['delta_frames']:+.1f} fr"
            elif r.get("delta") is not None:
                d = f"{r['delta']:+.3g}{unit(q) if what in ('start', 'end') else ''}"
            else:
                d = str(r.get("ours"))[:9] if r["result"] != "unknown" else "unknown"
            ax.text(x + 0.44, y + 0.28, d, ha="center", va="center", fontsize=11.5, color=BG)
    return save(fig, a.out)


# ---------------------------------------------------------------- tracking: boxes and gate numbers, no pixels

def tracking(a):
    eff = Effect(Path(a.effects[0]))
    tr = eff.read("tracks.json")
    meta = eff.meta()
    W, H = meta["source"]["width"], meta["source"]["height"]
    f0, f1 = (int(x) for x in a.frames.split("-"))
    tracks = {t["id"]: t for t in tr["tracks"]}
    focus = tracks[a.track]
    palette = ["#FF9640", "#5AAAFF", "#8CDC78", "#C88CFF", "#FFD166", "#FF6FB5", "#4DD6C9", "#B0B0FF"]
    colour = {tid: palette[i % len(palette)] for i, tid in enumerate(tracks)}
    fig = plt.figure(figsize=(16, 9))
    header(fig, a.title or f"Tracking {a.track}", a.subtitle)
    # left: every track's box at one frame
    ax = fig.add_axes([0.04, 0.07, min(0.36, 0.72 * 9 / 16 * W / H), 0.72])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Every track at frame {a.at}", fontsize=15, color=INK, loc="left")
    for tid, t in tracks.items():
        row = next((r for r in t["rows"] if r["f"] == a.at), None)
        if row:
            x0, y0, x1, y1 = row["box"]
            hot = tid == a.track
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=hot, fc=colour[tid] + "33", ec=colour[tid] if hot else MUTED,
                                   lw=3.5 if hot else 1.5))
            ax.text(x0 + 8, y0 - 18, tid, color=colour[tid] if hot else MUTED, fontsize=14 if hot else 12)
    # right: the focus track over its frames, plus the stitch it needed
    bx = fig.add_axes([0.42, 0.07, 0.55, 0.72])
    bx.set_title(f"{a.track} from frame {f0} to {f1}: width and height of its box", fontsize=15, color=INK, loc="left")
    rows = [r for r in focus["rows"] if f0 <= r["f"] <= f1]
    fr = [r["f"] for r in rows]
    bx.plot(fr, [r["w"] for r in rows], color=ROWS[1], lw=2.6, marker="o", label="width")
    bx.plot(fr, [r["h"] for r in rows], color=ROWS[0], lw=2.6, marker="o", label="height")
    bx.set_xlabel("frame"); bx.set_ylabel("px")
    bx.set_ylim(0, max(r["h"] for r in rows) * 1.25)
    bx.legend(loc="lower right", frameon=False, labelcolor=INK, fontsize=14)
    st = next((j for j in focus["joins"] if "stitched_at" in j), None)
    if st:
        bx.axvline(st["stitched_at"] - 0.5, color=MUTED, ls="--", lw=1.2)
        bx.text(st["stitched_at"] - 0.4, max(r["h"] for r in rows) * 1.18,
                f"stitched at frame {st['stitched_at']}: position {st['pos']:.2f} (gate {st['pos_gate']:.2f}), "
                f"size {st['size']:.3f}, look {st['app']:.2f}", fontsize=13, color=INK, va="top")
    return save(fig, a.out)


# ---------------------------------------------------------------- camera drift: chained against frame 0

def camera_drift(a):
    import cv2

    from motionre import frames as frmod
    eff = Effect(Path(a.effects[0]))
    meta = eff.meta()
    with tempfile.TemporaryDirectory() as tmp:
        rows = frmod.extract(Path(meta["source"]["path"]), meta["window"]["t0"], meta["window"]["t1"], Path(tmp))
        imgs = [cv2.imread(str(Path(tmp) / f"f{r['i']:04d}.png"), cv2.IMREAD_GRAYSCALE) for r in rows]
    H, W = imgs[0].shape
    mask = np.full((H, W), 255, np.uint8)
    for b in a.exclude_box or ():
        x0, y0, x1, y1 = (int(v) for v in b.split(","))
        mask[y0:y1, x0:x1] = 0
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    feats = [orb.detectAndCompute(g, mask) for g in imgs]

    def est(ka, kb):
        ms = bf.match(ka[1], kb[1])
        p0 = np.float32([ka[0][m.queryIdx].pt for m in ms])
        p1 = np.float32([kb[0][m.trainIdx].pt for m in ms])
        M, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=1.5)
        return np.vstack([M, [0, 0, 1]]) if M is not None else np.eye(3)

    scale = lambda M: float(np.hypot(M[0, 0], M[1, 0]))  # noqa: E731
    direct = [1.0] + [scale(est(feats[0], feats[i])) for i in range(1, len(feats))]
    chain, acc = [1.0], np.eye(3)
    for i in range(1, len(feats)):
        acc = est(feats[i - 1], feats[i]) @ acc
        chain.append(scale(acc))
    t = np.array([r["t"] - rows[0]["t"] for r in rows])
    direct, chain = np.array(direct), np.array(chain)
    drift = (chain / direct - 1) * 100
    fig = plt.figure(figsize=(16, 9))
    k = int(np.argmax(np.abs(drift)))
    header(fig, a.title or "Chained registration drifts",
           a.subtitle or f"Camera scale measured two ways. The chain ends {drift[-1]:+.1f}% off "
           f"({chain[-1]:.3f} against {direct[-1]:.3f}).")
    ax = fig.add_axes([0.07, 0.1, 0.88, 0.72])
    ax.plot(t, direct, color=PASS, lw=3, label="every frame to frame 0")
    ax.plot(t, chain, color=FAIL, lw=2.4, ls="--", label="each frame to the one before, chained")
    ax.annotate(f"{drift[k]:+.1f}%", (t[k], chain[k]), xytext=(-70, -30), textcoords="offset points",
                color=FAIL, fontsize=18, fontweight="bold", arrowprops=dict(arrowstyle="->", color=FAIL))
    ax.set_xlabel("seconds from the start of the window"); ax.set_ylabel("camera scale (1 = frame 0)")
    ax.legend(loc="upper left", frameon=False, labelcolor=INK, fontsize=14)
    res = save(fig, a.out)
    res.update(frames=len(t), drift_at_90=round(float(drift[min(90, len(t) - 1)]), 2), drift_max=round(float(np.abs(drift).max()), 2),
               scale_end_direct=round(float(direct[-1]), 4), scale_end_chain=round(float(chain[-1]), 4))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("figure", choices=["curve", "eases", "scorecard", "tracking", "camera-drift"])
    ap.add_argument("effects", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantity")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--title")
    ap.add_argument("--subtitle")
    ap.add_argument("--group", nargs="+")
    ap.add_argument("--track")
    ap.add_argument("--frames", default="0-10")
    ap.add_argument("--at", type=int, default=0)
    ap.add_argument("--exclude-box", nargs="+")
    a = ap.parse_args()
    style()
    fn = {"curve": curve, "eases": eases, "scorecard": scorecard, "tracking": tracking,
          "camera-drift": camera_drift}[a.figure]
    print(json.dumps(fn(a), indent=1))


if __name__ == "__main__":
    main()
