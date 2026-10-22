"""Join per-frame detections into element tracks. Masks have no identity; this gives them one.

Two passes, both judged on position, size and appearance together:

1. Forward. Each live track predicts where its element is in this frame from its last two observations (constant
   velocity, so a fast jump lands inside the gate instead of past it). The position gate is measured in element
   sizes and widens with speed. A track with only one observation has no velocity yet, so its gate is wide and
   appearance carries the decision. Size is compared on the longer side, which a card flipping edge-on keeps.
   Appearance is a 24x24 grey thumbnail (normalised correlation) plus an HSV histogram (Bhattacharyya), so a new
   element that lands on top of an old one (Justyn's HUGE card over the white text card) starts its own track
   instead of continuing the old one. Assignment is Hungarian over everything that passed the gates. A track may
   miss up to `max_gap` frames (a hand in front, a dropped mask) and continue.
2. Stitch. Tracklets that end and start within a few frames are joined when either one's motion, extrapolated
   across the gap, lands on the other and they look alike. This recovers single edge-on frames at the start of a
   flip (card 4's f9 in the deck), where the forward pass had no velocity to predict with.

Every gate and cost is in tracks.json `params`, and each join records its costs, so a wrong link can be traced.
A track inside another track's box on most shared frames is marked `inside` (a thumbnail within its card).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .study import Effect

PARAMS = {
    "max_gap": 2,            # frames a track may go unseen and continue
    "pos_gate": 0.35,        # element sizes, plus speed_slack x the predicted step
    "speed_slack": 0.6,
    "young_pos_gate": 1.6,   # element sizes, for a track with one observation (no velocity yet)
    "size_gate": 0.45,       # |log| ratio of the longer side (1.57x)
    "app_gate": 0.40,        # appearance distance, 0 same .. 1 unrelated
    "young_app_gate": 0.30,  # stricter, because the young position gate is wide
    "w_size": 1.0,
    "w_app": 2.0,
    "stitch_gap": 3,         # frames between tracklets the stitch pass bridges
    "inside": 0.9,           # box containment for `inside`
}


def maxdim(d):
    return max(d["w"], d["h"])


def thumb(d):
    if "_t" not in d:
        t = np.frombuffer(bytes.fromhex(d["thumb"]), np.uint8).astype(np.float32)
        d["_t"] = (t - t.mean()) / (t.std() + 1e-6)
    return d["_t"]


def app_dist(a, b) -> float:
    ncc = float(np.mean(thumb(a) * thumb(b)))
    bc = float(np.sum(np.sqrt(np.array(a["hist"]) * np.array(b["hist"]))))
    bhat = math.sqrt(max(0.0, 1.0 - bc))
    # 0 same .. 1 unrelated. The thumbnail counts half as much as colour: a flip squashes the thumbnail, not the colours.
    return round(0.25 * (1 - max(-1.0, min(1.0, ncc))) + 0.5 * bhat, 4)


class Track:
    def __init__(self, i, det):
        self.obs = {i: det}
        self.joins = []

    @property
    def first(self):
        return min(self.obs)

    @property
    def last(self):
        return max(self.obs)

    def pos(self, i):
        d = self.obs[i]
        return np.array([d["cx"], d["cy"]])

    def size(self, at_end=True):
        ks = sorted(self.obs)
        ks = ks[-3:] if at_end else ks[:3]
        return float(np.median([maxdim(self.obs[k]) for k in ks]))

    def velocity(self, at_end=True):
        ks = sorted(self.obs)
        if len(ks) < 2:
            return None
        a, b = (ks[-2], ks[-1]) if at_end else (ks[0], ks[1])
        if b - a > 3:
            return None
        return (self.pos(b) - self.pos(a)) / (b - a)


def gate_and_cost(track: Track, k: int, det: dict, p: dict):
    """(cost, parts) when det can continue track at frame k, else (None, parts)."""
    v = track.velocity()
    dt = k - track.last
    s = track.size()
    pred = track.pos(track.last) + (v * dt if v is not None else 0)
    dn = float(np.linalg.norm(pred - np.array([det["cx"], det["cy"]])) / s)
    last = track.obs[track.last]
    ds = abs(math.log(maxdim(det) / s))
    da = app_dist(last, det)
    if v is None:
        pg, ag = p["young_pos_gate"], p["young_app_gate"]
    else:
        pg, ag = p["pos_gate"] + p["speed_slack"] * float(np.linalg.norm(v)) * dt / s + 0.15 * (dt - 1), p["app_gate"]
    parts = {"pos": round(dn, 3), "pos_gate": round(pg, 3), "size": round(ds, 3), "app": round(da, 3), "app_gate": ag}
    if dn > pg or ds > p["size_gate"] or da > ag:
        return None, parts
    return dn + p["w_size"] * ds + p["w_app"] * da, parts


def similarity(p1, p2, q1, q2):
    """Scale, rotation and shift taking points p1, p2 to q1, q2 (complex-number form)."""
    a, b = complex(*p1), complex(*p2)
    c, d = complex(*q1), complex(*q2)
    if abs(b - a) < 1e-6:
        return None
    m = (d - c) / (b - a)
    return m, c - m * a


def group_move(tracks, dets, k, p):
    """Elements that jump together in one frame (a deck exiting, a punch-in, a whole layout scaling): find the one
    similarity transform that carries most unmatched tracks onto unmatched detections that look like them.
    Returns [(track, det_index)] and the transform, or ([], None)."""
    if len(tracks) < 2 or len(dets) < 2:
        return [], None
    pred = [t.pos(t.last) for t in tracks]
    ok = [[app_dist(t.obs[t.last], d) <= p["app_gate"] for d in dets] for t in tracks]
    pairs = [(a, b) for a in range(len(tracks)) for b in range(len(dets)) if ok[a][b]]
    best = ([], None, 1e9)
    for i, (a1, b1) in enumerate(pairs):
        for a2, b2 in pairs[i + 1:]:
            if a1 == a2 or b1 == b2:
                continue
            q1, q2 = (dets[b1]["cx"], dets[b1]["cy"]), (dets[b2]["cx"], dets[b2]["cy"])
            tr = similarity(pred[a1], pred[a2], q1, q2)
            if tr is None or not (0.33 < abs(tr[0]) < 3):
                continue
            m, sh = tr
            C = np.full((len(tracks), len(dets)), 1e6)
            for a, t in enumerate(tracks):
                z = m * complex(*pred[a]) + sh
                s = t.size() * abs(m)
                for b, d in enumerate(dets):
                    dn = abs(z - complex(d["cx"], d["cy"])) / s
                    ds = abs(math.log(maxdim(d) / s))
                    if ok[a][b] and dn <= p["pos_gate"] and ds <= p["size_gate"]:
                        C[a, b] = dn + ds
            got = [(a, b) for a, b in zip(*linear_sum_assignment(C)) if C[a, b] < 1e6]
            res = sum(C[a, b] for a, b in got)
            if len(got) > len(best[0]) or (len(got) == len(best[0]) and res < best[2]):
                best = (got, tr, res)
    got, tr, _ = best
    if len(got) < 2 or len(got) < 0.5 * len(tracks):
        return [], None
    return [(tracks[a], b) for a, b in got], tr


def forward(frames, p):
    tracks = []
    for fr in frames:
        k, dets = fr["i"], fr["dets"]
        live = [t for t in tracks if k - t.last <= p["max_gap"] + 1]
        used, matched = set(), set()
        if live and dets:
            C = np.full((len(live), len(dets)), 1e6)
            for a, t in enumerate(live):
                for b, d in enumerate(dets):
                    c, _ = gate_and_cost(t, k, d, p)
                    if c is not None:
                        C[a, b] = c
            for a, b in zip(*linear_sum_assignment(C)):
                if C[a, b] < 1e6:
                    live[a].obs[k] = dets[b]
                    used.add(b)
                    matched.add(id(live[a]))
            # what is left over may have moved together: one shared transform instead of one gate each
            rest_t = [t for t in live if id(t) not in matched and t.last == k - 1]
            rest_d = [b for b in range(len(dets)) if b not in used]
            got, tr = group_move(rest_t, [dets[b] for b in rest_d], k, p)
            for t, j in got:
                t.obs[k] = dets[rest_d[j]]
                t.joins.append({"group_at": k, "scale": round(abs(tr[0]), 4),
                                "shift": [round(tr[1].real, 1), round(tr[1].imag, 1)], "with": len(got)})
                used.add(rest_d[j])
        tracks += [Track(k, d) for b, d in enumerate(dets) if b not in used]
    return tracks


def stitch_cost(A: Track, B: Track, p):
    gap = B.first - A.last
    if gap < 1 or gap > p["stitch_gap"] + 1:
        return None, {}
    a, b = A.obs[A.last], B.obs[B.first]
    s = max(A.size(), B.size(at_end=False))
    pa, pb = A.pos(A.last), B.pos(B.first)
    errs = []
    vb = B.velocity(at_end=False)
    if vb is not None:  # B's motion run backwards across the gap
        errs.append((np.linalg.norm(pb - vb * gap - pa) / s, np.linalg.norm(vb) * gap / s))
    va = A.velocity()
    if va is not None:  # A's motion run forwards
        errs.append((np.linalg.norm(pa + va * gap - pb) / s, np.linalg.norm(va) * gap / s))
    if not errs:
        errs.append((np.linalg.norm(pb - pa) / s, 0.0))
    dn, step = min(errs)
    ds = abs(math.log(maxdim(b) / maxdim(a))) if min(len(A.obs), len(B.obs)) > 1 else abs(math.log(max(b["h"], b["w"]) / max(a["h"], a["w"])))
    da = app_dist(a, b)
    pg = p["pos_gate"] + p["speed_slack"] * step + 0.15 * (gap - 1)
    parts = {"gap": gap, "pos": round(float(dn), 3), "pos_gate": round(float(pg), 3), "size": round(ds, 3), "app": round(da, 3)}
    if dn > pg or ds > p["size_gate"] or da > p["app_gate"]:
        return None, parts
    return float(dn + p["w_size"] * ds + p["w_app"] * da), parts


def stitch(tracks, p):
    while True:
        best = None
        for A in tracks:
            for B in tracks:
                if A is B or B.first <= A.last:
                    continue
                c, parts = stitch_cost(A, B, p)
                if c is not None and (best is None or c < best[0]):
                    best = (c, A, B, parts)
        if best is None:
            return tracks
        _, A, B, parts = best
        A.obs.update(B.obs)
        A.joins += [{"stitched_at": B.first, **parts}] + B.joins
        tracks.remove(B)


def contained(a, b) -> float:
    """Share of box a inside box b."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    return iw * ih / max(1, (ax1 - ax0) * (ay1 - ay0))


ROW_KEYS = ("cx", "cy", "w", "h", "area", "box", "rot", "corners", "quad_fit", "edge", "score", "sources", "lab")


def link(effect: Effect, params: dict | None = None) -> dict:
    p = {**PARAMS, **(params or {})}
    det = effect.read("raw/detections.json")
    if not det:
        raise SystemExit("no detections.json; run `motion_re.py measure` first")
    times = {f["i"]: f["t"] for f in det["frames"]}
    dups = {f["i"] for f in det["frames"] if f["dup"]}
    tracks = stitch(forward(det["frames"], p), p)
    tracks.sort(key=lambda t: (t.first, t.obs[t.first]["cx"]))
    out = []
    for n, t in enumerate(tracks, 1):
        ks = sorted(t.obs)
        rows = [{"f": k, "t": times[k], "dup": k in dups, **{x: t.obs[k][x] for x in ROW_KEYS}} for k in ks]
        out.append({"id": f"e{n}", "first": ks[0], "last": ks[-1], "n": len(ks),
                     "missing": [k for k in range(ks[0], ks[-1] + 1) if k not in t.obs],
                     "flicker": len(ks) < 3, "joins": t.joins, "rows": rows})
    for a in out:  # `inside`: a part of another element (a thumbnail within its card)
        for b in out:
            if a is b:
                continue
            shared = [(ra, rb) for ra in a["rows"] for rb in b["rows"] if ra["f"] == rb["f"]]
            if shared and sum(contained(ra["box"], rb["box"]) >= p["inside"] and ra["area"] < rb["area"]
                              for ra, rb in shared) >= 0.8 * len(shared):
                a["inside"] = b["id"]
                break
    data = {"params": p, "frames": len(det["frames"]), "tracks": out}
    effect.write("tracks.json", data)
    return {"tracks": len(out), "real": sum(not t["flicker"] for t in out),
            "summary": [f"{t['id']} f{t['first']}-{t['last']} n={t['n']}" + (f" missing={t['missing']}" if t["missing"] else "")
                        + (" flicker" if t["flicker"] else "") + (f" inside={t['inside']}" if t.get("inside") else "")
                        + "".join(f" stitched@{j['stitched_at']}" if "stitched_at" in j else f" group@{j['group_at']}"
                                  for j in t["joins"]) for t in out]}
