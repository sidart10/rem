"""Cloud spend: refuse before sending anything that would pass the caps, log every call that was sent.

Caps live in references/budget.json ($10 per study, $40 per month by default). Every paid call is one line in the
effect's spend.jsonl and in the month ledger (~/.motion-re/spend.jsonl, or MOTION_RE_LEDGER). The study total
sums every effect's spend.jsonl under <study>/motion/; the month total sums the ledger for the calendar month
(local time). A run is priced whole before its first call and refused whole: no surprise halfway.
Going over a cap needs the user's say; the refusal tells the agent to ask.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from .study import PLUGIN, Effect

BUDGET = PLUGIN / "references/budget.json"
TZ = datetime.now().astimezone().tzinfo


class OverBudget(SystemExit):
    pass


def load() -> dict:
    return json.loads(BUDGET.read_text())


def price(endpoint: str) -> float:
    p = load()["prices"].get(endpoint)
    if not p:
        raise SystemExit(f"no price recorded for {endpoint} in {BUDGET}; check fal's model page and add it first")
    return float(p["usd_per_call"])


def month_ledger() -> Path:
    return Path(os.environ.get("MOTION_RE_LEDGER") or load()["month_ledger"]).expanduser()


def _lines(path: Path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def month_spend(now: datetime | None = None) -> float:
    ym = (now or datetime.now(TZ)).strftime("%Y-%m")
    return round(sum(r["usd"] for r in _lines(month_ledger()) if r["ts"][:7] == ym), 4)


def study_spend(study: Path) -> float:
    return round(sum(r["usd"] for p in Path(study).glob("motion/*/spend.jsonl") for r in _lines(p)), 4)


def check(effect: Effect, planned_usd: float):
    """Raise OverBudget when planned_usd would pass the study or the month cap."""
    b = load()
    s, m = study_spend(effect.study), month_spend()
    msgs = []
    if s + planned_usd > b["per_study_usd"] + 1e-9:
        msgs.append(f"study {effect.study.name}: ${s:.2f} spent + ${planned_usd:.2f} planned > ${b['per_study_usd']:.2f} cap")
    if m + planned_usd > b["per_month_usd"] + 1e-9:
        msgs.append(f"this month: ${m:.2f} spent + ${planned_usd:.2f} planned > ${b['per_month_usd']:.2f} cap")
    if msgs:
        raise OverBudget("Refused, nothing sent: " + "; ".join(msgs) + ". Ask the user before going over, or raise the cap in " + str(BUDGET) + ".")
    return {"study_usd": s, "month_usd": m, "planned_usd": round(planned_usd, 4)}


def log(effect: Effect, endpoint: str, usd: float, **fields):
    rec = {"ts": datetime.now(TZ).isoformat(timespec="seconds"), "study": effect.study.name, "effect": effect.root.name,
           "endpoint": endpoint, "usd": round(usd, 4), **fields}
    line = json.dumps(rec) + "\n"
    for p in (effect.spend_path, month_ledger()):
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(line)
    return rec


def report(study: Path | None = None) -> dict:
    b = load()
    out = {"month_usd": month_spend(), "month_cap_usd": b["per_month_usd"]}
    if study:
        out.update({"study": Path(study).name, "study_usd": study_spend(study), "study_cap_usd": b["per_study_usd"]})
    return out
