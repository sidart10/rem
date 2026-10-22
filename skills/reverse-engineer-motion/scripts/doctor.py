#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "opencv-python-headless>=4.9", "scipy>=1.11"]
# ///
"""Check that everything reverse-engineer-motion needs is in place. Prints one line per check; exits 1 on a blocker.

  uv run scripts/doctor.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
ok = True


def line(state, what, hint=""):
    global ok
    ok = ok and state != "missing"
    print(f"{state:>8}  {what}" + (f"  ({hint})" if hint and state != "ok" else ""))


line("ok" if sys.version_info >= (3, 10) else "missing", f"python {sys.version.split()[0]}", "needs 3.10+")
for mod, pkg in (("numpy", "numpy"), ("cv2", "opencv-python-headless"), ("scipy", "scipy")):
    line("ok" if importlib.util.find_spec(mod) else "missing", pkg, "run with `uv run`, or pip install " + pkg)
for tool, hint in (("ffmpeg", "brew install ffmpeg / apt install ffmpeg"),
                   ("ffprobe", "comes with ffmpeg"),
                   ("yt-dlp", "only for URLs: brew install yt-dlp / pipx install yt-dlp")):
    found = shutil.which(tool)
    line("ok" if found else ("optional" if tool == "yt-dlp" else "missing"), tool, "" if found else hint)
line("ok" if os.environ.get("FAL_KEY") else "optional", "FAL_KEY",
     "" if os.environ.get("FAL_KEY") else "only for SAM 3 masks: https://fal.ai/dashboard/keys, then export FAL_KEY=...")
b = json.loads((SKILL / "references/budget.json").read_text())
ledger = Path(os.environ.get("MOTION_RE_LEDGER") or b["month_ledger"]).expanduser()
print(f"    info  budget ${b['per_study_usd']:.0f}/study, ${b['per_month_usd']:.0f}/month; ledger {ledger}")
print("ready: try `uv run scripts/motion_re.py --self-test`" if ok else "fix the missing items above first")
sys.exit(0 if ok else 1)
