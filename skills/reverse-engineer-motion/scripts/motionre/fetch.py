"""Fetch one public video with yt-dlp. No cookies, ever: a post that needs a login is not public footage."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def fetch(url: str, tmp: Path):
    """Download one public video with yt-dlp into tmp/videos; returns (path, info)."""
    vdir = Path(tmp) / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    cmd = ["yt-dlp", "--no-playlist", "--no-progress", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
           "--merge-output-format", "mp4", "--write-info-json", "-o", str(vdir / "%(extractor)s-%(id)s.%(ext)s"), url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit("yt-dlp is not installed (brew install yt-dlp, or pipx install yt-dlp). "
                 "Or download the clip yourself and pass the file with --public-url <the post>.")
    if r.returncode:
        sys.exit(f"yt-dlp could not fetch {url}: {r.stderr.strip()[-400:]}\n"
                 "If it needs a login, it isn't public footage; never export cookies.")
    vids = [p for p in vdir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".m4v"}]
    if not vids:
        sys.exit(f"yt-dlp finished but left no video in {vdir}")
    info = {}
    for j in vdir.glob("*.info.json"):
        d = json.loads(j.read_text())
        info = {k: d.get(k) for k in ("id", "title", "uploader", "uploader_id", "channel", "upload_date",
                                      "duration", "webpage_url", "extractor")}
        j.unlink()
    return vids[0], info
