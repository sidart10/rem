"""Effect folder layout and the cloud privacy rule.

One effect window per folder, inside a creator study:

  <study>/motion/<effect>/
    effect.json        what was studied: source, provenance, window, fps, prompts (tracked)
    frames.json        every frame's index, time and duplicate flag (tracked)
    raw/detections.json every mask's measures and appearance thumbnails (git-ignored: thumbnails are his pixels)
    tracks.json        element tracks (tracked)
    spend.jsonl        one line per cloud call, with its cost (tracked)
    raw/               frames, masks and sheets: pixels of the creator's video, git-ignored (C13)

Cloud rule: only public footage goes to a cloud model. An effect is public only when effect.json names the public
URL it came from; a file under ~/Pictures, ~/Movies, an external volume or a folder listed in
MOTION_RE_PRIVATE_DIRS (separated by the path separator) is never public, whatever it says.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PLUGIN = Path(__file__).resolve().parents[2]  # the skill folder
PRIVATE_ROOTS = [Path.home() / "Pictures", Path.home() / "Movies", Path("/Volumes")] + [
    Path(p).expanduser().resolve() for p in os.environ.get("MOTION_RE_PRIVATE_DIRS", "").split(os.pathsep) if p]


class Effect:
    def __init__(self, root: Path):
        self.root = Path(root)

    raw = property(lambda s: s.root / "raw")
    frames_dir = property(lambda s: s.root / "raw/frames")
    masks_dir = property(lambda s: s.root / "raw/masks")
    spend_path = property(lambda s: s.root / "spend.jsonl")

    @property
    def study(self) -> Path:
        """The creator study folder (<study>/motion/<effect>), or the effect's parent when laid out otherwise."""
        return self.root.parent.parent if self.root.parent.name == "motion" else self.root.parent

    def meta(self) -> dict:
        p = self.root / "effect.json"
        if not p.exists():
            raise SystemExit(f"{self.root} has no effect.json; start it with `motion_re.py window`")
        return json.loads(p.read_text())

    def save_meta(self, meta: dict):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "effect.json").write_text(json.dumps(meta, indent=1) + "\n")

    def read(self, name, default=None):
        p = self.root / name
        return json.loads(p.read_text()) if p.exists() else default

    def write(self, name, data):
        (self.root / name).write_text(json.dumps(data, indent=1) + "\n")

    def frames(self) -> list:
        return self.read("frames.json", {}).get("frames", [])

    def frame_png(self, i: int) -> Path:
        return self.frames_dir / f"f{i:04d}.png"


def ensure_ignored(folder: Path):
    """The study keeps media in raw/; make sure git never sees it (a .gitignore beside it, as study_reference does)."""
    folder.mkdir(parents=True, exist_ok=True)
    gi = folder / ".gitignore"
    lines = gi.read_text().splitlines() if gi.exists() else []
    if "raw/" not in lines:
        gi.write_text("\n".join(lines + ["raw/"]) + "\n")


def is_private_path(path) -> bool:
    p = Path(path).expanduser().resolve()
    return any(p == r or r in p.parents for r in PRIVATE_ROOTS)


def cloud_refusal(meta: dict):
    """None when this effect may go to a cloud model, else the reason it may not."""
    if meta.get("family"):
        return "marked family or personal footage"
    src = meta.get("source", {})
    if src.get("path") and is_private_path(src["path"]):
        return f"the source file is under a private folder ({src['path']})"
    if not str(src.get("public_url", "")).startswith(("http://", "https://")):
        return "no public URL recorded for the source (pass --public-url when the window is made)"
    return None


def sha12(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]
