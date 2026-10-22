# Contributing

The skill lives in `skills/reverse-engineer-motion/` (`SKILL.md` is the method; `scripts/motion_re.py` the CLI;
`scripts/motionre/` the stages).

- Run `uv run skills/reverse-engineer-motion/scripts/motion_re.py --self-test` after any change. It runs the whole
  proof loop on a synthetic clip with a known ease, offline.
- Keep the proof rules in `SKILL.md`: every number has a basis, unknown never passes, ready means a measured render.
- Never commit study media, frames or masks (`raw/`, `studies/`). They're other people's pixels.
- Cloud calls go through the budget check (`motionre/budget.py`) and the public-footage check (`motionre/study.py`).
