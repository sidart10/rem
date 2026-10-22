---
name: reverse-engineer-motion
description: "Reverse-engineer how the elements in a finished video move (cards, type, UI, logos, lines, glows, layouts in any YouTube, Instagram or TikTok video) and prove a rebuild matches. Reads one effect window at native frame rate, masks every element per frame (SAM 3 on fal for public footage, or the alpha channel of your own RGBA renders, locally), links masks into tracks, measures the camera and what masks miss (thin lines drawing on, glows, reveals) locally, fits GSAP power, sine, expo and circ eases by BIC, writes claims, replays the same steps on a rebuild and passes or fails it claim by claim, and writes a motion recipe a builder can read. Use when someone wants a creator's effect learned or rebuilt from a reel or video, asks how an effect moves (timing, easing, stagger, flips), or needs a rebuilt effect checked against the original frame by frame. Not for a whole video's edit pacing or for building the graphic itself."
---

# Reverse-engineer motion

Reconstruct how an effect in a finished video moves, as precisely as the pixels allow, then prove your rebuild moves
the same way. Copying an effect from stills and written notes goes wrong quietly: entrances come out 2-4x slower,
directions get flipped, ease-outs read as linear, and nothing flags it. This skill measures instead, and it measures
your rebuild with the same code.

All commands below run from this skill's folder. `uv run` installs the Python dependencies on first use.

```bash
mre() { uv run scripts/motion_re.py "$@"; }
mre --self-test        # the whole proof loop on a synthetic clip with a known ease; no network, no cost
uv run scripts/doctor.py   # checks ffmpeg, yt-dlp and FAL_KEY
```

## The rules (the proof method)

- **Every number has a basis.** `observed` means read off pixels. `derived` means fitted or computed from observed
  values. `inferred` means a judgement call.
- **Claims are made on what can be seen:** t10/t50/t90/t95 of a move, overshoot, start and end values, stagger. An ease
  name is a claim only when its fit clearly beats the next family (a large BIC gap); otherwise the ease is `unknown`.
  GSAP's nominal duration is never a claim, because many ease and duration pairs look the same.
- **Each claim is checked on its own.** Anything missing on either side is `unknown`, and unknown never passes.
- **Ready means rendered.** An effect is learned only when a render of the rebuild, measured by this same pipeline,
  passes every claim. A person watching both side by side still decides; the claims are the floor.
- **Cloud: public footage only.** SAM 3 runs on fal, so frames leave the machine. `effect.json` must name the public
  URL the clip came from. A local file counts only with `--public-url <the post it came from>`. Files under
  `~/Pictures`, `~/Movies`, `/Volumes` or any folder in `MOTION_RE_PRIVATE_DIRS` (separated by `:`) never go out.
  Your own renders are measured locally from their alpha channel.
- **Finish the effect that was picked before offering another.** Measuring is not the deliverable: take the pick
  through measure, rebuild, measure the rebuild with the same code, and a side-by-side the user can watch.
- **Budget: $10 per study and $40 per month by default** (`references/budget.json`, edit to change). A run is priced
  whole and refused whole before anything is sent. Every call is logged to the effect's `spend.jsonl` and to the month
  ledger (`~/.motion-re/spend.jsonl`, or `MOTION_RE_LEDGER`). Going over a cap needs the user's OK.

## Stages

| Stage | Status | Command |
|---|---|---|
| Window: native-rate frames with decoder timestamps and duplicate flags | built | `window` |
| Masks: SAM 3 image per frame (cloud) or alpha (local) | built | `mask` |
| Measure: box, centroid, rotated box, 4-corner fit, edge flag, colour, appearance | built | `measure` |
| Link: velocity-predicted gate, appearance, tracklet stitching, group moves, `inside` parts | built | `link`, `sheet` |
| Camera: every frame registered straight to frame 0 (features, or two fixed anchors) | built | `camera` |
| Series: track quantities by place, pixel-rule measures (area, edges, a channel's mean in a ring, a line's draw-on along its own path) | built | `track`, `pixels` |
| Fit: GSAP ease families by BIC plus the visible checkpoints; pulse shape | built | `fit` |
| Claims, replay, compare: claims with tolerances, the same steps replayed on a rebuild, per-claim pass or fail | built | `claims`, `replay`, `compare` |
| Recipe: the build-facing file a graphic's builder reads | built | `recipe` |
| Kinetic type: letters keyed locally, per-letter moves, arrivals, stacked echoes, streaks | built | `glyph_motion.py` |
| Perspective: rotateX/Y and tilt from the corners (flips can't be measured without it) | not yet | |

## How to run it

A study is any folder you pick for one creator or video; each effect lives in `<study>/motion/<slug>/`. Media,
frames and masks go to `raw/` folders that the tool git-ignores, so the creator's pixels never end up in a commit.

```bash
S=studies/2026-10-<creator>
E=$S/motion/<nn-slug>
mre window <post-url or file> --study $S --effect <nn-slug> --t0 4.03 --t1 5.38 [--public-url <post>]
mre mask $E --prompt card --frames 12,24     # try a prompt on 2 frames first ($0.01)
mre mask $E --prompt card                    # every frame
mre run $E                                   # measure + link + raw/tracks-sheet.jpg
# what moves, as quantities (each step is recorded in effect.json `pipeline`, so replay can repeat it)
mre camera $E --anchors 640,240 1370,764     # or: mre camera $E --exclude-source <moving element's masks>
mre track $E --name card --near 300,500 --fields cx,cy,w
mre pixels $E --name line --rule "luma<60" --rule "dluma<-40" --within-source sam3-dotted-line \
    --not-source sam3-octagon --not-pad 2 --open 3 --path-auto 898,609 --toward 640,240 --origin 74 --skip 138
mre pixels $E --name pulse --value red --rule "dluma>-40" --ring 898,609,65,136 --not-source sam3-octagon
mre fit $E line.drawn ; mre fit $E pulse.mean --pulse ; mre claims $E
# prove a rebuild: render it with a transparent background, window it as its own effect, alpha masks, replay
mre window <render.mp4> --study $S --effect <nn-slug>-rebuild-v1 --t0 0 --t1 <window length>
mre mask $S/motion/<nn-slug>-rebuild-v1 --alpha <layer.mov> --name badges   # one per element layer, free
mre run $S/motion/<nn-slug>-rebuild-v1
mre replay $E --on $S/motion/<nn-slug>-rebuild-v1 --map sam3-octagon=alpha-badges
mre compare $E $S/motion/<nn-slug>-rebuild-v1
mre recipe $E --set beads.gap_px=22.91       # recipe.json; hand-measured numbers go in `manual`
mre clean $S                                 # frame caches go; masks, numbers and recipes stay
mre spend --study $S
```

- **Pick the window from just before the effect's cue to just after it settles or leaves.** Every frame costs
  $0.005 per prompt.
- **Prompts matter.** On a storyboard-deck effect "card" found every card; "photo" over-detected; "poster" found
  nothing. On map badges, "octagon" and "flag" each dropped whole runs of frames; the two together held.
- **Check the sheet before trusting a track.** Each track has one colour. A track that changes colour mid-move, or two
  elements sharing a colour, is a wrong link. `tracks.json` records every stitch and group join with its costs, and
  every gate is in `params`, so a wrong link can be traced and tuned (`--param young_app_gate=0.35`).
- **For your own rebuild,** render with a transparent background and use `mask --alpha <render>`. It's free, local,
  and the truth is known, so it also calibrates the tracker.

## Measuring rules learned the hard way

- **Look at an overlay before trusting a number.** Draw the auto path and the passing pixels on the last frame
  (`paths/<name>.json`). Every bug below showed in a picture first.
- **SAM 3 can't see thin, half-drawn lines** until they're nearly complete. Measure those locally: undo the camera
  move, subtract frame 0, and walk the new pixels out along the line (`pixels --path-auto`).
- **Fit eases along the path, not as distance from a point.** A straight-line fit bends on a curved line.
- **Count a draw-on from where the line starts (`--origin`), not from where gaps stop being forgiven (`--skip`).**
  Dropping the first pixels throws away an ease-out's fast start.
- **Keep exclusions tight.** SAM's masks are already generous; a 12 px pad ate a line's slow tail and the ease read
  as linear. `--not-pad 2` fixed it.
- **Register every frame straight to frame 0.** A frame-to-frame chain drifted about 2% over 90 frames and invented a
  pan that wasn't there.
- **Look at the structure at full size before fitting anything.** One scale for a whole badge was wrong: the label
  never scaled, only its parts grew onto it. Measure each part's own geometry.
- **Time a rebuild at the original's levels,** and drive the rebuild in the recipe's units (px from the measuring
  origin), or the visible part runs at a different speed than the ease you set.
- **The duplicate check flags frames whose only change is a letter fading in.** Keep every frame for type effects.

## The hand-off to a builder

`recipe.json` is what a builder (HyperFrames, GSAP, Remotion, After Effects by hand) reads: each move's ease, or its
nearest family marked unproven; fitted start and settle times and levels; each pulse's levels and keyframes; the
camera's rate; a `manual` block for hand-measured numbers; and `ready`, true only once a rebuild passed every claim.
Read the numbers from the file; don't copy them by hand.

## Results so far

- A storyboard-deck effect (41 frames, $0.30): 6 real tracks and no wrong links, including cards seen edge-on and a
  card landing on top of another. The hand-written notes had the motion backwards: it's a fast flip-in from the left,
  one card every 0.075 s.
- A map effect with badges and beaded connector lines over a camera push (84 frames, $1.36): the pipeline reproduced a
  hand measurement within a frame (zoom 1.0860 against 1.0860). The first read called the lines linear; a linear
  rebuild measured 4 frames late at t50, which exposed a sine.out. The rebuild lands within about a frame on 15 of 16
  timing checkpoints and passes 26 of 29 claims; the rest is a masking difference in the last 5% of each line.

Files: `scripts/motion_re.py` (CLI and `--self-test`), `scripts/glyph_motion.py` (kinetic type),
`scripts/motionre/` (study, budget, frames, masks, measure, link, sheet, camera, series, pixels, fit, claims, glyphs,
fetch, fal_client), `scripts/doctor.py`, `references/budget.json`.
