"""reverse-engineer-motion: measure how elements move in a finished video, as claims a rebuild must pass.

Modules, one per stage:
  study    effect folder layout, effect.json, privacy (only public creator footage goes to the cloud)
  budget   per-study and per-month cloud caps, the spend ledger
  frames   native-rate frames of one effect window, with timestamps and duplicate flags
  masks    mask providers: SAM 3 image on fal (cloud, per frame) and alpha (local, from an RGBA render)
  measure  per-mask geometry (box, centroid, rotated box, corners, edge heights) and appearance
  link     masks joined into element tracks: velocity-predicted gate, appearance, tracklet stitching
  sheet    contact sheet with each track's masks in its own colour
  camera   every frame registered straight to frame 0 (features or two anchors), so later stages measure in frame 0
  series   measured quantities over time (series.json), the recorded steps replay runs, tracks picked by place
  pixels   pixel-rule measures in frame-0 coordinates: area and edges, a channel's mean, a line's draw-on along its path
  fit      GSAP ease families by BIC plus visible checkpoints; pulse shape
  claims   claims with tolerances, the per-claim compare of a rebuild, and recipe.json for builders
"""
