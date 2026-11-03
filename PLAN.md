# PLAN.md — CourtTally

Phases are ordered so each one is useful on its own. Each ends with a demo on real
clips. Check items off as they land; add discovered failure modes under each phase.

## Phase 0 — Skeleton + smoke test (day 1)
- [x] `uv init`, pyproject, ruff, pytest, `src/courttally/` package, `ct` CLI entrypoint
- [x] `video.py`: read frames with OpenCV, write annotated MP4, downscale option
- [x] `scripts/download_models.py`: YOLOv8x, court keypoint weights, ball weights
      (court + ball URLs are placeholders until Phase 1/2 pick the exact checkpoints)
- [ ] Put 3 real clips in `data/clips/` (behind-baseline phone, 15-30 s each,
      different courts/lighting). Name one `smoke.mp4`.
- [x] `ct analyze <clip>` runs end to end with stub stages and
      writes `outputs/<run>/annotated.mp4` + `stats.json`
      (verified on the synthetic clip; still needs a real clip)
- [x] `scripts/make_synthetic_clip.py` for plumbing checks without footage
- [ ] Reference: clone abdullahtarek/tennis_analysis into `reference/` (gitignored)
      and get it running on `smoke.mp4` as a baseline. Note what breaks.

## Phase 1 — Court detection + homography
- [ ] 14-keypoint court model (ResNet50 head from yastrebksv dataset) behind `KeypointModel`
- [ ] Confirm `geometry.KEYPOINT_NAMES` order matches the yastrebksv label order
- [ ] Run once per N frames + RANSAC homography; cache to `court.json` (plumbing done)
- [x] `geometry.py` with ITF dimensions and `pixel_to_court()` / `court_to_pixel()`
- [x] Mini-court overlay drawn from the homography
- [x] Unit tests: known keypoints → known homography → round-trips within 1 cm
- Failure modes to expect: non-broadcast angle, partial court in frame, fences,
  shadows, green/blue/clay. Record which clips fail and why.

## Phase 2 — Ball tracking + bounces
- [ ] Fine-tuned YOLO ball detector (Roboflow tennis-ball dataset) as v1 behind `Detector`
- [x] Tracker: linear interpolation over gaps ≤ 5 frames; drop longer (Kalman later)
- [ ] Bounce detection from vertical-velocity sign change in court space
- [x] Save `ball.parquet` (frame_idx, x, y, conf, interpolated flag)
- [ ] Compare against TrackNetV3 on the fastest-serve clip; keep the better one
- Failure modes: ball lost against white lines/fence, motion blur on serves, two balls on court.

## Phase 3 — Player tracking
- [ ] YOLOv8 person detection, filter to the two on-court players by court polygon
- [ ] Persistent IDs (ByteTrack via ultralytics `track`)
- [x] Foot point = bottom-center of box → court coordinates (plumbing in `stage_annotate`)
- [ ] Player distance covered, average position per point
- Failure modes: ball kids, people behind fence, players occluding each other at net.

## Phase 4 — Shots + stats
- [ ] Shot = ball direction change near a player's position
- [ ] Attribute hitter by nearest player at contact
- [ ] Speed: court-space distance / time between contact and next bounce
- [x] Depth and placement zone helpers (short/mid/deep; left/center/right) in geometry
- [ ] Rally length, shots per player, unforced-ish error proxy (ball out + last hitter)
- [ ] Serve detection (ball starts above player, first shot of rally)
- [ ] `stats.json` + `shots.csv` export (match SwingVision's export shape loosely
      so users can compare)

## Phase 5 — Line calling
- [x] Bounce point vs court polygon with configurable tolerance (start 5 cm): `calls/inout.py`
- [x] Singles vs doubles lines as a flag
- [x] "too close to call" verdict rather than guessing
- [ ] Overlay IN/OUT label at bounce frame
- Test: hand-label 30 bounces across the clips; report agreement honestly.

## Phase 6 — Reports + highlights
- [ ] Annotated full video with mini-court, ball trail, shot speed labels (trail + mini-court done)
- [ ] Auto-cut highlight clips: longest rallies, fastest serves
- [ ] One-page HTML summary per session

## Later (not now)
- On-device: export court + ball models to Core ML, measure fps on iPhone
- iOS capture app (Swift) that records and uploads or runs locally
- Pickleball court geometry as a second sport
- Stroke classification (forehand/backhand/volley) via pose, see SwingForge
