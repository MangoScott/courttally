# CLAUDE.md — CourtTally

Open-source, single-phone-camera tennis analysis: ball tracking, court detection,
player tracking, shot stats, and line calling. Think SwingVision, built from public
research models, running offline first and on-device later.

## What we are building (in priority order)

The owner's priorities, in order: recording quality, point tracking (score),
automatic dead-time removal, and easy editing. Tracking models come after.

1. Recording: never degrade the original; ingest any phone format (HEVC, HDR,
   variable frame rate) into a proxy for analysis and cut exports from the
   original with stream copy. See `docs/recording.md`.
2. Dead-time removal: motion + audio activity → points → per-point clips and a
   condensed video, with no learned model in the loop.
3. Point tracking: a fast keyboard review page to tag who won each point and a
   tennis scoring state machine (ad / no-ad, tiebreaks, best-of-N).
4. Easy editing: CSV, CMX3600 EDL and JSON exports; trimming and re-cutting
   from the review page without re-running analysis.
5. Court detection → homography from camera view to a 2D court (meters).
6. Ball tracking → per-frame ball position + bounce detection.
7. Player detection/tracking → feet positions projected onto the 2D court.
8. Shot segmentation → speed, depth, placement, and who hit it.
9. Line calling → bounce point in/out using the court polygon + tolerance.
10. Session summary → stats JSON + CSV, annotated video, highlight clips.
11. (Later) iOS capture app; on-device Core ML / TFLite export.

## Non-negotiables

- Rec-court footage first. The target input is a phone on a tripod or fence behind
  the baseline, 1080p, 30 or 60 fps, any surface color, any lighting. Broadcast
  video is a nice-to-have, never the benchmark.
- Nothing is "working" until it runs end to end on a real clip in `data/clips/`
  and the output is visually checked in `outputs/`. Do not report accuracy numbers
  you did not measure. The synthetic clip from `scripts/make_synthetic_clip.py`
  is for plumbing checks only, never for claims about accuracy.
- Every pipeline stage must be runnable alone from the CLI and must save
  intermediate results (JSON/parquet) so later stages can be re-run without
  re-running detection.
- Keep models swappable behind the small interfaces in `src/courttally/types.py`
  (`Detector.detect(frame) -> list[Box]`, `KeypointModel.keypoints(frame) -> (14, 2)`).
  We will replace models as we learn.
- Model weights are never committed. `scripts/download_models.py` fetches them
  into `models/` (gitignored).
- Prefer boring, well-maintained libraries: PyTorch, ultralytics, OpenCV, NumPy,
  pandas. No new framework without asking. The CLI is plain `argparse`.

## Stack

- Python 3.11+, managed with `uv` (`uv sync`, `uv run ...`).
- Core deps are light (numpy, opencv-headless, pandas, pyarrow). The heavy ML
  stack is the `ml` extra: `uv sync --extra ml` (torch picks MPS on Apple
  Silicon, CUDA if available, CPU fallback).
- ultralytics (YOLOv8) for players; fine-tuned YOLO or TrackNetV3 for the ball;
  ResNet50 keypoint head for the 14 court keypoints.
- OpenCV for video I/O, homography, drawing.
- pytest for tests. ruff for lint/format.

## Prior art to borrow from (read before writing from scratch)

- abdullahtarek/tennis_analysis — the reference pipeline layout
  (trackers/, court_line_detector/, mini_court/, utils/). Mirror its structure.
- SinbadSails/SwingForge — MediaPipe pose + swing classification on top of the
  above. MIT. Good reference for stroke labeling.
- TrackNetV3 — ball tracking for small, fast objects. Use when YOLO ball
  detection drops frames on fast shots.
- yastrebksv/TennisCourtDetector — 14-keypoint court dataset and model.

Borrow ideas and structure freely. Copy code only when the license allows and
note the source in a comment.

## Repo layout

```
courttally/
  CLAUDE.md
  PLAN.md                     # phased task list, update as we go
  pyproject.toml
  docs/recording.md           # capture recipe + capture app requirements
  src/courttally/
    cli.py                    # `ct autoedit clip.mp4`, `ct cut`, `ct analyze`, ...
    pipeline.py               # stages + run-folder layout
    types.py                  # Box, detections, Detector / KeypointModel protocols
    video.py                  # frame reading/writing, fps, resizing
    media.py                  # ffmpeg: probe, proxy, audio, lossless cut, concat
    activity.py               # motion + audio activity signal per frame
    points.py                 # point segmentation, JSON/CSV/EDL
    score.py                  # tennis scoring state machine
    editor.py                 # review.html generator (tag winners, trim, drop)
    export.py                 # per-point clips + condensed.mp4
    court/                    # geometry (meters), keypoint model, homography
    ball/                     # detector, tracker, interpolation, bounce detection
    players/                  # YOLO detection, ID tracking, foot-point projection
    shots/                    # shot segmentation, speed/depth/placement, hitter
    calls/                    # in/out logic
    report/                   # stats JSON/CSV, annotated video, highlights
    viz/                      # mini-court drawing, overlays
  scripts/download_models.py
  scripts/make_synthetic_clip.py
  models/                     # gitignored
  data/clips/                 # a few short real clips (10-30 s), gitignored
  data/labels/                # any hand labels we create
  outputs/                    # run artifacts, gitignored
  tests/
```

Run folder (`outputs/<clip-stem>/`): `ingest.json`, `proxy.mp4`,
`activity.parquet`, `points.json` (+ `.csv`, `.edl`), `review.html`,
`points.edited.json` (from the review page), `points/point_NNN.mp4`,
`condensed.mp4`, `score.json`; and from the analysis pipeline `court.json`,
`ball.parquet`, `players.parquet`, `annotated.mp4`, `stats.json`.

ffmpeg is required for ingest/cut. A system `ffmpeg` is used when present;
otherwise the static binary from `imageio-ffmpeg` (a core dependency). `ffprobe`
is optional and only improves `ingest.json`.

## Court geometry (single source of truth)

All court math lives in `src/courttally/court/geometry.py` using ITF dimensions
in meters: 23.77 x 10.97 (doubles), singles width 8.23, service line 6.40 from
net. Court space is `(x_m, y_m)` with x=0 at the center line and y=0 at the net;
the near (camera) baseline is at y = -11.885 and draws at the bottom of the
mini-court. Everything downstream (depth, placement, line calls) uses these
coordinates, never pixels. The 14 keypoint order is `KEYPOINT_NAMES` there.

## Conventions

- Frame index is the universal key. Every detection carries `frame_idx`.
- Coordinates: pixel space is `(x, y)` floats; court space is `(x_m, y_m)`.
- Missing detections are `None` (NaN in tables), never 0,0. Interpolation is an
  explicit step and is flagged in the `interpolated` column.
- Functions over classes unless state is genuinely needed.
- Type hints everywhere. Docstrings say what and why, not how.
- Tests: unit tests for geometry/homography/line-call logic (pure math, fast).
  Slow integration tests run the full pipeline on `data/clips/smoke.mp4` (or a
  generated synthetic clip when it is absent) and are marked `@pytest.mark.slow`.
  `uv run pytest` skips slow tests; `uv run pytest -m slow` runs them.

## Working agreements with Claude Code

- Before a new stage: read the matching prior-art code, propose the approach in
  2-5 bullets, then build.
- After a stage: run it on at least two clips, write the annotated video to
  `outputs/`, and tell me what to look at and what looks wrong.
- When something fails on rec footage (it will), diagnose with saved frames
  before changing models or thresholds. Label the failure mode in PLAN.md.
- Do not spend more than ~30 minutes of effort on an optimization before showing
  me the current result.
- Keep PLAN.md current: check things off, add the failure modes you find.
