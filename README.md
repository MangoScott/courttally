# CourtTally

Open-source tennis analysis from a single phone camera: court detection, ball and
player tracking, shot statistics and line calling. A SwingVision-style tool built
on public research models, offline first.

## Quick start

```bash
uv sync                      # core deps + dev tools
uv run pytest                # fast unit tests
uv run pytest -m slow        # full pipeline on data/clips/smoke.mp4 (or a synthetic clip)
```

Cut the dead time out of a session and tag the points (no ML models needed):

```bash
uv run ct autoedit session.mp4                 # -> outputs/session/
open outputs/session/review.html               # J/K step points, A/B who won, X drop, E export
uv run ct cut session.mp4 --points outputs/session/points.edited.json
uv run ct score session.mp4                    # tennis score progression from the tags
```

`autoedit` builds a proxy, finds points from motion + sound, writes
`points.json/.csv/.edl`, a review page, per-point clips and `condensed.mp4`.
Cuts are stream-copied from the original, so exports keep full capture quality.
See `docs/recording.md` for how to film.

Run the analysis pipeline on a clip:

```bash
uv run ct info data/clips/smoke.mp4
uv run ct analyze data/clips/smoke.mp4          # all stages -> outputs/smoke/
uv run ct annotate data/clips/smoke.mp4         # re-render from cached stage outputs
```

Each stage (`court`, `ball`, `players`, `annotate`, `stats`) is its own subcommand and
caches its result in `outputs/<clip>/`, so later stages re-run without re-detecting.

No footage yet? Generate a synthetic clip for plumbing checks:

```bash
uv run scripts/make_synthetic_clip.py data/clips/synthetic.mp4
uv run ct analyze data/clips/synthetic.mp4 --keypoints data/clips/synthetic.keypoints.json
```

Real models land in Phase 1+ and need the ML extra and weights:

```bash
uv sync --extra ml
uv run scripts/download_models.py
```

See `CLAUDE.md` for conventions and `PLAN.md` for the phased roadmap.
