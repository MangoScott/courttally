"""Generate ``review.html``: a local, no-server page for reviewing and tagging points.

Open it in any browser next to the proxy video. Keyboard driven so a whole session
takes a couple of minutes: step through points, mark who won, drop the junk, nudge
the edges, then export ``points.edited.json`` for ``ct cut`` and ``ct score``.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from courttally.points import Point

_TEMPLATE_PATH = resources.files("courttally") / "review_template.html"


def write_review_page(
    points: list[Point],
    run_dir: Path,
    video_rel: str,
    clip_name: str,
    duration_s: float,
    fps: float,
) -> Path:
    data = {
        "clip": clip_name,
        "duration_s": duration_s,
        "fps": fps,
        "points": [
            {
                "index": p.index,
                "start_s": p.start_s,
                "end_s": p.end_s,
                "winner": p.winner,
                "keep": p.keep,
                "serve": p.serve,
                "label": p.label,
                "tags": p.tags,
            }
            for p in points
        ],
    }
    template = _TEMPLATE_PATH.read_text()
    html = template.replace("__VIDEO__", video_rel).replace("__DATA__", json.dumps(data))
    out = run_dir / "review.html"
    out.write_text(html)
    return out
