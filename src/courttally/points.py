"""Points: contiguous stretches of play, plus everything the editor attaches to them."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Point:
    index: int
    start_s: float
    end_s: float
    winner: str | None = None  # "A" (near player) | "B" (far player) | None
    keep: bool = True
    serve: str | None = None  # who served, "A"/"B"
    label: str = ""  # free text: "ace", "great rally", ...
    tags: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def segment(
    activity: pd.DataFrame,
    fps: float,
    on: float = 0.5,
    off: float = 0.25,
    min_len_s: float = 1.5,
    merge_gap_s: float = 1.5,
    pad_pre_s: float = 1.0,
    pad_post_s: float = 1.5,
) -> list[Point]:
    """Hysteresis thresholding of the activity signal into points.

    Play starts when activity rises above ``on`` and continues while it stays above
    ``off``. Nearby stretches are merged, short blips dropped, and each point is
    padded so the serve toss and the ball's final bounce are not clipped.
    """
    a = activity["activity"].to_numpy()
    t = activity["t_s"].to_numpy()
    duration = float(t[-1] + 1 / fps) if len(t) else 0.0
    raw: list[tuple[float, float]] = []
    active = False
    start = 0.0
    for ti, ai in zip(t, a, strict=True):
        if not active and ai >= on:
            active, start = True, ti
        elif active and ai < off:
            raw.append((start, ti))
            active = False
    if active:
        raw.append((start, duration))

    merged: list[list[float]] = []
    for s, e in raw:
        if merged and s - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out: list[Point] = []
    for s, e in merged:
        if e - s < min_len_s:
            continue
        s2, e2 = max(0.0, s - pad_pre_s), min(duration, e + pad_post_s)
        if out and s2 < out[-1].end_s:  # padding overlapped the previous point
            s2 = out[-1].end_s
        out.append(Point(len(out), round(s2, 3), round(e2, 3)))
    return out


def dead_time_s(points: list[Point], duration_s: float) -> float:
    return duration_s - sum(p.duration_s for p in points if p.keep)


# ---- serialization -------------------------------------------------------


def save_json(
    points: list[Point], path: str | Path, clip: str, duration_s: float, fps: float
) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "clip": clip,
                "duration_s": duration_s,
                "fps": fps,
                "points": [asdict(p) for p in points],
            },
            indent=2,
        )
    )


def load_json(path: str | Path) -> list[Point]:
    data = json.loads(Path(path).read_text())
    return [
        Point(**{k: v for k, v in p.items() if k in Point.__dataclass_fields__})
        for p in data["points"]
    ]


def _tc(seconds: float, fps: float) -> str:
    """SMPTE-style HH:MM:SS:FF timecode (non-drop)."""
    total = int(round(seconds * fps))
    f = total % int(round(fps))
    s = total // int(round(fps))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}:{f:02d}"


def save_csv(points: list[Point], path: str | Path, fps: float) -> None:
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "index",
                "start_s",
                "end_s",
                "duration_s",
                "start_tc",
                "end_tc",
                "winner",
                "serve",
                "keep",
                "label",
            ]
        )
        for p in points:
            w.writerow(
                [
                    p.index,
                    p.start_s,
                    p.end_s,
                    round(p.duration_s, 3),
                    _tc(p.start_s, fps),
                    _tc(p.end_s, fps),
                    p.winner or "",
                    p.serve or "",
                    int(p.keep),
                    p.label,
                ]
            )


def save_edl(points: list[Point], path: str | Path, fps: float, title: str = "CourtTally") -> None:
    """CMX3600 EDL: imports into DaVinci Resolve, Premiere and Final Cut (via converters).

    Each kept point becomes one event cutting the source into a continuous timeline.
    """
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    rec = 0.0
    n = 1
    for p in points:
        if not p.keep:
            continue
        src_in, src_out = _tc(p.start_s, fps), _tc(p.end_s, fps)
        rec_in, rec_out = _tc(rec, fps), _tc(rec + p.duration_s, fps)
        lines.append(f"{n:03d}  AX       V     C        {src_in} {src_out} {rec_in} {rec_out}")
        lines.append(
            f"* FROM CLIP NAME: point_{p.index:03d}" + (f" winner={p.winner}" if p.winner else "")
        )
        rec += p.duration_s
        n += 1
    Path(path).write_text("\n".join(lines) + "\n")


def activity_to_points(activity: pd.DataFrame, fps: float, **kw) -> list[Point]:
    if activity.empty:
        return []
    return segment(activity.assign(activity=activity["activity"].fillna(0.0)), fps, **kw)


def summary(points: list[Point], duration_s: float) -> dict:
    kept = [p for p in points if p.keep]
    return {
        "n_points": len(kept),
        "play_s": round(sum(p.duration_s for p in kept), 1),
        "dead_s": round(dead_time_s(points, duration_s), 1),
        "dead_fraction": round(dead_time_s(points, duration_s) / duration_s, 3)
        if duration_s
        else 0.0,
        "longest_point_s": round(max((p.duration_s for p in kept), default=0.0), 1),
        "mean_point_s": round(float(np.mean([p.duration_s for p in kept])), 1) if kept else 0.0,
    }
