"""Turn per-frame ball boxes into a continuous track with explicit interpolation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from courttally.types import BallDetection, Box


def pick_ball(boxes: list[Box]) -> Box | None:
    """Highest-confidence box; later: prefer continuity with the previous position."""
    return max(boxes, key=lambda b: b.conf) if boxes else None


def to_detections(per_frame: dict[int, list[Box]], frame_indices: list[int]) -> list[BallDetection]:
    out: list[BallDetection] = []
    for idx in frame_indices:
        b = pick_ball(per_frame.get(idx, []))
        out.append(
            BallDetection(idx, None if b is None else b.center, 0.0 if b is None else b.conf)
        )
    return out


def interpolate(dets: list[BallDetection], max_gap: int = 5) -> list[BallDetection]:
    """Linearly fill gaps of at most ``max_gap`` missing frames; leave longer gaps None.

    A gap is only filled when it is bounded by real detections on both sides and
    its whole length fits in ``max_gap``; a 7-frame hole stays empty rather than
    being half filled.
    """
    xy = np.array([(np.nan, np.nan) if d.xy is None else d.xy for d in dets], dtype=np.float64)
    missing = np.isnan(xy[:, 0])
    filled = xy.copy()
    i = 0
    n = len(dets)
    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < n and missing[j]:
            j += 1
        gap = j - i
        if i > 0 and j < n and gap <= max_gap:
            t = np.arange(1, gap + 1) / (gap + 1)
            filled[i:j] = xy[i - 1] + t[:, None] * (xy[j] - xy[i - 1])
        i = j
    out: list[BallDetection] = []
    for d, was_missing, p in zip(dets, missing, filled, strict=True):
        if np.isnan(p[0]):
            out.append(BallDetection(d.frame_idx, None, 0.0))
        elif was_missing:
            out.append(
                BallDetection(d.frame_idx, (float(p[0]), float(p[1])), 0.0, interpolated=True)
            )
        else:
            out.append(d)
    return out


def to_frame(dets: list[BallDetection]) -> pd.DataFrame:
    """Flat table: frame_idx, x, y, conf, interpolated. Saved as ``ball.parquet``."""
    return pd.DataFrame(
        {
            "frame_idx": [d.frame_idx for d in dets],
            "x": [np.nan if d.xy is None else d.xy[0] for d in dets],
            "y": [np.nan if d.xy is None else d.xy[1] for d in dets],
            "conf": [d.conf for d in dets],
            "interpolated": [d.interpolated for d in dets],
        }
    )
