"""Turn points into files: per-point clips, a condensed video, and editor exports."""

from __future__ import annotations

import logging
from pathlib import Path

from courttally import media
from courttally import points as points_mod
from courttally.points import Point

log = logging.getLogger(__name__)


def cut_points(
    src: Path,
    run_dir: Path,
    points: list[Point],
    mode: str = "copy",
    condensed: bool = True,
    per_point: bool = True,
) -> dict:
    """Write ``points/point_NNN.mp4`` for each kept point and ``condensed.mp4`` joining them.

    ``mode="copy"`` keeps the original encoding (no quality loss, instant). Use
    ``"accurate"`` for frame-exact cuts at the cost of a high-quality re-encode.
    """
    kept = [p for p in points if p.keep]
    clip_dir = run_dir / "points"
    written: list[Path] = []
    for p in kept:
        dst = clip_dir / f"point_{p.index:03d}.mp4"
        media.cut(src, dst, p.start_s, p.end_s, mode=mode)
        written.append(dst)
    result = {"n_clips": len(written), "mode": mode, "clips": [str(w) for w in written]}
    if condensed and written:
        out = media.concat(written, run_dir / "condensed.mp4")
        result["condensed"] = str(out)
        if not per_point:
            for w in written:
                w.unlink()
            clip_dir.rmdir()
            result["clips"] = []
    log.info("cut %d points (%s) -> %s", len(written), mode, run_dir)
    return result


def write_editor_files(points: list[Point], run_dir: Path, fps: float, title: str) -> None:
    points_mod.save_csv(points, run_dir / "points.csv", fps)
    points_mod.save_edl(points, run_dir / "points.edl", fps, title=title)
