"""Session summary: ``stats.json`` built from the saved stage outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from courttally import __version__
from courttally.video import VideoInfo


def build_stats(
    info: VideoInfo, ball: pd.DataFrame, players: pd.DataFrame, court_ok: bool
) -> dict[str, Any]:
    detected = int(ball["x"].notna().sum()) if len(ball) else 0
    return {
        "courttally_version": __version__,
        "clip": {
            "path": str(info.path),
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "n_frames": info.n_frames,
            "duration_s": round(info.duration_s, 2),
        },
        "court": {"homography_found": court_ok},
        "ball": {
            "frames_with_ball": detected,
            "frames_interpolated": int(ball["interpolated"].sum()) if len(ball) else 0,
            "coverage": round(detected / len(ball), 3) if len(ball) else 0.0,
        },
        "players": {
            "n_tracks": int(players["track_id"].nunique()) if len(players) else 0,
            "detections": int(len(players)),
        },
        # Filled by Phase 4+.
        "shots": [],
        "rallies": [],
        "calls": [],
    }


def write_stats(stats: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(stats, indent=2))
