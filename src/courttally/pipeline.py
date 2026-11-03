"""End-to-end analysis with every stage runnable alone and cached on disk.

Run layout (``outputs/<run>/``):
  court.json       keypoints + homography
  ball.parquet     frame_idx, x, y, conf, interpolated   (pixel space)
  players.parquet  frame_idx, track_id, box, foot point   (pixel space)
  annotated.mp4    source video with mini-court overlay
  stats.json       session summary

Later stages read earlier results from disk, so re-running ``annotate`` after a
ball-model change does not re-run court detection.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from courttally import video
from courttally.ball import tracker as ball_tracker
from courttally.ball.detector import NoBall
from courttally.court import geometry
from courttally.court.detector import CourtResult, ManualKeypoints, NoCourt, detect_court
from courttally.players import detector as players_mod
from courttally.players.detector import NoPlayers
from courttally.report.stats import build_stats, write_stats
from courttally.types import Detector, KeypointModel, PlayerDetection
from courttally.viz import mini_court

log = logging.getLogger(__name__)


def run_dir_for(clip: Path, outputs: Path, run: str | None = None) -> Path:
    d = outputs / (run or clip.stem)
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_court(
    clip: Path,
    run_dir: Path,
    model: KeypointModel | None = None,
    keypoints_json: Path | None = None,
    n_samples: int = 5,
) -> CourtResult:
    info = video.probe(clip)
    if model is None:
        model = ManualKeypoints.from_json(keypoints_json) if keypoints_json else NoCourt()
    step = max(1, info.n_frames // n_samples)
    frames = [f for _, f in video.read_frames(clip, step=step, stop=step * n_samples)]
    result = detect_court(model, frames, (info.width, info.height))
    result.to_json(run_dir / "court.json")
    log.info(
        "court: %d/14 keypoints, homography=%s",
        int(np.isfinite(result.keypoints_px).all(axis=1).sum()),
        result.homography is not None,
    )
    return result


def stage_ball(
    clip: Path, run_dir: Path, model: Detector | None = None, max_gap: int = 5
) -> pd.DataFrame:
    model = model or NoBall()
    info = video.probe(clip)
    per_frame = {}
    indices = []
    for idx, frame in tqdm(video.read_frames(clip), total=info.n_frames, desc="ball", unit="f"):
        per_frame[idx] = model.detect(frame)
        indices.append(idx)
    dets = ball_tracker.interpolate(ball_tracker.to_detections(per_frame, indices), max_gap=max_gap)
    df = ball_tracker.to_frame(dets)
    df.to_parquet(run_dir / "ball.parquet", index=False)
    log.info("ball: %d/%d frames with a position", int(df["x"].notna().sum()), len(df))
    return df


def stage_players(clip: Path, run_dir: Path, model: Detector | None = None) -> pd.DataFrame:
    model = model or NoPlayers()
    info = video.probe(clip)
    dets: list[PlayerDetection] = []
    for idx, frame in tqdm(video.read_frames(clip), total=info.n_frames, desc="players", unit="f"):
        # Phase 0: no identity tracking; each box gets a per-frame id. Phase 3 adds ByteTrack.
        for i, box in enumerate(model.detect(frame)):
            dets.append(PlayerDetection(idx, i, box))
    df = players_mod.to_frame(dets)
    df.to_parquet(run_dir / "players.parquet", index=False)
    log.info("players: %d detections", len(df))
    return df


def stage_annotate(clip: Path, run_dir: Path, trail: int = 10) -> Path:
    """Render ``annotated.mp4`` from cached stage outputs."""
    info = video.probe(clip)
    court = CourtResult.from_json(run_dir / "court.json")
    ball = pd.read_parquet(run_dir / "ball.parquet").set_index("frame_idx")
    players = pd.read_parquet(run_dir / "players.parquet")
    by_frame = {k: g for k, g in players.groupby("frame_idx")} if len(players) else {}
    mc = mini_court.MiniCourt()
    h = court.homography
    out_path = run_dir / "annotated.mp4"
    recent: list[tuple[float, float]] = []
    with video.VideoWriter(out_path, info.fps) as w:
        for idx, frame in tqdm(
            video.read_frames(clip), total=info.n_frames, desc="annotate", unit="f"
        ):
            ball_m = players_m = None
            if h is not None:
                if idx in ball.index and not np.isnan(ball.loc[idx, "x"]):
                    ball_m = tuple(geometry.pixel_to_court(h, ball.loc[idx, ["x", "y"]].to_numpy()))
                    recent = (recent + [ball_m])[-trail:]
                if idx in by_frame:
                    g = by_frame[idx]
                    feet = geometry.pixel_to_court(h, g[["foot_x", "foot_y"]].to_numpy())
                    players_m = {int(t): tuple(p) for t, p in zip(g["track_id"], feet, strict=True)}
            frame = mini_court.draw_keypoints(frame, court.keypoints_px)
            if idx in ball.index and not np.isnan(ball.loc[idx, "x"]):
                bx, by = int(ball.loc[idx, "x"]), int(ball.loc[idx, "y"])
                frame = mini_court.draw_hud(frame, f"f{idx}  ball=({bx},{by})")
            else:
                frame = mini_court.draw_hud(frame, f"f{idx}")
            frame = mini_court.overlay(frame, mc.render(ball_m, players_m, recent))
            w.write(frame)
    return out_path


def stage_stats(clip: Path, run_dir: Path) -> dict:
    info = video.probe(clip)
    court = CourtResult.from_json(run_dir / "court.json")
    ball = pd.read_parquet(run_dir / "ball.parquet")
    players = pd.read_parquet(run_dir / "players.parquet")
    stats = build_stats(info, ball, players, court.homography is not None)
    write_stats(stats, run_dir / "stats.json")
    return stats


def analyze(
    clip: Path, outputs: Path, run: str | None = None, keypoints_json: Path | None = None
) -> Path:
    """Full pipeline with the default (currently stub) models."""
    run_dir = run_dir_for(clip, outputs, run)
    stage_court(clip, run_dir, keypoints_json=keypoints_json)
    stage_ball(clip, run_dir)
    stage_players(clip, run_dir)
    stage_annotate(clip, run_dir)
    stage_stats(clip, run_dir)
    return run_dir
