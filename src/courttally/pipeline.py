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

import json as _json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from courttally import activity as activity_mod
from courttally import editor, export, media, video
from courttally import points as points_mod
from courttally import score as score_mod
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


# ---- recording / editing stages -------------------------------------------
# These do not need any learned model. They are the SwingVision features users
# feel first: keep the original at full quality, find the points, cut the dead
# time, review and tag quickly, export to an editor.


def stage_ingest(clip: Path, run_dir: Path, proxy_width: int = 1280) -> Path:
    """Probe the original and build a browser/analysis-friendly proxy next to it.

    Returns the proxy path. The original is untouched; cuts always come from it.
    """
    info = media.probe_media(clip)
    (run_dir / "ingest.json").write_text(_json.dumps(info.to_dict(), indent=2))
    notes = []
    if info.variable_frame_rate:
        notes.append("variable frame rate: analysis uses the CFR proxy")
    if info.hdr:
        notes.append("HDR source: proxy is tone mapped to SDR")
    log.info(
        "ingest: %dx%d %.2ffps %s %s %.1fs audio=%s %s",
        info.width,
        info.height,
        info.fps,
        info.codec,
        info.pix_fmt,
        info.duration_s,
        info.has_audio,
        "; ".join(notes),
    )
    proxy = run_dir / "proxy.mp4"
    if not proxy.exists():
        media.make_proxy(clip, proxy, width=proxy_width)
    return proxy


def _analysis_video(clip: Path, run_dir: Path) -> Path:
    proxy = run_dir / "proxy.mp4"
    return proxy if proxy.exists() else clip


def stage_activity(clip: Path, run_dir: Path, audio_weight: float = 0.5) -> pd.DataFrame:
    src = _analysis_video(clip, run_dir)
    court_path = run_dir / "court.json"
    court = CourtResult.from_json(court_path) if court_path.exists() else None
    df = activity_mod.compute_activity(src, court, audio_weight=audio_weight)
    df.to_parquet(run_dir / "activity.parquet", index=False)
    log.info(
        "activity: %d frames, audio=%s, court mask=%s",
        len(df),
        df["audio"].notna().any(),
        court is not None and court.homography is not None,
    )
    return df


def stage_points(clip: Path, run_dir: Path, **thresholds) -> list[points_mod.Point]:
    src = _analysis_video(clip, run_dir)
    info = video.probe(src)
    df = pd.read_parquet(run_dir / "activity.parquet")
    pts = points_mod.activity_to_points(df, info.fps, **thresholds)
    points_mod.save_json(pts, run_dir / "points.json", clip.name, info.duration_s, info.fps)
    export.write_editor_files(pts, run_dir, info.fps, title=clip.stem)
    s = points_mod.summary(pts, info.duration_s)
    log.info(
        "points: %d found, play %.0fs, dead %.0fs (%.0f%%)",
        s["n_points"],
        s["play_s"],
        s["dead_s"],
        100 * s["dead_fraction"],
    )
    return pts


def _load_points(run_dir: Path, points_file: Path | None = None) -> list[points_mod.Point]:
    """Prefer the user's edited points over the automatic ones."""
    if points_file is None:
        edited = run_dir / "points.edited.json"
        points_file = edited if edited.exists() else run_dir / "points.json"
    return points_mod.load_json(points_file)


def stage_review(clip: Path, run_dir: Path, points_file: Path | None = None) -> Path:
    src = _analysis_video(clip, run_dir)
    info = video.probe(src)
    pts = _load_points(run_dir, points_file)
    rel = src.name if src.parent == run_dir else str(src.resolve())
    out = editor.write_review_page(pts, run_dir, rel, clip.name, info.duration_s, info.fps)
    log.info("review: open %s in a browser", out)
    return out


def stage_cut(
    clip: Path,
    run_dir: Path,
    mode: str = "copy",
    points_file: Path | None = None,
    per_point: bool = True,
) -> dict:
    pts = _load_points(run_dir, points_file)
    info = video.probe(clip)
    export.write_editor_files(pts, run_dir, info.fps, title=clip.stem)
    result = export.cut_points(clip, run_dir, pts, mode=mode, per_point=per_point)
    (run_dir / "cut.json").write_text(_json.dumps(result, indent=2))
    return result


def stage_score(
    run_dir: Path, points_file: Path | None = None, best_of: int = 3, no_ad: bool = False
) -> dict:
    pts = _load_points(run_dir, points_file)
    winners = [p.winner if p.keep else None for p in pts]
    progression = score_mod.score_points(winners, best_of=best_of, no_ad=no_ad)
    final = next((s for s in reversed(progression) if s), None)
    result = {
        "tagged_points": sum(1 for w in winners if w),
        "untagged_points": sum(1 for w in winners if not w),
        "points_won": {"A": winners.count("A"), "B": winners.count("B")},
        "final": final,
        "progression": [
            {"point": p.index, "start_s": p.start_s, "winner": p.winner, "score_after": s}
            for p, s in zip(pts, progression, strict=True)
        ],
    }
    (run_dir / "score.json").write_text(_json.dumps(result, indent=2))
    return result


def autoedit(
    clip: Path,
    outputs: Path,
    run: str | None = None,
    keypoints_json: Path | None = None,
    mode: str = "copy",
    cut_video: bool = True,
    **thresholds,
) -> Path:
    """Ingest -> (court) -> activity -> points -> review page -> cuts. No ML needed."""
    run_dir = run_dir_for(clip, outputs, run)
    stage_ingest(clip, run_dir)
    if keypoints_json is not None:
        stage_court(clip, run_dir, keypoints_json=keypoints_json)
    stage_activity(clip, run_dir)
    stage_points(clip, run_dir, **thresholds)
    stage_review(clip, run_dir)
    if cut_video:
        stage_cut(clip, run_dir, mode=mode)
    return run_dir
