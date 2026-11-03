"""Full pipeline on a real clip when one exists, otherwise on a synthetic one."""

import json
from pathlib import Path

import pytest

from courttally import pipeline
from courttally.cli import main

SMOKE = Path("data/clips/smoke.mp4")


@pytest.mark.slow
def test_analyze_end_to_end(tmp_path, synthetic_clip):
    clip = SMOKE if SMOKE.exists() else synthetic_clip
    kps = clip.with_suffix(".keypoints.json")
    run = pipeline.analyze(clip, tmp_path / "outputs", keypoints_json=kps if kps.exists() else None)
    for name in ("court.json", "ball.parquet", "players.parquet", "annotated.mp4", "stats.json"):
        assert (run / name).exists(), name
    stats = json.loads((run / "stats.json").read_text())
    assert stats["clip"]["n_frames"] > 0
    if kps.exists():
        assert stats["court"]["homography_found"] is True


@pytest.mark.slow
def test_cli_stages_run_alone(tmp_path, synthetic_clip):
    out = str(tmp_path / "outputs")
    for stage in ("court", "ball", "players", "annotate", "stats"):
        assert main([stage, str(synthetic_clip), "--outputs", out]) == 0
    assert main(["info", str(synthetic_clip)]) == 0
