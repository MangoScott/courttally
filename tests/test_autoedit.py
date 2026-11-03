"""Dead-time removal end to end on the synthetic clip (3 s rally / 6 s pause cycles)."""

import json
import sys

import pytest

from courttally import pipeline
from courttally.cli import main


@pytest.fixture
def long_synthetic(tmp_path):
    import make_synthetic_clip  # noqa: PLC0415

    out = tmp_path / "synthetic.mp4"
    argv = sys.argv
    sys.argv = ["make_synthetic_clip", str(out), "--seconds", "12"]
    try:
        make_synthetic_clip.main()
    finally:
        sys.argv = argv
    return out


@pytest.mark.slow
def test_autoedit_finds_points_and_cuts(tmp_path, long_synthetic):
    run = pipeline.autoedit(
        long_synthetic,
        tmp_path / "out",
        keypoints_json=long_synthetic.with_suffix(".keypoints.json"),
    )
    data = json.loads((run / "points.json").read_text())
    pts = data["points"]
    assert len(pts) == 2, pts  # rallies at 0-3 s and 9-12 s
    starts = [p["start_s"] for p in pts]
    assert abs(starts[1] - 8.0) < 1.0, starts  # second rally at 9 s minus 1 s pre-pad
    assert pts[0]["end_s"] < 6.0, pts  # the 6 s pause is cut, not bridged
    from courttally import points as pm  # noqa: PLC0415

    assert pm.summary(pm.load_json(run / "points.json"), 12.0)["dead_fraction"] > 0.2
    for name in (
        "ingest.json",
        "proxy.mp4",
        "activity.parquet",
        "points.csv",
        "points.edl",
        "review.html",
        "condensed.mp4",
        "cut.json",
    ):
        assert (run / name).exists(), name
    assert (run / "points" / "point_000.mp4").exists()
    assert "A won" in (run / "review.html").read_text()


@pytest.mark.slow
def test_edit_stages_run_alone_and_score(tmp_path, long_synthetic):
    out = str(tmp_path / "out")
    for stage in ("ingest", "activity", "points", "review"):
        assert main([stage, str(long_synthetic), "--outputs", out]) == 0
    run = tmp_path / "out" / "synthetic"
    data = json.loads((run / "points.json").read_text())
    for i, p in enumerate(data["points"]):
        p["winner"] = "AB"[i % 2]
    (run / "points.edited.json").write_text(json.dumps(data))
    assert main(["score", str(long_synthetic), "--outputs", out]) == 0
    score = json.loads((run / "score.json").read_text())
    assert score["tagged_points"] == len(data["points"]) and score["final"]["games"] == "0-0"
    assert main(["cut", str(long_synthetic), "--outputs", out, "--condensed-only"]) == 0
    assert (run / "condensed.mp4").exists() and not (run / "points").exists()
