import numpy as np
import pandas as pd

from courttally import points as pm


def _activity(levels: list[tuple[float, float]], fps: float = 10.0) -> pd.DataFrame:
    """Build a signal from (seconds, level) runs."""
    a = np.concatenate([np.full(int(sec * fps), lvl) for sec, lvl in levels])
    return pd.DataFrame(
        {"frame_idx": np.arange(len(a)), "t_s": np.arange(len(a)) / fps, "activity": a}
    )


def test_segments_with_hysteresis_merge_and_padding():
    # 3 s dead, 4 s play, 1 s dip (merged), 3 s play, 5 s dead, 0.5 s blip (dropped), 4 s dead
    df = _activity([(3, 0.0), (4, 1.0), (1, 0.3), (3, 1.0), (5, 0.0), (0.5, 1.0), (4, 0.0)])
    pts = pm.segment(df, fps=10, pad_pre_s=1.0, pad_post_s=1.0)
    assert len(pts) == 1
    p = pts[0]
    assert p.start_s == 2.0 and p.end_s == 12.0  # 3-1 .. 11+1
    assert p.keep and p.winner is None


def test_two_points_do_not_overlap_after_padding():
    df = _activity([(2, 0.0), (3, 1.0), (2, 0.0), (3, 1.0), (2, 0.0)])
    pts = pm.segment(df, fps=10, merge_gap_s=0.5, pad_pre_s=1.5, pad_post_s=1.5)
    assert len(pts) == 2
    assert pts[0].end_s <= pts[1].start_s
    assert pm.summary(pts, 12.0)["n_points"] == 2


def test_point_still_open_at_end_is_closed():
    df = _activity([(2, 0.0), (3, 1.0)])
    pts = pm.segment(df, fps=10, pad_post_s=0.0)
    assert len(pts) == 1 and pts[0].end_s == 5.0


def test_serialization_round_trip_and_exports(tmp_path):
    pts = [
        pm.Point(0, 1.0, 4.5, winner="A"),
        pm.Point(1, 6.0, 9.0, keep=False),
        pm.Point(2, 10.0, 15.2, label="ace"),
    ]
    pm.save_json(pts, tmp_path / "p.json", "clip.mp4", 20.0, 30.0)
    assert pm.load_json(tmp_path / "p.json") == pts
    pm.save_csv(pts, tmp_path / "p.csv", 30.0)
    assert "00:00:01:00" in (tmp_path / "p.csv").read_text()
    pm.save_edl(pts, tmp_path / "p.edl", 30.0)
    edl = (tmp_path / "p.edl").read_text()
    assert edl.count("\n* FROM CLIP") == 2  # dropped point is excluded
    assert "00:00:10:00 00:00:15:06" in edl  # 15.2 s -> frame 6 at 30 fps
    assert abs(pm.dead_time_s(pts, 20.0) - (20.0 - 3.5 - 5.2)) < 1e-9
