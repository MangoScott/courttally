from courttally.ball import tracker
from courttally.types import BallDetection, Box


def _d(i, xy, conf=0.9):
    return BallDetection(i, xy, conf)


def test_pick_ball_prefers_confidence():
    boxes = [Box(0, 0, 2, 2, 0.3), Box(10, 10, 12, 12, 0.8)]
    assert tracker.pick_ball(boxes).center == (11, 11)
    assert tracker.pick_ball([]) is None


def test_interpolates_short_gaps_only():
    dets = (
        [_d(0, (0, 0)), _d(1, None), _d(2, (2, 2))]
        + [_d(i, None) for i in range(3, 10)]
        + [_d(10, (10, 10))]
    )
    out = tracker.interpolate(dets, max_gap=5)
    assert out[1].xy == (1.0, 1.0) and out[1].interpolated
    assert all(o.xy is None for o in out[3:10])
    assert out[10].xy == (10, 10) and not out[10].interpolated
    df = tracker.to_frame(out)
    assert list(df.columns) == ["frame_idx", "x", "y", "conf", "interpolated"]
