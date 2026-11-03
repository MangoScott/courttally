import numpy as np
import pytest

from courttally.court import geometry as g


def _synthetic_h() -> np.ndarray:
    """A plausible court->pixel perspective for a camera behind the baseline."""
    import cv2

    court = np.array(
        [(-5.485, -11.885), (5.485, -11.885), (5.485, 11.885), (-5.485, 11.885)], np.float32
    )
    px = np.array([(100, 1000), (1800, 1000), (1300, 300), (600, 300)], np.float32)
    return cv2.getPerspectiveTransform(court, px)


def test_keypoint_table_is_consistent():
    assert len(g.KEYPOINT_NAMES) == 14
    assert g.KEYPOINTS_M.shape == (14, 2)
    assert np.isclose(g.KEYPOINTS_M[:, 1].max(), 11.885)
    assert np.isclose(g.KEYPOINTS_M[:, 0].max(), 5.485)
    for a, b in g.LINES:
        assert 0 <= a < 14 and 0 <= b < 14


def _project(m2px: np.ndarray) -> np.ndarray:
    import cv2

    pts = g.KEYPOINTS_M.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, m2px).reshape(-1, 2).astype(np.float64)


def test_homography_round_trip_within_1cm():
    px = _project(_synthetic_h())
    h = g.fit_homography(px)
    back = g.pixel_to_court(h, px)
    assert np.abs(back - g.KEYPOINTS_M).max() < 0.01
    again = g.court_to_pixel(h, back)
    assert np.abs(again - px).max() < 0.1


def test_homography_with_subpixel_noise_stays_within_5cm():
    """At the far baseline one pixel spans >1 cm, so noisy input cannot hit 1 cm."""
    px = _project(_synthetic_h())
    px += np.random.default_rng(0).normal(0, 0.3, px.shape)
    h = g.fit_homography(px)
    back = g.pixel_to_court(h, px)
    assert np.abs(back - g.KEYPOINTS_M).max() < 0.05


def test_fit_homography_tolerates_missing_points():
    m2px = _synthetic_h()
    import cv2

    px = cv2.perspectiveTransform(g.KEYPOINTS_M.reshape(-1, 1, 2).astype(np.float32), m2px).reshape(
        -1, 2
    )
    px[[2, 3, 12, 13]] = np.nan  # near baseline + T points hidden
    h = g.fit_homography(px)
    assert np.abs(g.pixel_to_court(h, px[0]) - g.KEYPOINTS_M[0]).max() < 0.01


def test_fit_homography_needs_four_points():
    px = np.full((14, 2), np.nan)
    px[:3] = [(0, 0), (1, 0), (0, 1)]
    with pytest.raises(ValueError):
        g.fit_homography(px)


@pytest.mark.parametrize(
    ("pt", "singles", "expected"),
    [
        ((0, 0), True, True),
        ((4.115, 11.885), True, True),  # on the corner: in
        ((4.2, 0), True, False),  # outside singles, inside doubles
        ((4.2, 0), False, True),
        ((0, 11.9), True, False),
    ],
)
def test_is_inside(pt, singles, expected):
    assert g.is_inside(pt, singles=singles) is expected


def test_zones():
    assert g.depth_zone(1.0) == "short"
    assert g.depth_zone(-8.0) == "mid"
    assert g.depth_zone(11.0) == "deep"
    assert g.placement_zone(-3.5) == "left"
    assert g.placement_zone(0.0) == "center"
    assert g.placement_zone(3.5) == "right"
