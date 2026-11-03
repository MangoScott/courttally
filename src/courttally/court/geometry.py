"""Court geometry in meters. The single source of truth for all court math.

Coordinate frame (court space):
  * ``x_m`` runs across the court, 0 at the center service line.
  * ``y_m`` runs along the court, 0 at the net. The near baseline (the one the
    camera sits behind) is at negative y and draws at the bottom of the mini-court.

ITF dimensions: 23.77 m long, 10.97 m wide for doubles, 8.23 m wide for singles,
service lines 6.40 m from the net.
"""

from __future__ import annotations

import cv2
import numpy as np

COURT_LENGTH = 23.77
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
SERVICE_LINE_FROM_NET = 6.40
NET_HEIGHT_CENTER = 0.914

HALF_LENGTH = COURT_LENGTH / 2
HALF_DOUBLES = DOUBLES_WIDTH / 2
HALF_SINGLES = SINGLES_WIDTH / 2

# The 14 keypoints used by the court detector, in a fixed order. ``near`` is the
# camera side (negative y). This order mirrors yastrebksv/TennisCourtDetector so
# its weights can be used without remapping; verify against its README when the
# model lands in Phase 1.
KEYPOINT_NAMES: tuple[str, ...] = (
    "far_doubles_left",
    "far_doubles_right",
    "near_doubles_left",
    "near_doubles_right",
    "far_singles_left",
    "far_singles_right",
    "near_singles_left",
    "near_singles_right",
    "far_service_left",
    "far_service_right",
    "near_service_left",
    "near_service_right",
    "far_service_T",
    "near_service_T",
)

KEYPOINTS_M: np.ndarray = np.array(
    [
        (-HALF_DOUBLES, HALF_LENGTH),
        (HALF_DOUBLES, HALF_LENGTH),
        (-HALF_DOUBLES, -HALF_LENGTH),
        (HALF_DOUBLES, -HALF_LENGTH),
        (-HALF_SINGLES, HALF_LENGTH),
        (HALF_SINGLES, HALF_LENGTH),
        (-HALF_SINGLES, -HALF_LENGTH),
        (HALF_SINGLES, -HALF_LENGTH),
        (-HALF_SINGLES, SERVICE_LINE_FROM_NET),
        (HALF_SINGLES, SERVICE_LINE_FROM_NET),
        (-HALF_SINGLES, -SERVICE_LINE_FROM_NET),
        (HALF_SINGLES, -SERVICE_LINE_FROM_NET),
        (0.0, SERVICE_LINE_FROM_NET),
        (0.0, -SERVICE_LINE_FROM_NET),
    ],
    dtype=np.float64,
)

# Line segments as pairs of keypoint indices; used by the mini-court drawer.
LINES: tuple[tuple[int, int], ...] = (
    (0, 1),  # far baseline
    (2, 3),  # near baseline
    (0, 2),  # left doubles sideline
    (1, 3),  # right doubles sideline
    (4, 6),  # left singles sideline
    (5, 7),  # right singles sideline
    (8, 9),  # far service line
    (10, 11),  # near service line
    (12, 13),  # center service line
)
NET_LINE: tuple[tuple[float, float], tuple[float, float]] = (
    (-HALF_DOUBLES, 0.0),
    (HALF_DOUBLES, 0.0),
)


def court_polygon(singles: bool = True) -> np.ndarray:
    """Playable rectangle as (4, 2) court-space corners, counter-clockwise."""
    w = HALF_SINGLES if singles else HALF_DOUBLES
    return np.array(
        [(-w, -HALF_LENGTH), (w, -HALF_LENGTH), (w, HALF_LENGTH), (-w, HALF_LENGTH)],
        dtype=np.float64,
    )


def fit_homography(
    pixel_pts: np.ndarray,
    court_pts: np.ndarray | None = None,
    ransac_thresh_px: float = 5.0,
) -> np.ndarray:
    """Pixel -> court homography from matched points.

    ``pixel_pts`` is (N, 2); rows with NaN are dropped. ``court_pts`` defaults to
    ``KEYPOINTS_M`` so a full 14-point prediction can be passed straight in.
    Raises ``ValueError`` if fewer than 4 valid correspondences remain.
    """
    pixel_pts = np.asarray(pixel_pts, dtype=np.float64)
    court_pts = KEYPOINTS_M if court_pts is None else np.asarray(court_pts, dtype=np.float64)
    if pixel_pts.shape != court_pts.shape:
        raise ValueError(f"shape mismatch: {pixel_pts.shape} vs {court_pts.shape}")
    valid = ~np.isnan(pixel_pts).any(axis=1)
    if valid.sum() < 4:
        raise ValueError(f"need >= 4 valid keypoints, got {int(valid.sum())}")
    src = pixel_pts[valid].astype(np.float32)
    dst = court_pts[valid].astype(np.float32)
    method = cv2.RANSAC if valid.sum() > 4 else 0
    h, _ = cv2.findHomography(src, dst, method, ransac_thresh_px)
    if h is None:
        raise ValueError("findHomography failed")
    return h.astype(np.float64)


def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    single = pts.ndim == 1
    pts2 = pts.reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts2, h).reshape(-1, 2)
    return out[0] if single else out


def pixel_to_court(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map pixel (x, y) to court (x_m, y_m). Accepts one point or (N, 2)."""
    return _apply(h, pts)


def court_to_pixel(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map court (x_m, y_m) back to pixels using the inverse homography."""
    return _apply(np.linalg.inv(h), pts)


def is_inside(pt_m: tuple[float, float], singles: bool = True, tolerance_m: float = 0.0) -> bool:
    """Whether a court-space point is in bounds. Lines count as in (ITF rule).

    ``tolerance_m`` expands the court; use it for the ball's radius / detector error.
    """
    x, y = pt_m
    w = (HALF_SINGLES if singles else HALF_DOUBLES) + tolerance_m
    return abs(x) <= w and abs(y) <= HALF_LENGTH + tolerance_m


def depth_zone(y_m: float) -> str:
    """short / mid / deep by distance from the net on either side."""
    d = abs(y_m)
    if d < SERVICE_LINE_FROM_NET:
        return "short"
    if d < SERVICE_LINE_FROM_NET + (HALF_LENGTH - SERVICE_LINE_FROM_NET) / 2:
        return "mid"
    return "deep"


def placement_zone(x_m: float, singles: bool = True) -> str:
    """left / center / right thirds across the playable width."""
    w = HALF_SINGLES if singles else HALF_DOUBLES
    third = 2 * w / 3
    if x_m < -w + third:
        return "left"
    if x_m > w - third:
        return "right"
    return "center"
