"""Court keypoint detection and per-clip homography estimation.

Phase 0 ships a ``ManualKeypoints`` model (keypoints from a JSON file, e.g. hand
clicked) so the rest of the pipeline can be exercised before the learned model
lands. Phase 1 adds a ResNet50 keypoint head trained on the yastrebksv dataset
behind the same ``KeypointModel`` interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from courttally.court import geometry
from courttally.types import KeypointModel


class ManualKeypoints:
    """Return the same 14 fixed pixel keypoints for every frame (static camera)."""

    def __init__(self, pixel_pts: np.ndarray) -> None:
        pixel_pts = np.asarray(pixel_pts, dtype=np.float64)
        if pixel_pts.shape != (14, 2):
            raise ValueError(f"expected (14, 2) keypoints, got {pixel_pts.shape}")
        self.pixel_pts = pixel_pts

    @classmethod
    def from_json(cls, path: str | Path) -> ManualKeypoints:
        data = json.loads(Path(path).read_text())
        pts = [data.get(name, [np.nan, np.nan]) for name in geometry.KEYPOINT_NAMES]
        return cls(np.array(pts, dtype=np.float64))

    def keypoints(self, frame: np.ndarray) -> np.ndarray:
        return self.pixel_pts.copy()


class NoCourt:
    """Stub that detects nothing. Lets the pipeline run end to end without a model."""

    def keypoints(self, frame: np.ndarray) -> np.ndarray:
        return np.full((14, 2), np.nan)


@dataclass
class CourtResult:
    """Cached output of the court stage (``court.json``)."""

    keypoints_px: np.ndarray  # (14, 2), NaN where unseen
    homography: np.ndarray | None  # pixel -> court meters, None if not enough points
    frame_size: tuple[int, int]  # (width, height) the keypoints were measured in

    def to_json(self, path: str | Path) -> None:
        payload = {
            "frame_size": list(self.frame_size),
            "keypoints_px": {
                name: (None if np.isnan(p).any() else [float(p[0]), float(p[1])])
                for name, p in zip(geometry.KEYPOINT_NAMES, self.keypoints_px, strict=True)
            },
            "homography": None if self.homography is None else self.homography.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> CourtResult:
        data = json.loads(Path(path).read_text())
        pts = [data["keypoints_px"].get(n) or [np.nan, np.nan] for n in geometry.KEYPOINT_NAMES]
        h = data.get("homography")
        return cls(
            keypoints_px=np.array(pts, dtype=np.float64),
            homography=None if h is None else np.array(h, dtype=np.float64),
            frame_size=tuple(data["frame_size"]),
        )


def detect_court(
    model: KeypointModel, frames: list[np.ndarray], frame_size: tuple[int, int]
) -> CourtResult:
    """Run the keypoint model on a few frames, take the median, fit a homography.

    The camera is static on a tripod so the median over sampled frames is a cheap,
    robust estimate. Returns ``homography=None`` when fewer than 4 points are seen.
    """
    preds = (
        np.stack([model.keypoints(f) for f in frames]) if frames else np.full((1, 14, 2), np.nan)
    )
    kps = np.nanmedian(preds, axis=0) if np.isfinite(preds).any() else np.full((14, 2), np.nan)
    try:
        h = geometry.fit_homography(kps)
    except ValueError:
        h = None
    return CourtResult(keypoints_px=kps, homography=h, frame_size=frame_size)
