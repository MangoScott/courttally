"""Shared data types and the small model interfaces every stage talks through.

Frame index is the universal key: every detection carries ``frame_idx``.
Missing detections are ``None``, never ``(0, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned pixel box with a class label and confidence."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    label: str = ""

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-center of the box: where a person touches the ground."""
        return ((self.x1 + self.x2) / 2, self.y2)


@dataclass(frozen=True)
class BallDetection:
    frame_idx: int
    xy: tuple[float, float] | None
    conf: float
    interpolated: bool = False


@dataclass(frozen=True)
class PlayerDetection:
    frame_idx: int
    track_id: int
    box: Box


class Detector(Protocol):
    """Anything that finds boxes in a frame. Swap models freely behind this."""

    def detect(self, frame: np.ndarray) -> list[Box]: ...


class KeypointModel(Protocol):
    """Anything that predicts the 14 court keypoints as an (14, 2) pixel array.

    Rows follow ``courttally.court.geometry.KEYPOINT_NAMES``. Unseen points are NaN.
    """

    def keypoints(self, frame: np.ndarray) -> np.ndarray: ...
