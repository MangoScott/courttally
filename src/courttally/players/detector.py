"""Player detection and tracking. Phase 0: a stub. Phase 3: YOLOv8 + ByteTrack."""

from __future__ import annotations

import numpy as np
import pandas as pd

from courttally.types import Box, PlayerDetection


class NoPlayers:
    def detect(self, frame: np.ndarray) -> list[Box]:
        return []


def to_frame(dets: list[PlayerDetection]) -> pd.DataFrame:
    """Flat table saved as ``players.parquet``; foot point is the court-projection input."""
    return pd.DataFrame(
        {
            "frame_idx": [d.frame_idx for d in dets],
            "track_id": [d.track_id for d in dets],
            "x1": [d.box.x1 for d in dets],
            "y1": [d.box.y1 for d in dets],
            "x2": [d.box.x2 for d in dets],
            "y2": [d.box.y2 for d in dets],
            "conf": [d.box.conf for d in dets],
            "foot_x": [d.box.foot[0] for d in dets],
            "foot_y": [d.box.foot[1] for d in dets],
        }
    )
