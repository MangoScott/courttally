"""Ball detection. Phase 0: a stub. Phase 2: fine-tuned YOLO, then TrackNetV3."""

from __future__ import annotations

import numpy as np

from courttally.types import Box


class NoBall:
    """Detects nothing, so downstream code is exercised with ``None`` positions."""

    def detect(self, frame: np.ndarray) -> list[Box]:
        return []
