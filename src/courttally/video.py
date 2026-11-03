"""Frame reading and writing with OpenCV."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        return VideoInfo(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_frames(
    path: str | Path,
    max_width: int | None = None,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_idx, bgr_frame)``. ``frame_idx`` is the index in the source.

    ``max_width`` downscales (keeping aspect) so detectors run faster; callers must
    then scale pixel coordinates back if they want full-res output.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    try:
        while True:
            if stop is not None and idx >= stop:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start) % step == 0:
                if max_width and frame.shape[1] > max_width:
                    frame = resize_width(frame, max_width)
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


class VideoWriter:
    """Thin context-managed wrapper that sizes itself from the first frame."""

    def __init__(self, path: str | Path, fps: float, codec: str = "mp4v") -> None:
        self.path = Path(path)
        self.fps = fps
        self.codec = codec
        self._writer: cv2.VideoWriter | None = None
        self.n_written = 0

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"cannot open writer for {self.path}")
        self._writer.write(frame)
        self.n_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
