"""Per-frame "is anything happening" signal from motion and sound.

Dead time between points is the single biggest thing a user wants cut. Two cheap
cues cover it without any learned model: pixel motion inside the court region and
audio energy (ball strikes are loud, sharp transients; dead time is quiet).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from courttally import media, video
from courttally.court import geometry
from courttally.court.detector import CourtResult


def court_mask(
    court: CourtResult | None, frame_shape: tuple[int, int], margin_m: float = 2.0
) -> np.ndarray | None:
    """Binary mask of the court plus a margin, in the frame's pixel space."""
    if court is None or court.homography is None:
        return None
    poly = geometry.court_polygon(singles=False)
    poly = poly * (1 + margin_m / np.array([geometry.HALF_DOUBLES, geometry.HALF_LENGTH]))
    px = geometry.court_to_pixel(court.homography, poly)
    sx = frame_shape[1] / court.frame_size[0]
    sy = frame_shape[0] / court.frame_size[1]
    px = (px * np.array([sx, sy])).astype(np.int32)
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [px], 255)
    return mask


def motion_energy(clip: Path, court: CourtResult | None = None, width: int = 320) -> pd.DataFrame:
    """Mean absolute frame difference (0-255) per frame, masked to the court if known."""
    prev = None
    mask = None
    rows = []
    for idx, frame in video.read_frames(clip, max_width=width):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if mask is None and court is not None:
            mask = court_mask(court, gray.shape)
        if prev is None:
            rows.append((idx, 0.0))
        else:
            diff = cv2.absdiff(gray, prev)
            rows.append((idx, float(cv2.mean(diff, mask=mask)[0])))
        prev = gray
    return pd.DataFrame(rows, columns=["frame_idx", "motion"])


def audio_energy(clip: Path, fps: float, n_frames: int) -> np.ndarray | None:
    """RMS audio level per video frame, emphasising transients (ball strikes)."""
    got = media.extract_audio(clip)
    if got is None:
        return None
    samples, sr = got
    # Crude high-pass: ball hits and shoe squeaks live above wind/traffic rumble.
    hp = samples - np.convolve(samples, np.ones(64) / 64, mode="same")
    per_frame = int(sr / fps)
    n = min(n_frames, len(hp) // per_frame)
    rms = np.sqrt(np.mean(hp[: n * per_frame].reshape(n, per_frame) ** 2, axis=1))
    out = np.zeros(n_frames)
    out[:n] = rms
    return out


def _normalize(x: np.ndarray) -> np.ndarray:
    """Map to roughly [0, 1] using robust percentiles so thresholds transfer across clips."""
    lo, hi = np.percentile(x, 20), np.percentile(x, 90)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1.5)


def smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


def compute_activity(
    clip: Path, court: CourtResult | None = None, smooth_s: float = 0.5, audio_weight: float = 0.5
) -> pd.DataFrame:
    """Table: frame_idx, t_s, motion, audio, activity (0..~1). Saved as ``activity.parquet``."""
    info = video.probe(clip)
    df = motion_energy(clip, court)
    win = max(1, int(round(smooth_s * info.fps)))
    df["t_s"] = df["frame_idx"] / info.fps
    motion = _normalize(smooth(df["motion"].to_numpy(), win))
    audio = audio_energy(clip, info.fps, len(df))
    if audio is None:
        df["audio"] = np.nan
        df["activity"] = motion
    else:
        df["audio"] = audio
        a = _normalize(smooth(audio, win))
        df["activity"] = (1 - audio_weight) * motion + audio_weight * a
    return df[["frame_idx", "t_s", "motion", "audio", "activity"]]
