"""Draw the 2D court and overlay it on video frames."""

from __future__ import annotations

import cv2
import numpy as np

from courttally.court import geometry

WHITE = (255, 255, 255)
GREEN = (60, 140, 60)
YELLOW = (0, 220, 255)
RED = (0, 0, 230)


class MiniCourt:
    """Fixed-scale canvas of the court, near baseline at the bottom."""

    def __init__(self, px_per_m: float = 8.0, margin_m: float = 1.5) -> None:
        self.px_per_m = px_per_m
        self.margin_m = margin_m
        self.width = int((geometry.DOUBLES_WIDTH + 2 * margin_m) * px_per_m)
        self.height = int((geometry.COURT_LENGTH + 2 * margin_m) * px_per_m)
        self._base = self._draw_base()

    def to_px(self, pt_m: tuple[float, float] | np.ndarray) -> tuple[int, int]:
        x, y = float(pt_m[0]), float(pt_m[1])
        px = (x + geometry.HALF_DOUBLES + self.margin_m) * self.px_per_m
        py = (geometry.HALF_LENGTH + self.margin_m - y) * self.px_per_m
        return int(round(px)), int(round(py))

    def _draw_base(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), GREEN, dtype=np.uint8)
        for a, b in geometry.LINES:
            cv2.line(
                img,
                self.to_px(geometry.KEYPOINTS_M[a]),
                self.to_px(geometry.KEYPOINTS_M[b]),
                WHITE,
                1,
            )
        cv2.line(img, self.to_px(geometry.NET_LINE[0]), self.to_px(geometry.NET_LINE[1]), WHITE, 2)
        return img

    def render(
        self,
        ball_m: tuple[float, float] | None = None,
        players_m: dict[int, tuple[float, float]] | None = None,
        trail_m: list[tuple[float, float]] | None = None,
    ) -> np.ndarray:
        img = self._base.copy()
        for p in trail_m or []:
            cv2.circle(img, self.to_px(p), 1, YELLOW, -1)
        for tid, p in (players_m or {}).items():
            cv2.circle(img, self.to_px(p), 4, RED, -1)
            cv2.putText(img, str(tid), self.to_px(p), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)
        if ball_m is not None:
            cv2.circle(img, self.to_px(ball_m), 3, YELLOW, -1)
        return img


def overlay(
    frame: np.ndarray, panel: np.ndarray, margin: int = 12, alpha: float = 0.85
) -> np.ndarray:
    """Paste ``panel`` into the top-right corner of ``frame``."""
    out = frame.copy()
    ph, pw = panel.shape[:2]
    x0 = out.shape[1] - pw - margin
    y0 = margin
    roi = out[y0 : y0 + ph, x0 : x0 + pw]
    out[y0 : y0 + ph, x0 : x0 + pw] = cv2.addWeighted(panel, alpha, roi, 1 - alpha, 0)
    return out


def draw_keypoints(frame: np.ndarray, kps_px: np.ndarray) -> np.ndarray:
    out = frame.copy()
    for i, p in enumerate(kps_px):
        if np.isnan(p).any():
            continue
        cv2.circle(out, (int(p[0]), int(p[1])), 4, RED, -1)
        cv2.putText(
            out, str(i), (int(p[0]) + 4, int(p[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1
        )
    return out


def draw_hud(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    return out
