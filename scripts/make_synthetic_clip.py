"""Render a short synthetic behind-the-baseline clip so the pipeline can be smoke
tested before real footage exists. Draws a perspective court, a bouncing ball and
two rectangle "players", and writes the matching 14 keypoints to JSON.

Usage: uv run scripts/make_synthetic_clip.py data/clips/synthetic.mp4 [--seconds 5]
Real clips always come first (see CLAUDE.md); this is only for plumbing checks.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import cv2
import numpy as np

from courttally import media
from courttally.court import geometry
from courttally.video import VideoWriter

W, H = 1280, 720


def camera_homography() -> np.ndarray:
    """Court meters -> pixels for a phone behind the near baseline, slightly elevated."""
    court = np.array(
        [
            (-geometry.HALF_DOUBLES, -geometry.HALF_LENGTH),
            (geometry.HALF_DOUBLES, -geometry.HALF_LENGTH),
            (geometry.HALF_DOUBLES, geometry.HALF_LENGTH),
            (-geometry.HALF_DOUBLES, geometry.HALF_LENGTH),
        ],
        dtype=np.float32,
    )
    pixels = np.array([(120, 690), (1160, 690), (820, 250), (460, 250)], dtype=np.float32)
    return cv2.getPerspectiveTransform(court, pixels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--rally", type=float, default=3.0, help="seconds of play per cycle")
    ap.add_argument("--pause", type=float, default=6.0, help="seconds of dead time per cycle")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    m2px = camera_homography()
    kps_px = cv2.perspectiveTransform(
        geometry.KEYPOINTS_M.reshape(-1, 1, 2).astype(np.float32), m2px
    ).reshape(-1, 2)
    kp_json = {
        n: [float(x), float(y)] for n, (x, y) in zip(geometry.KEYPOINT_NAMES, kps_px, strict=True)
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".keypoints.json").write_text(json.dumps(kp_json, indent=2))

    n = int(args.seconds * args.fps)
    cycle = args.rally + args.pause
    hit_times: list[float] = []
    silent = args.out.with_name(args.out.stem + ".silent.mp4")
    with VideoWriter(silent, args.fps) as w:
        for i in range(n):
            t = i / args.fps
            playing = (t % cycle) < args.rally
            img = np.full((H, W, 3), (70, 120, 45), dtype=np.uint8)
            for a, b in geometry.LINES:
                cv2.line(
                    img,
                    tuple(kps_px[a].astype(int)),
                    tuple(kps_px[b].astype(int)),
                    (255, 255, 255),
                    2,
                )
            net = cv2.perspectiveTransform(
                np.array(geometry.NET_LINE, dtype=np.float32).reshape(-1, 1, 2), m2px
            ).reshape(-1, 2)
            cv2.line(img, tuple(net[0].astype(int)), tuple(net[1].astype(int)), (230, 230, 230), 3)
            # Ball: rally along the court, parabolic hop every 1.2 s; absent during pauses.
            if playing:
                y_m = -10 + 20 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 2.4))
                x_m = 2.5 * np.sin(2 * np.pi * t / 2.4 + 1)
                z = abs(np.sin(np.pi * t / 1.2)) * 1.2
                px = cv2.perspectiveTransform(np.array([[[x_m, y_m]]], dtype=np.float32), m2px)
                px = px[0, 0]
                cv2.circle(img, (int(px[0]), int(px[1] - z * 25)), 6, (0, 220, 255), -1)
                if (t % cycle) % 1.2 < 1 / args.fps:
                    hit_times.append(t)
            # Players as rectangles standing on their foot points; still during pauses.
            move = np.sin(t) if playing else 0.0
            for x_p, y_p, col in (
                (1.0 + move, -10.5, (200, 60, 60)),
                (-1.0 - move, 10.5, (60, 60, 200)),
            ):
                fp = cv2.perspectiveTransform(np.array([[[x_p, y_p]]], dtype=np.float32), m2px)[
                    0, 0
                ]
                hgt = 160 if y_p < 0 else 70
                cv2.rectangle(
                    img,
                    (int(fp[0]) - hgt // 5, int(fp[1]) - hgt),
                    (int(fp[0]) + hgt // 5, int(fp[1])),
                    col,
                    -1,
                )
            w.write(img)
    if args.no_audio:
        silent.replace(args.out)
    else:
        wav = write_click_track(args.out.with_suffix(".wav"), hit_times, args.seconds)
        try:
            media.mux_audio(silent, wav, args.out)
            silent.unlink()
        except Exception as e:  # noqa: BLE001 - ffmpeg missing: fall back to silent clip
            print(f"audio mux failed ({e}); writing silent clip")
            silent.replace(args.out)
        finally:
            wav.unlink(missing_ok=True)
    kp_path = args.out.with_suffix(".keypoints.json")
    print(f"wrote {args.out} ({n} frames, {len(hit_times)} hits) and {kp_path}")


def write_click_track(path: Path, hit_times: list[float], seconds: float, sr: int = 16000) -> Path:
    """Short decaying bursts at each hit over faint noise, like a ball strike on a quiet court."""
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.005, int(seconds * sr))
    burst = np.exp(-np.arange(int(0.03 * sr)) / (0.006 * sr)) * rng.normal(0, 0.6, int(0.03 * sr))
    for t in hit_times:
        i = int(t * sr)
        audio[i : i + len(burst)] += burst[: len(audio) - i]
    pcm = np.clip(audio, -1, 1)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return path


if __name__ == "__main__":
    main()
