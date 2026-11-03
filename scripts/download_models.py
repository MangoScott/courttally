"""Fetch model weights into ``models/`` (gitignored). Idempotent; skips existing files.

Usage: uv run scripts/download_models.py [--only yolo|court|ball]

Sources (Phase 0 list; revisit as models are swapped):
  yolov8x.pt   ultralytics release assets, AGPL-3.0 (players)
  court        yastrebksv/TennisCourtDetector keypoint weights (14 keypoints)
  ball         fine-tuned YOLO ball weights from the tennis_analysis reference
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"

SOURCES: dict[str, tuple[str, str | None]] = {
    "yolo": (
        "yolov8x.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
    ),
    # Not pinned yet. Phase 1 picks the court keypoint checkpoint (yastrebksv) and
    # Phase 2 the ball detector; both READMEs host weights on Google Drive, which
    # needs gdown. Fill the URL in when chosen and drop the None handling.
    "court": ("court_keypoints_resnet50.pt", None),
    "ball": ("ball_yolo.pt", None),
}


def fetch(name: str, url: str | None) -> Path:
    dest = MODELS / name
    if url is None:
        print(f"skip  {dest} (no source pinned yet; see PLAN.md)")
        return dest
    if dest.exists():
        print(f"skip  {dest} (exists)")
        return dest
    MODELS.mkdir(exist_ok=True)
    print(f"fetch {url}\n  -> {dest}")
    if "drive.google.com" in url:
        try:
            import gdown  # type: ignore
        except ImportError:
            sys.exit("Google Drive source: `uv pip install gdown` first")
        gdown.download(url, str(dest), quiet=False)
    else:
        urllib.request.urlretrieve(url, dest)  # noqa: S310
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(SOURCES), nargs="*", default=None)
    args = ap.parse_args()
    for key in args.only or SOURCES:
        fetch(*SOURCES[key])


if __name__ == "__main__":
    main()
