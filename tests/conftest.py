from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> Path:
    """A 2 s synthetic clip + keypoints JSON, generated on the fly."""
    import make_synthetic_clip  # noqa: PLC0415

    out = tmp_path / "synthetic.mp4"
    argv = sys.argv
    sys.argv = ["make_synthetic_clip", str(out), "--seconds", "2"]
    try:
        make_synthetic_clip.main()
    finally:
        sys.argv = argv
    return out
