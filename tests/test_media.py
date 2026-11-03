import shutil

import numpy as np
import pytest

from courttally import media, video

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not installed")


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "c.mp4"
    with video.VideoWriter(path, fps=30) as w:
        for i in range(90):
            w.write(np.full((144, 256, 3), (i * 2) % 255, np.uint8))
    return path


@needs_ffmpeg
def test_probe_media(clip):
    info = media.probe_media(clip)
    assert (info.width, info.height) == (256, 144)
    assert abs(info.fps - 30) < 0.01 and abs(info.duration_s - 3.0) < 0.1
    assert info.has_audio is False and info.hdr is False


def test_cut_and_concat_and_proxy(clip, tmp_path):
    a = media.cut(clip, tmp_path / "a.mp4", 0.0, 1.0, mode="accurate")
    b = media.cut(clip, tmp_path / "b.mp4", 2.0, 3.0, mode="accurate")
    out = media.concat([a, b], tmp_path / "joined.mp4")
    assert abs(video.probe(out).n_frames - 60) <= 2
    proxy = media.make_proxy(clip, tmp_path / "proxy.mp4", width=128)
    assert video.probe(proxy).width == 128
    assert media.extract_audio(clip) is None
