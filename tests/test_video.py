import numpy as np

from courttally import video


def test_write_then_read_round_trip(tmp_path):
    path = tmp_path / "t.mp4"
    with video.VideoWriter(path, fps=30) as w:
        for i in range(12):
            w.write(np.full((120, 160, 3), i * 10, np.uint8))
    info = video.probe(path)
    assert (info.width, info.height, info.n_frames) == (160, 120, 12)
    frames = list(video.read_frames(path, max_width=80, step=4))
    assert [i for i, _ in frames] == [0, 4, 8]
    assert frames[0][1].shape == (60, 80, 3)
