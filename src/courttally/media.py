"""ffmpeg/ffprobe wrappers: probing, proxies, audio extraction, lossless cutting.

The original recording is never modified or re-encoded unless the caller asks.
Analysis runs on a small constant-frame-rate proxy; cuts are made from the
original with stream copy so exported clips keep full capture quality.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # bundled static binary, works without a system install

    return imageio_ffmpeg.get_ffmpeg_exe()


def ffprobe_exe() -> str | None:
    return shutil.which("ffprobe")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


@dataclass(frozen=True)
class MediaInfo:
    path: str
    width: int
    height: int
    fps: float
    duration_s: float
    n_frames: int
    codec: str
    pix_fmt: str
    has_audio: bool
    variable_frame_rate: bool
    hdr: bool
    rotation: int

    def to_dict(self) -> dict:
        return asdict(self)


def _fps(s: str | None) -> float:
    if not s or s == "0/0":
        return 0.0
    return float(Fraction(s))


def probe_media(path: str | Path) -> MediaInfo:
    """Container-level facts. Uses ffprobe when present, OpenCV otherwise."""
    path = Path(path)
    probe = ffprobe_exe()
    if probe is None:
        from courttally import video

        info = video.probe(path)
        return MediaInfo(
            str(path),
            info.width,
            info.height,
            info.fps,
            info.duration_s,
            info.n_frames,
            "unknown",
            "unknown",
            False,
            False,
            False,
            0,
        )
    out = _run(
        [probe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)]
    ).stdout
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    avg = _fps(v.get("avg_frame_rate"))
    real = _fps(v.get("r_frame_rate"))
    duration = float(v.get("duration") or data["format"].get("duration") or 0.0)
    n_frames = int(v.get("nb_frames") or round(duration * avg))
    rotation = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    rotation = int(v.get("tags", {}).get("rotate", rotation))
    color_transfer = v.get("color_transfer", "")
    return MediaInfo(
        path=str(path),
        width=int(v["width"]),
        height=int(v["height"]),
        fps=avg or real,
        duration_s=duration,
        n_frames=n_frames,
        codec=v.get("codec_name", "unknown"),
        pix_fmt=v.get("pix_fmt", "unknown"),
        has_audio=has_audio,
        # Phones often record VFR; a gap between avg and nominal rate is the tell.
        variable_frame_rate=bool(avg and real and abs(avg - real) > 0.05),
        hdr=color_transfer in {"smpte2084", "arib-std-b67"},
        rotation=rotation,
    )


def make_proxy(
    src: str | Path, dst: str | Path, width: int = 1280, fps: float | None = None
) -> Path:
    """Small constant-frame-rate H.264 proxy for analysis and the review page.

    HDR sources are tone mapped to SDR so browsers and OpenCV show them correctly.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    info = probe_media(src)
    vf = [f"scale={width}:-2"]
    if info.hdr:
        vf = [
            "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable,"
            "zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
        ] + vf
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-i", str(src), "-vf", ",".join(vf)]
    if fps:
        cmd += ["-r", f"{fps}"]
    cmd += [
        "-fps_mode",  # ffmpeg >= 5.1; "-vsync" was removed in ffmpeg 9
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    cmd += ["-c:a", "aac", "-b:a", "96k"] if info.has_audio else ["-an"]
    cmd.append(str(dst))
    _run(cmd)
    return dst


def extract_audio(src: str | Path, sample_rate: int = 16000) -> tuple[np.ndarray, int] | None:
    """Mono float32 samples in [-1, 1], or None when the clip has no audio track."""
    if not probe_media(src).has_audio:
        return None
    cmd = [
        ffmpeg_exe(),
        "-v",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32), sample_rate


def cut(src: str | Path, dst: str | Path, start_s: float, end_s: float, mode: str = "copy") -> Path:
    """Cut ``[start_s, end_s)`` out of ``src``.

    ``copy`` is instant and lossless but snaps the start to the previous keyframe,
    so clips may begin up to a GOP (~1-2 s on phones) early. ``accurate`` re-encodes
    at high quality (CRF 17) for frame-exact edges.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-v", "error"]
    if mode == "copy":
        cmd += [
            "-ss",
            f"{start_s:.3f}",
            "-to",
            f"{end_s:.3f}",
            "-i",
            str(src),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
        ]
    elif mode == "accurate":
        cmd += [
            "-ss",
            f"{start_s:.3f}",
            "-to",
            f"{end_s:.3f}",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
        ]
    else:
        raise ValueError(f"unknown cut mode {mode!r}")
    cmd.append(str(dst))
    _run(cmd)
    return dst


def concat(parts: list[Path], dst: str | Path) -> Path:
    """Join clips (same codec/params) without re-encoding."""
    dst = Path(dst)
    listing = dst.with_suffix(".concat.txt")
    listing.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts))
    _run(
        [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    )
    listing.unlink()
    return dst


def mux_audio(video_src: Path, wav: Path, dst: Path) -> Path:
    _run(
        [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-i",
            str(video_src),
            "-i",
            str(wav),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(dst),
        ]
    )
    return dst
