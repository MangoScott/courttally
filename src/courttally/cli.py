"""``ct`` command line. Every stage is its own subcommand so it can be re-run alone."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from courttally import __version__, pipeline, video


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("clip", type=Path, help="input video (mp4/mov)")
    p.add_argument("--outputs", type=Path, default=Path("outputs"), help="root for run folders")
    p.add_argument("--run", default=None, help="run folder name (default: clip stem)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ct", description="CourtTally tennis video analysis")
    p.add_argument("--version", action="version", version=f"courttally {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="print clip metadata")
    s.add_argument("clip", type=Path)

    s = sub.add_parser("analyze", help="run every stage end to end")
    _add_common(s)
    s.add_argument(
        "--keypoints", type=Path, default=None, help="JSON of 14 hand-labelled court keypoints"
    )

    s = sub.add_parser("court", help="court keypoints + homography -> court.json")
    _add_common(s)
    s.add_argument("--keypoints", type=Path, default=None)

    s = sub.add_parser("ball", help="ball positions -> ball.parquet")
    _add_common(s)

    s = sub.add_parser("players", help="player boxes -> players.parquet")
    _add_common(s)

    s = sub.add_parser("annotate", help="render annotated.mp4 from cached stage outputs")
    _add_common(s)

    s = sub.add_parser("stats", help="write stats.json from cached stage outputs")
    _add_common(s)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "info":
        info = video.probe(args.clip)
        print(
            json.dumps(
                {k: (str(v) if isinstance(v, Path) else v) for k, v in info.__dict__.items()}
                | {"duration_s": round(info.duration_s, 2)},
                indent=2,
            )
        )
        return 0

    run_dir = pipeline.run_dir_for(args.clip, args.outputs, args.run)
    if args.cmd == "analyze":
        pipeline.analyze(args.clip, args.outputs, args.run, args.keypoints)
    elif args.cmd == "court":
        pipeline.stage_court(args.clip, run_dir, keypoints_json=args.keypoints)
    elif args.cmd == "ball":
        pipeline.stage_ball(args.clip, run_dir)
    elif args.cmd == "players":
        pipeline.stage_players(args.clip, run_dir)
    elif args.cmd == "annotate":
        pipeline.stage_annotate(args.clip, run_dir)
    elif args.cmd == "stats":
        print(json.dumps(pipeline.stage_stats(args.clip, run_dir), indent=2))
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
