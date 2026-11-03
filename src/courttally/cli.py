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

    # --- recording / editing (no ML models needed) ---
    s = sub.add_parser("autoedit", help="ingest, find points, cut dead time, build review page")
    _add_common(s)
    _add_points_opts(s)
    s.add_argument(
        "--keypoints",
        type=Path,
        default=None,
        help="court keypoints JSON to mask motion to the court",
    )
    s.add_argument(
        "--mode",
        choices=["copy", "accurate"],
        default="copy",
        help="cut mode (copy = lossless/instant)",
    )
    s.add_argument(
        "--no-cut", action="store_true", help="skip writing clips; just points + review page"
    )

    s = sub.add_parser("ingest", help="probe the original and build proxy.mp4 for analysis/review")
    _add_common(s)
    s.add_argument("--proxy-width", type=int, default=1280)

    s = sub.add_parser("activity", help="motion + audio activity per frame -> activity.parquet")
    _add_common(s)
    s.add_argument("--audio-weight", type=float, default=0.5)

    s = sub.add_parser("points", help="segment activity into points -> points.json/.csv/.edl")
    _add_common(s)
    _add_points_opts(s)

    s = sub.add_parser("review", help="write review.html for tagging winners and trimming points")
    _add_common(s)
    s.add_argument(
        "--points",
        type=Path,
        default=None,
        help="points JSON (default: points.edited.json if present, else points.json)",
    )

    s = sub.add_parser("cut", help="export per-point clips + condensed.mp4 from the ORIGINAL")
    _add_common(s)
    s.add_argument("--points", type=Path, default=None)
    s.add_argument("--mode", choices=["copy", "accurate"], default="copy")
    s.add_argument("--condensed-only", action="store_true", help="do not keep per-point clips")

    s = sub.add_parser(
        "score", help="tennis score progression from tagged point winners -> score.json"
    )
    _add_common(s)
    s.add_argument("--points", type=Path, default=None)
    s.add_argument("--best-of", type=int, default=3)
    s.add_argument("--no-ad", action="store_true")
    return p


def _add_points_opts(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("point detection")
    g.add_argument("--on", type=float, default=0.5, help="activity level that starts a point")
    g.add_argument(
        "--off", type=float, default=0.25, help="activity level below which a point ends"
    )
    g.add_argument("--min-len", type=float, default=1.5, help="drop points shorter than this (s)")
    g.add_argument("--merge-gap", type=float, default=1.5, help="merge points closer than this (s)")
    g.add_argument("--pad-pre", type=float, default=1.0, help="seconds added before each point")
    g.add_argument("--pad-post", type=float, default=1.5, help="seconds added after each point")


def _thresholds(args: argparse.Namespace) -> dict:
    return {
        "on": args.on,
        "off": args.off,
        "min_len_s": args.min_len,
        "merge_gap_s": args.merge_gap,
        "pad_pre_s": args.pad_pre,
        "pad_post_s": args.pad_post,
    }


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
    elif args.cmd == "autoedit":
        pipeline.autoedit(
            args.clip,
            args.outputs,
            args.run,
            args.keypoints,
            mode=args.mode,
            cut_video=not args.no_cut,
            **_thresholds(args),
        )
        print(
            f"review: open {run_dir / 'review.html'} in a browser, tag winners, press E to export,"
        )
        print(f"        then: ct cut {args.clip} --points {run_dir / 'points.edited.json'}")
    elif args.cmd == "ingest":
        pipeline.stage_ingest(args.clip, run_dir, proxy_width=args.proxy_width)
    elif args.cmd == "activity":
        pipeline.stage_activity(args.clip, run_dir, audio_weight=args.audio_weight)
    elif args.cmd == "points":
        pipeline.stage_points(args.clip, run_dir, **_thresholds(args))
    elif args.cmd == "review":
        pipeline.stage_review(args.clip, run_dir, args.points)
    elif args.cmd == "cut":
        pipeline.stage_cut(
            args.clip,
            run_dir,
            mode=args.mode,
            points_file=args.points,
            per_point=not args.condensed_only,
        )
    elif args.cmd == "score":
        result = pipeline.stage_score(run_dir, args.points, best_of=args.best_of, no_ad=args.no_ad)
        print(json.dumps({k: v for k, v in result.items() if k != "progression"}, indent=2))
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
