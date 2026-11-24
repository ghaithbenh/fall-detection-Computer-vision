"""Run fall detection on a video file.

Example:
    python -m src.run_video --input clip.mp4 --show
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from pathlib import Path

import cv2
import pandas as pd

from .config import (
    DEFAULT_POSE_MODEL,
    OUTPUT_DIR,
    DetectorConfig,
    PipelineConfig,
)
from .fall_detector import FallEvent
from .model import FallClassifier
from .pipeline import FEATURE_ROW_COLUMNS, FallPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--classifier", type=Path, default=None)
    parser.add_argument("--num-poses", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--lying-angle", type=float, default=55.0)
    parser.add_argument("--drop-speed", type=float, default=2.0)
    parser.add_argument("--confirm", type=float, default=0.35)
    parser.add_argument("--no-output", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        print(f"Input not found: {args.input}")
        return 1

    detector_cfg = DetectorConfig(
        lying_angle=args.lying_angle,
        drop_speed=args.drop_speed,
        confirm_lying_s=args.confirm,
    )
    cfg = PipelineConfig(
        model_path=args.model,
        num_poses=max(1, args.num_poses),
        stride=max(1, args.stride),
        detector=detector_cfg,
    )
    classifier = (
        FallClassifier.load(args.classifier) if args.classifier else None
    )

    stem = args.input.stem
    output = (
        None
        if args.no_output
        else args.output or OUTPUT_DIR / f"{stem}_annotated.mp4"
    )
    features_csv = args.csv or OUTPUT_DIR / f"{stem}_features.csv"
    events_csv = args.events or OUTPUT_DIR / f"{stem}_events.csv"

    def on_frame(result) -> bool:
        if not args.show:
            return True
        cv2.imshow("Fall detection (q to quit)", result.annotated)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    print(f"Processing {args.input} ...")
    try:
        with FallPipeline(cfg, classifier) as pipeline:
            report = pipeline.process_video(
                args.input,
                output,
                on_frame=on_frame if args.show else None,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    finally:
        if args.show:
            cv2.destroyAllWindows()

    print(
        f"Done: {report.frames_processed} frames | "
        f"{report.width}x{report.height} | "
        f"{report.processing_fps:.1f} processed FPS"
    )
    if report.events:
        print(f"Fall events: {len(report.events)}")
        for event in report.events:
            print(
                f"  t={event.timestamp_s:6.2f}s | track {event.track_id} | "
                f"confidence {event.confidence:.2f} | {event.source}"
            )
    else:
        print("Fall events: none")
    if report.output_path is not None:
        print(f"Annotated video: {report.output_path}")

    if not args.no_csv:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        features_csv.parent.mkdir(parents=True, exist_ok=True)
        events_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report.rows, columns=FEATURE_ROW_COLUMNS).to_csv(
            features_csv, index=False
        )
        pd.DataFrame(
            [asdict(event) for event in report.events],
            columns=[field.name for field in fields(FallEvent)],
        ).to_csv(events_csv, index=False)
        print(f"Features CSV: {features_csv}")
        print(f"Events CSV: {events_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
