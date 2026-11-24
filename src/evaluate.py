"""Evaluate the baseline and the full fall detector on labelled clips.

Pose landmarks are extracted once per clip and cached as ``.npz`` files, so
repeated runs (including threshold tuning) do not re-run MediaPipe.

    python -m src.evaluate --split test
    python -m src.evaluate --split test --tune

Labels come from ``data/labels.csv`` (``clip,label,split,fall_start``), as
written by ``scripts/make_labels.py``. Detection is clip-level: a clip is
predicted positive if the detector reports at least one fall in it.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

from .baseline import DEFAULT_ANGLE_THR, DEFAULT_HOLD_S, baseline_detect
from .config import (
    DATA_DIR,
    DEFAULT_FPS,
    DEFAULT_POSE_MODEL,
    DetectorConfig,
    FRAME_HEIGHT,
    MAX_FRAME_WIDTH,
)
from .fall_detector import FallDetector
from .features import extract_features
from .pipeline import target_size
from .pose_estimator import PoseEstimator

DEFAULT_LABELS = DATA_DIR / "labels.csv"
DEFAULT_RAW = DATA_DIR / "raw"
DEFAULT_CACHE = DATA_DIR / "pose_cache"

# Small grids: enough to show the trade-off without minutes of tuning.
BASELINE_GRID: tuple[dict[str, float], ...] = tuple(
    {"angle_thr": angle, "hold": hold}
    for angle in (45.0, 55.0, 65.0, 75.0)
    for hold in (0.5, 1.0, 1.5)
)
DETECTOR_GRID: tuple[dict[str, float], ...] = tuple(
    {"lying_angle": lying, "drop_speed": drop, "confirm_lying_s": confirm}
    for lying in (45.0, 55.0, 65.0)
    for drop in (1.0, 1.5, 2.5)
    for confirm in (0.2, 0.5)
)


@dataclass
class ClipData:
    """One labelled clip with cached landmarks turned into features."""

    clip: str
    label: int
    split: str
    fall_start: float | None
    times: np.ndarray
    angles: np.ndarray
    feature_times: np.ndarray
    features: list[dict[str, float]]


@dataclass
class ClipResult:
    clip: str
    label: int
    predicted: bool
    detected_at: float | None
    delay: float | None
    fall_start: float | None


@dataclass
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denom if denom else 0.0


def _cache_path(cache_dir: Path, clip: str) -> Path:
    # Keep the folder origin in the name so falls/ and nonfalls/ cannot clash.
    return cache_dir / f"{clip.replace('/', '__').replace(chr(92), '__')}.npz"


def _is_fresh(
    cache: Path, video: Path, clip: str, stride: int
) -> bool:
    if not cache.is_file():
        return False
    try:
        with np.load(cache, allow_pickle=False) as data:
            if str(data["clip"]) != clip or int(data["stride"]) != stride:
                return False
            if not video.is_file():
                return True  # cache-only use is fine
            stat = video.stat()
            return (
                int(data["mtime_ns"]) == stat.st_mtime_ns
                and int(data["size"]) == stat.st_size
            )
    except (OSError, KeyError, ValueError):
        return False


def extract_landmarks(
    video: Path,
    clip: str,
    cache_dir: Path,
    stride: int,
    model_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Return ``(times, landmarks, bboxes, frame_size)``, using the cache."""
    cache = _cache_path(cache_dir, clip)
    if _is_fresh(cache, video, clip, stride):
        with np.load(cache, allow_pickle=False) as data:
            return (
                data["times"].copy(),
                data["landmarks"].copy(),
                data["bboxes"].copy(),
                tuple(int(v) for v in data["frame_size"]),
            )

    if not video.is_file():
        raise FileNotFoundError(f"Clip not found: {video}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = DEFAULT_FPS
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = target_size(src_w, src_h, FRAME_HEIGHT, MAX_FRAME_WIDTH)

    times: list[float] = []
    landmarks: list[np.ndarray] = []
    bboxes: list[np.ndarray] = []
    frame_idx = 0
    with PoseEstimator(model_path, num_poses=1) as estimator:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                if (frame.shape[1], frame.shape[0]) != (width, height):
                    frame = cv2.resize(
                        frame, (width, height), interpolation=cv2.INTER_AREA
                    )
                timestamp_s = frame_idx / fps
                poses = estimator.detect(
                    frame, int(round(timestamp_s * 1000))
                )
                # One person per clip; keep the most visible detection.
                pose = max(poses, key=lambda p: p.score) if poses else None
                times.append(timestamp_s)
                landmarks.append(
                    pose.landmarks.copy()
                    if pose is not None
                    else np.full((33, 4), np.nan, dtype=np.float32)
                )
                bboxes.append(
                    pose.bbox.copy()
                    if pose is not None
                    else np.full(4, np.nan, dtype=np.float32)
                )
            frame_idx += 1
    cap.release()

    if not times:
        raise RuntimeError(f"No frames read from {video}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = video.stat()
    np.savez_compressed(
        cache,
        clip=np.array(clip),
        stride=np.array(stride),
        mtime_ns=np.array(stat.st_mtime_ns, dtype=np.int64),
        size=np.array(stat.st_size, dtype=np.int64),
        times=np.asarray(times, dtype=np.float64),
        landmarks=np.asarray(landmarks, dtype=np.float32),
        bboxes=np.asarray(bboxes, dtype=np.float32),
        frame_size=np.asarray([width, height], dtype=np.int32),
    )
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(landmarks, dtype=np.float32),
        np.asarray(bboxes, dtype=np.float32),
        (width, height),
    )


def build_clip_data(
    row: Mapping[str, object],
    raw_dir: Path,
    cache_dir: Path,
    stride: int,
    model_path: Path,
) -> ClipData:
    """Extract landmarks and per-frame geometry features for one clip."""
    clip = str(row["clip"])
    times, landmarks, bboxes, (width, height) = extract_landmarks(
        raw_dir / clip, clip, cache_dir, stride, model_path
    )

    angles = np.full(len(times), np.nan, dtype=np.float64)
    feature_times: list[float] = []
    features: list[dict[str, float]] = []
    prev: dict[str, float] | None = None
    for i, (t, lm, bbox) in enumerate(zip(times, landmarks, bboxes)):
        if not np.isfinite(lm).any():
            prev = None  # pose lost: do not bridge the gap
            continue
        dt = float(t) - float(prev["t"]) if prev else 0.0
        values = extract_features(lm, bbox, width, height, prev=prev, dt=dt)
        angles[i] = values["torso_angle"]
        feature_times.append(float(t))
        x1, y1, x2, y2 = (float(v) for v in bbox)
        prev = {
            **values,
            "t": float(t),
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
        }
        features.append(values)

    fall_start = row.get("fall_start")
    if fall_start is None or not math.isfinite(float(fall_start)):
        parsed_start = None
    else:
        parsed_start = float(fall_start)

    return ClipData(
        clip=clip,
        label=int(row["label"]),
        split=str(row["split"]),
        fall_start=parsed_start,
        times=times,
        angles=angles,
        feature_times=np.asarray(feature_times, dtype=np.float64),
        features=features,
    )


def _detector_time(
    data: ClipData, cfg: DetectorConfig
) -> float | None:
    detector = FallDetector(cfg)
    for frame_idx, values in enumerate(data.features):
        _, event = detector.update(
            1, values, float(data.feature_times[frame_idx]), frame_idx
        )
        if event is not None:
            return event.timestamp_s
    return None


def _result(
    data: ClipData, detected_at: float | None
) -> ClipResult:
    delay = None
    if (
        data.label == 1
        and detected_at is not None
        and data.fall_start is not None
    ):
        delay = detected_at - data.fall_start
    return ClipResult(
        clip=data.clip,
        label=data.label,
        predicted=detected_at is not None,
        detected_at=detected_at,
        delay=delay,
        fall_start=data.fall_start,
    )


def run_baseline(
    clips: Sequence[ClipData],
    angle_thr: float = DEFAULT_ANGLE_THR,
    hold: float = DEFAULT_HOLD_S,
) -> list[ClipResult]:
    return [
        _result(
            data,
            baseline_detect(
                data.times, data.angles, angle_thr=angle_thr, hold=hold
            ),
        )
        for data in clips
    ]


def run_detector(
    clips: Sequence[ClipData],
    cfg: DetectorConfig | None = None,
) -> list[ClipResult]:
    cfg = cfg or DetectorConfig()
    return [_result(data, _detector_time(data, cfg)) for data in clips]


def compute_metrics(results: Iterable[ClipResult]) -> Metrics:
    tp = fp = tn = fn = 0
    for result in results:
        if result.label == 1:
            if result.predicted:
                tp += 1
            else:
                fn += 1
        elif result.predicted:
            fp += 1
        else:
            tn += 1
    return Metrics(tp=tp, fp=fp, tn=tn, fn=fn)


def mean_delay(results: Sequence[ClipResult]) -> tuple[float | None, int]:
    delays = [r.delay for r in results if r.delay is not None]
    if not delays:
        return None, 0
    return float(np.mean(delays)), len(delays)


def _print_model(
    name: str,
    results: Sequence[ClipResult],
    detail: str,
) -> Metrics:
    metrics = compute_metrics(results)
    delay, delay_n = mean_delay(results)
    print(f"\n{name}")
    print(f"  {detail}")
    print(
        f"  precision {metrics.precision:.3f} | recall {metrics.recall:.3f} "
        f"| F1 {metrics.f1:.3f}"
    )
    print(
        "  confusion matrix [[TN, FP], [FN, TP]]: "
        f"[[{metrics.tn}, {metrics.fp}], [{metrics.fn}, {metrics.tp}]]"
    )
    if delay is None:
        print("  mean detection delay: n/a (no true positives with fall_start)")
    else:
        print(f"  mean detection delay: {delay:+.2f} s over {delay_n} clip(s)")

    wrong = [r for r in results if r.predicted != (r.label == 1)]
    if wrong:
        print("  wrongly classified clips:")
        for result in wrong:
            truth = "fall" if result.label == 1 else "no-fall"
            if result.predicted:
                guess = (
                    f"fall @ {result.detected_at:.2f}s"
                    if result.detected_at is not None
                    else "fall"
                )
            else:
                guess = "no-fall"
            print(f"    {result.clip:<36} true={truth:<7} pred={guess}")
    else:
        print("  wrongly classified clips: none")
    return metrics


def _score(metrics: Metrics) -> tuple[float, float, int]:
    # Prefer F1, then recall, then fewer false alarms.
    return metrics.f1, metrics.recall, -metrics.fp


def tune_baseline(
    clips: Sequence[ClipData],
) -> tuple[dict[str, float], Metrics]:
    best_params = {"angle_thr": DEFAULT_ANGLE_THR, "hold": DEFAULT_HOLD_S}
    best_metrics = compute_metrics(run_baseline(clips, **best_params))
    for params in BASELINE_GRID:
        metrics = compute_metrics(run_baseline(clips, **params))
        if _score(metrics) > _score(best_metrics):
            best_params, best_metrics = dict(params), metrics
    return best_params, best_metrics


def tune_detector(
    clips: Sequence[ClipData],
    base: DetectorConfig,
) -> tuple[DetectorConfig, Metrics]:
    best_cfg = base
    best_metrics = compute_metrics(run_detector(clips, best_cfg))
    for params in DETECTOR_GRID:
        cfg = replace(base, **params)
        metrics = compute_metrics(run_detector(clips, cfg))
        if _score(metrics) > _score(best_metrics):
            best_cfg, best_metrics = cfg, metrics
    return best_cfg, best_metrics


def load_labelled_clips(
    labels_path: Path,
    split: str,
) -> list[dict[str, object]]:
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"Labels not found: {labels_path}. Run: "
            "python scripts/make_labels.py"
        )
    df = pd.read_csv(labels_path)
    required = {"clip", "label", "split", "fall_start"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{labels_path} is missing columns: {sorted(missing)}"
        )
    df = df.copy()
    df["clip"] = df["clip"].astype(str)
    df["label"] = df["label"].astype(int)
    df["split"] = df["split"].astype(str)
    df["fall_start"] = pd.to_numeric(df["fall_start"], errors="coerce")
    if split != "all":
        df = df[df["split"] == split]
    if df.empty:
        raise ValueError(f"No clips with split={split!r} in {labels_path}")
    return df.to_dict("records")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the baseline and full fall detector by clip."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test", "all"),
        help="Split to report on (default: test)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Grid-search thresholds on the validation split first",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_POSE_MODEL)
    args = parser.parse_args(argv)

    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if not args.model.is_file():
        print(
            f"Pose model not found: {args.model}. "
            "Run: python -m src.download_model"
        )
        return 1

    try:
        rows = load_labelled_clips(args.labels, args.split)
        val_rows = (
            load_labelled_clips(args.labels, "val") if args.tune else []
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    # Load every clip once: tuning only re-runs the cheap detectors.
    needed = {str(r["clip"]): r for r in [*rows, *val_rows]}
    loaded: dict[str, ClipData] = {}
    for clip, row in needed.items():
        print(f"Loading {clip} ...", flush=True)
        try:
            loaded[clip] = build_clip_data(
                row, args.raw, args.cache, args.stride, args.model
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"error: {exc}")
            return 1

    clips = [loaded[str(r["clip"])] for r in rows]
    val_clips = [loaded[str(r["clip"])] for r in val_rows]

    base_cfg = DetectorConfig()
    baseline_params: dict[str, float] = {
        "angle_thr": DEFAULT_ANGLE_THR,
        "hold": DEFAULT_HOLD_S,
    }
    if args.tune:
        if not val_clips:
            print("error: --tune needs a non-empty validation split")
            return 1
        print(f"\nTuning on val: {len(val_clips)} clip(s)")
        baseline_params, baseline_val = tune_baseline(val_clips)
        base_cfg, detector_val = tune_detector(val_clips, base_cfg)
        print(
            "  baseline best: "
            f"angle_thr={baseline_params['angle_thr']:.0f} "
            f"hold={baseline_params['hold']:.1f}s "
            f"(val F1 {baseline_val.f1:.3f})"
        )
        print(
            "  detector best: "
            f"lying_angle={base_cfg.lying_angle:.0f} "
            f"drop_speed={base_cfg.drop_speed:.1f} "
            f"confirm_lying_s={base_cfg.confirm_lying_s:.1f}s "
            f"(val F1 {detector_val.f1:.3f})"
        )

    positives = sum(1 for data in clips if data.label == 1)
    print(
        f"\nSplit: {args.split} | clips: {len(clips)} "
        f"({positives} fall, {len(clips) - positives} no-fall)"
    )
    _print_model(
        "Baseline (torso horizontal for a fixed hold)",
        run_baseline(clips, **baseline_params),
        "angle_thr="
        f"{baseline_params['angle_thr']:.0f} deg, "
        f"hold={baseline_params['hold']:.1f}s",
    )
    _print_model(
        "Full detector (upright -> lying transition)",
        run_detector(clips, base_cfg),
        "lying_angle="
        f"{base_cfg.lying_angle:.0f}, drop_speed={base_cfg.drop_speed:.1f}, "
        f"confirm_lying_s={base_cfg.confirm_lying_s:.2f}s, "
        f"window={base_cfg.fall_window_s:.1f}s",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
