"""Video pipeline: pose -> tracking -> features -> fall logic -> annotation."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .annotate import annotate_frame
from .config import DEFAULT_FPS, PipelineConfig
from .fall_detector import FallDetector, FallEvent
from .features import FEATURE_NAMES, extract_features
from .model import FallClassifier
from .pose_estimator import Pose, PoseEstimator
from .tracker import CentroidTracker, Track

# Columns exported for one detected person per processed frame.
FEATURE_ROW_COLUMNS: tuple[str, ...] = (
    "frame_idx",
    "timestamp_s",
    "track_id",
    "posture",
    "ml_prob",
    "event",
    *FEATURE_NAMES,
)


@dataclass
class Detection:
    """One tracked person in one frame."""

    track_id: int
    pose: Pose
    features: dict[str, float]
    posture: str
    ml_prob: float | None
    event: FallEvent | None
    alert: bool
    trail: tuple[tuple[int, int], ...]


@dataclass
class FrameResult:
    frame_idx: int
    timestamp_s: float
    detections: list[Detection]
    annotated: np.ndarray


@dataclass
class VideoReport:
    source: Path
    output_path: Path | None
    width: int
    height: int
    fps: float
    frames_read: int
    frames_processed: int
    duration_s: float
    processing_fps: float
    events: list[FallEvent] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)


def target_size(
    src_w: int,
    src_h: int,
    frame_height: int,
    max_width: int,
) -> tuple[int, int]:
    """Resize to 480p while preserving aspect ratio and even dimensions."""
    if src_w <= 0 or src_h <= 0:
        return max_width, frame_height
    scale = min(frame_height / src_h, max_width / src_w)
    width = max(2, int(round(src_w * scale)))
    height = max(2, int(round(src_h * scale)))
    return width - width % 2, height - height % 2


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Prefer H.264 (browser friendly), fall back to mp4v."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            max(1.0, fps),
            size,
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not create video writer for {path}")


class FallPipeline:
    """Stateful per-video processor; create one instance per video."""

    def __init__(
        self,
        cfg: PipelineConfig | None = None,
        classifier: FallClassifier | None = None,
    ) -> None:
        self.cfg = cfg or PipelineConfig()
        self.estimator = PoseEstimator(self.cfg.model_path, self.cfg.num_poses)
        self.tracker = CentroidTracker()
        self.detector = FallDetector(self.cfg.detector)
        self.classifier = classifier
        self._prev: dict[int, dict[str, float]] = {}
        self._buffers: dict[int, deque[dict[str, float]]] = {}
        self._trails: dict[int, deque[tuple[int, int]]] = {}
        self._last_events: dict[int, FallEvent] = {}

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
        draw: bool = True,
    ) -> FrameResult:
        height, width = frame_bgr.shape[:2]
        poses = self.estimator.detect(frame_bgr, int(round(timestamp_s * 1000)))
        tracks = self.tracker.update([p.bbox for p in poses])

        detections: list[Detection] = []
        for pose, track in zip(poses, tracks):
            detection = self._process_person(
                pose, track, timestamp_s, frame_idx, width, height
            )
            detections.append(detection)

        annotated = (
            annotate_frame(
                frame_bgr, detections, timestamp_s, self.cfg.min_visibility
            )
            if draw
            else frame_bgr
        )
        return FrameResult(frame_idx, timestamp_s, detections, annotated)

    def _process_person(
        self,
        pose: Pose,
        track: Track,
        timestamp_s: float,
        frame_idx: int,
        width: int,
        height: int,
    ) -> Detection:
        track_id = track.track_id
        prev = self._prev.get(track_id)
        dt = timestamp_s - float(prev["t"]) if prev else 0.0
        features = extract_features(
            pose.landmarks, track.bbox, width, height, prev=prev, dt=dt
        )

        x1, y1, x2, y2 = (float(v) for v in track.bbox)
        self._prev[track_id] = {
            **features,
            "t": timestamp_s,
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
        }

        ml_prob: float | None = None
        if self.classifier is not None:
            buffer = self._buffers.setdefault(
                track_id, deque(maxlen=self.classifier.window)
            )
            buffer.append(features)
            ml_prob = self.classifier.probability(buffer)

        posture, event = self.detector.update(
            track_id, features, timestamp_s, frame_idx, ml_prob
        )
        if event is not None:
            self._last_events[track_id] = event
        last_event = self._last_events.get(track_id)
        alert = bool(
            last_event is not None
            and timestamp_s - last_event.timestamp_s <= 2.0
        )

        trail = self._trails.setdefault(track_id, deque(maxlen=40))
        trail.append((int(track.centroid[0]), int(track.centroid[1])))

        return Detection(
            track_id=track_id,
            pose=pose,
            features=features,
            posture=posture,
            ml_prob=ml_prob,
            event=event,
            alert=alert,
            trail=tuple(trail),
        )

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
        on_frame: Callable[[FrameResult], bool] | None = None,
        draw: bool = True,
    ) -> VideoReport:
        """Process a video file and optionally write an annotated copy."""
        source = Path(input_path)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")

        src_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(src_fps) or src_fps <= 0:
            src_fps = DEFAULT_FPS
        src_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        src_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width, height = target_size(
            src_w, src_h, self.cfg.frame_height, self.cfg.max_width
        )
        total_output = (
            max(1, math.ceil(total_frames / self.cfg.stride))
            if total_frames > 0
            else 0
        )

        writer: cv2.VideoWriter | None = None
        destination: Path | None = Path(output_path) if output_path else None
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            writer = _open_writer(
                destination, src_fps / self.cfg.stride, (width, height)
            )

        rows: list[dict] = []
        events: list[FallEvent] = []
        frames_read = 0
        processed = 0
        started = time.perf_counter()

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_idx = frames_read
                frames_read += 1
                if frame_idx % self.cfg.stride != 0:
                    continue

                frame = cv2.resize(
                    frame, (width, height), interpolation=cv2.INTER_AREA
                )
                timestamp_s = frame_idx / src_fps
                result = self.process_frame(
                    frame, frame_idx, timestamp_s, draw=draw
                )
                if writer is not None:
                    writer.write(result.annotated)

                for det in result.detections:
                    row = {
                        "frame_idx": frame_idx,
                        "timestamp_s": round(timestamp_s, 3),
                        "track_id": det.track_id,
                        "posture": det.posture,
                        "ml_prob": det.ml_prob,
                        "event": int(det.event is not None),
                    }
                    row.update({name: det.features[name] for name in FEATURE_NAMES})
                    rows.append(row)
                    if det.event is not None:
                        events.append(det.event)

                processed += 1
                if on_frame is not None and not on_frame(result):
                    break
                if progress_cb is not None and (
                    processed % 5 == 0 or processed == total_output
                ):
                    progress_cb(processed, total_output)
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        elapsed = max(time.perf_counter() - started, 1e-6)
        return VideoReport(
            source=source,
            output_path=destination,
            width=width,
            height=height,
            fps=src_fps,
            frames_read=frames_read,
            frames_processed=processed,
            duration_s=frames_read / src_fps,
            processing_fps=processed / elapsed,
            events=events,
            rows=rows,
        )

    def close(self) -> None:
        self.estimator.close()

    def __enter__(self) -> "FallPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
