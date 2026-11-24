"""Thin wrapper around the MediaPipe Tasks PoseLandmarker (VIDEO mode)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


@dataclass
class Pose:
    """One detected person."""

    landmarks: np.ndarray  # (33, 4): x, y, z, visibility (normalized)
    bbox: np.ndarray  # (4,): x1, y1, x2, y2 in pixels
    score: float  # mean visibility of the main body joints


class PoseEstimator:
    """Detects body landmarks in sequential BGR frames."""

    def __init__(self, model_path: str | Path, num_poses: int = 1) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Pose model not found: {path}. Run: python -m src.download_model"
            )
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=max(1, num_poses),
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Pose]:
        """Run pose detection; timestamps must increase across calls."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        # VIDEO mode rejects non-increasing timestamps.
        ts = max(int(timestamp_ms), self._last_ts_ms + 1)
        self._last_ts_ms = ts
        result = self._landmarker.detect_for_video(image, ts)
        return [self._to_pose(landmarks, w, h) for landmarks in result.pose_landmarks]

    @staticmethod
    def _to_pose(landmarks, width: int, height: int) -> Pose:
        arr = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
            dtype=np.float32,
        )
        visible = arr[arr[:, 3] >= 0.3, :2]
        points = visible if len(visible) >= 4 else arr[:, :2]
        x1, y1 = np.clip(points.min(axis=0), 0.0, 1.0) * [width, height]
        x2, y2 = np.clip(points.max(axis=0), 0.0, 1.0) * [width, height]
        key_joints = arr[[11, 12, 23, 24, 25, 26, 27, 28], 3]
        return Pose(
            landmarks=arr,
            bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
            score=float(key_joints.mean()),
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
