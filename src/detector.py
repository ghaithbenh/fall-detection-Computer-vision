"""MediaPipe PoseLandmarker wrapper for video frames."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"
)

# Pairs of landmark indices to connect when drawing the skeleton.
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
    (27, 31), (28, 32),
)


class PoseDetector:
    """Detects one person's 33 pose landmarks in sequential video frames."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_poses: int = 1,
    ) -> None:
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.is_file():
            self._download_model(path)

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

    def process(self, frame_bgr: np.ndarray, t_sec: float) -> np.ndarray | None:
        """Return (33, 3) landmarks as (x, y, visibility), or None.

        x is expressed in units of frame height (x_norm * width / height) so
        horizontal and vertical distances are not distorted by the aspect ratio.
        y stays normalized to frame height, and visibility is 0..1.
        """
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        # VIDEO mode rejects repeated or decreasing timestamps.
        timestamp_ms = max(int(t_sec * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        landmarks = np.array(
            [[lm.x, lm.y, lm.visibility] for lm in result.pose_landmarks[0]],
            dtype=np.float32,
        )
        landmarks[:, 0] *= width / height
        return landmarks

    @staticmethod
    def _download_model(path: Path) -> None:
        """Fetch the lite .task model the first time it is needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Model not found, downloading to {path} ...")
        try:
            urllib.request.urlretrieve(MODEL_URL, path)
        except (urllib.error.URLError, OSError) as exc:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download the pose model: {exc}. "
                f"Download it manually from {MODEL_URL}"
            ) from exc
        print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def draw_skeleton(
    frame: np.ndarray,
    landmarks: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    min_visibility: float = 0.4,
) -> np.ndarray:
    """Draw a skeleton in place and return the same frame.

    `landmarks` must be the scaled (33, 3) array returned by `process`.
    """
    height = frame.shape[0]
    points = np.rint(landmarks[:, :2] * height).astype(np.int32)
    visibility = landmarks[:, 2]

    for start, end in POSE_CONNECTIONS:
        if (
            visibility[start] >= min_visibility
            and visibility[end] >= min_visibility
        ):
            cv2.line(
                frame,
                tuple(points[start]),
                tuple(points[end]),
                color,
                thickness,
                cv2.LINE_AA,
            )
    for index, (x, y) in enumerate(points):
        if visibility[index] >= min_visibility:
            cv2.circle(
                frame,
                (int(x), int(y)),
                thickness + 1,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )
    return frame
