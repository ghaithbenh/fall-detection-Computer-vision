"""OpenCV drawing helpers; no dependency on the removed mp.solutions API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .pipeline import Detection

# Subset of MediaPipe pose connections, enough for a clear skeleton.
CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
)

POSTURE_COLORS = {
    "upright": (70, 200, 70),
    "transition": (0, 180, 255),
    "lying": (0, 0, 255),
    "unknown": (170, 170, 170),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_skeleton(
    canvas: np.ndarray,
    landmarks: np.ndarray,
    color: tuple[int, int, int],
    min_visibility: float,
) -> None:
    height, width = canvas.shape[:2]
    points = landmarks[:, :2] * np.array([width, height], dtype=np.float32)
    for a, b in CONNECTIONS:
        if landmarks[a, 3] >= min_visibility and landmarks[b, 3] >= min_visibility:
            cv2.line(
                canvas,
                tuple(points[a].astype(int)),
                tuple(points[b].astype(int)),
                color,
                2,
                cv2.LINE_AA,
            )
    for index in range(len(landmarks)):
        if landmarks[index, 3] >= min_visibility:
            cv2.circle(
                canvas,
                tuple(points[index].astype(int)),
                2,
                color,
                -1,
                cv2.LINE_AA,
            )


def _put_label(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    height, width = canvas.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    x = min(max(2, x), max(2, width - tw - 8))
    y = min(max(th + 4, y), height - 4)
    cv2.rectangle(canvas, (x, y - th - 4), (x + tw + 6, y + 2), (0, 0, 0), -1)
    cv2.putText(
        canvas, text, (x + 3, y - 2), FONT, 0.5, color, 1, cv2.LINE_AA
    )


def annotate_frame(
    frame: np.ndarray,
    detections: list["Detection"],
    timestamp_s: float,
    min_visibility: float = 0.5,
) -> np.ndarray:
    """Draw tracks, skeletons and fall alerts on a copy of the frame."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    for det in detections:
        color = POSTURE_COLORS.get(det.posture, POSTURE_COLORS["unknown"])
        if len(det.trail) >= 2:
            cv2.polylines(
                canvas,
                [np.array(det.trail, dtype=np.int32)],
                False,
                color,
                1,
                cv2.LINE_AA,
            )
        _draw_skeleton(canvas, det.pose.landmarks, color, min_visibility)

        x1, y1, x2, y2 = det.pose.bbox.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"ID {det.track_id} | {det.posture.upper()}"
        angle = det.features.get("torso_angle", float("nan"))
        if np.isfinite(angle):
            label += f" | {angle:.0f} deg"
        if det.ml_prob is not None:
            label += f" | ML {det.ml_prob:.2f}"
        _put_label(canvas, label, x1, y1 - 6, color)

    if any(det.alert for det in detections):
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (0, 0, 255), 6)
        text = "FALL DETECTED"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.9, 2)
        cv2.putText(
            canvas,
            text,
            ((width - tw) // 2, th + 18),
            FONT,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"t={timestamp_s:6.1f}s",
        (width - 105, height - 12),
        FONT,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas
