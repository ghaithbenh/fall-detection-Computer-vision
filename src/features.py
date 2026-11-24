"""Geometry features computed from pose landmarks."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "torso_angle",
    "torso_len",
    "bbox_aspect",
    "bbox_height",
    "hip_y",
    "shoulder_y",
    "head_y",
    "knee_angle",
    "hip_drop_speed",
    "center_speed",
)

# MediaPipe Pose landmark indices.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at joint b in degrees (180 = straight limb)."""
    v1, v2 = a - b, c - b
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def extract_features(
    landmarks: np.ndarray,
    bbox: np.ndarray,
    frame_w: int,
    frame_h: int,
    prev: Mapping[str, float] | None = None,
    dt: float = 0.0,
) -> dict[str, float]:
    """Turn 33 landmarks into scale-invariant, interpretable features."""
    lm = np.asarray(landmarks, dtype=np.float32)
    xy = lm[:, :2] * np.array([frame_w, frame_h], dtype=np.float32)
    visibility = lm[:, 3]

    def visible(*indices: int) -> bool:
        return all(visibility[i] >= 0.4 for i in indices)

    if visible(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        shoulder = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2.0
        hip = (xy[L_HIP] + xy[R_HIP]) / 2.0
        vec = shoulder - hip
        torso_px = float(np.linalg.norm(vec))
        # 0 deg = vertical torso, 90 deg = horizontal torso.
        torso_angle = math.degrees(
            math.atan2(abs(float(vec[0])), abs(float(vec[1])) + 1e-6)
        )
        hip_y = float(hip[1] / frame_h)
        shoulder_y = float(shoulder[1] / frame_h)
    else:
        torso_px = math.nan
        torso_angle = hip_y = shoulder_y = math.nan

    torso_len = torso_px / frame_h if math.isfinite(torso_px) else math.nan
    x1, y1, x2, y2 = (float(v) for v in bbox)
    box_w, box_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    bbox_aspect = box_w / box_h
    bbox_height = box_h / frame_h
    head_y = float(xy[NOSE, 1] / frame_h) if visibility[NOSE] >= 0.4 else math.nan

    knee_angles = []
    for hip_i, knee_i, ankle_i in (
        (L_HIP, L_KNEE, L_ANKLE),
        (R_HIP, R_KNEE, R_ANKLE),
    ):
        if visible(hip_i, knee_i, ankle_i):
            knee_angles.append(_joint_angle(xy[hip_i], xy[knee_i], xy[ankle_i]))
    knee_angle = float(np.mean(knee_angles)) if knee_angles else math.nan

    hip_drop_speed = 0.0
    center_speed = 0.0
    if prev is not None and dt > 0 and math.isfinite(torso_px) and torso_px > 1.0:
        prev_hip_y = prev.get("hip_y", math.nan)
        if math.isfinite(prev_hip_y) and math.isfinite(hip_y):
            # Positive = hips moving down the image, in torso lengths / second.
            hip_drop_speed = max(
                0.0, (hip_y - prev_hip_y) / (torso_px / frame_h) / dt
            )
        prev_cx, prev_cy = prev.get("cx"), prev.get("cy")
        if prev_cx is not None and prev_cy is not None:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = math.hypot(cx - prev_cx, cy - prev_cy)
            center_speed = dist / torso_px / dt

    return {
        "torso_angle": torso_angle,
        "torso_len": torso_len,
        "bbox_aspect": bbox_aspect,
        "bbox_height": bbox_height,
        "hip_y": hip_y,
        "shoulder_y": shoulder_y,
        "head_y": head_y,
        "knee_angle": knee_angle,
        "hip_drop_speed": min(hip_drop_speed, 20.0),
        "center_speed": min(center_speed, 20.0),
    }
