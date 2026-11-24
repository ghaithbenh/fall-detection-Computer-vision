"""Landmark-only fall state machine: STANDING -> FALLING -> LYING -> ALERT.

The other detectors in this repo need extra inputs (tracked tracks, feature
CSVs, an ML model). This one works from a single frame of pose landmarks, so
it can be dropped into a video loop or a live camera check.

The bbox used by the rules is estimated from the visible landmarks, since
``update`` deliberately receives only the landmarks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .features import extract_features

VISIBILITY_THR = 0.4


class State(str, Enum):
    STANDING = "standing"
    FALLING = "falling"
    LYING = "lying"
    ALERT = "alert"


@dataclass(frozen=True)
class Params:
    """Thresholds for the state machine.

    Speeds are in torso lengths per second (as in
    :func:`src.features.extract_features`), so they do not depend on how
    large the person is on screen.
    """

    frame_size: tuple[int, int] = (640, 480)  # used to un-normalise landmarks
    drop_speed: float = 2.0  # hips moving down this fast starts a FALLING
    fall_window_s: float = 1.5  # time allowed to actually end up horizontal
    lying_angle: float = 55.0  # torso this far from vertical counts as lying
    upright_angle: float = 35.0  # torso this close to vertical counts as up
    lying_aspect: float = 1.0  # bbox wider than tall also counts as lying
    stillness_s: float = 2.0  # lying still this long triggers the alert
    max_lying_s: float = 8.0  # lying this long triggers even while moving
    still_speed: float = 0.4  # bbox motion below this counts as still
    still_angle_rate: float = 30.0  # torso rotation below this (deg/s) counts as still
    recover_s: float = 0.3  # upright this long to leave LYING / ALERT
    min_torso_len: float = 0.05  # ignore frames with no usable torso


class FallDetector:
    """State machine that turns per-frame landmarks into a fall decision."""

    def __init__(self, params: Params | None = None) -> None:
        self.params = params or Params()
        self.state = State.STANDING
        self.state_since = 0.0
        self.alert_time: float | None = None
        self.alert_count = 0

        self._fall_start: float | None = None
        self._lying_since: float | None = None
        self._still_since: float | None = None
        self._upright_since: float | None = None

        self._prev_t: float | None = None
        self._prev_features: dict[str, float] | None = None
        self._prev_cx: float | None = None
        self._prev_cy: float | None = None
        self._prev_angle: float = math.nan
        self._angle_rate = 0.0

    def update(
        self, t: float, landmarks: np.ndarray
    ) -> tuple[State, dict[str, float]]:
        """Advance the machine with one frame.

        Args:
            t: Timestamp in seconds, non-decreasing across calls.
            landmarks: ``(33, 4)`` array of (x, y, z, visibility), with
                ``x``/``y`` normalised to the frame.

        Returns:
            The state after this frame and the geometry features used to
            decide it. Features keep their previous meaning on frames with
            no visible pose (NaN torso angle, zero speeds).
        """
        features = self._features(t, landmarks)
        angle = features["torso_angle"]
        aspect = features["bbox_aspect"]
        speed = features["hip_drop_speed"]
        torso_len = features["torso_len"]
        p = self.params
        usable = (
            math.isfinite(angle)
            and math.isfinite(torso_len)
            and torso_len >= p.min_torso_len
        )

        if self.state is State.STANDING:
            # Only a fast drop starts a fall; slow lying down is ignored.
            if usable and speed >= p.drop_speed:
                self._go(State.FALLING, t)

        elif self.state is State.FALLING:
            if usable and (angle >= p.lying_angle or aspect >= p.lying_aspect):
                self._go(State.LYING, t)
            elif (
                self._fall_start is not None
                and t - self._fall_start > p.fall_window_s
            ):
                # Fast drop but never horizontal: a fast sit or a stumble.
                self._go(State.STANDING, t)

        if self.state is State.LYING:
            if usable and angle <= p.upright_angle:
                # Person is getting up; require a moment of upright frames.
                if self._upright_since is None:
                    self._upright_since = t
                elif t - self._upright_since >= p.recover_s:
                    self._go(State.STANDING, t)
            else:
                self._upright_since = None

            if self.state is State.LYING:
                still = (
                    usable
                    and features["center_speed"] < p.still_speed
                    and self._angle_rate < p.still_angle_rate
                )
                if still:
                    if self._still_since is None:
                        self._still_since = t
                else:
                    self._still_since = None

                lying_for = (
                    t - self._lying_since if self._lying_since is not None else 0.0
                )
                still_for = (
                    t - self._still_since if self._still_since is not None else 0.0
                )
                if still_for >= p.stillness_s or lying_for >= p.max_lying_s:
                    self._go(State.ALERT, t)

        elif self.state is State.ALERT:
            # Stay alert until the person is properly back on their feet.
            if usable and angle <= p.upright_angle:
                if self._upright_since is None:
                    self._upright_since = t
                elif t - self._upright_since >= p.recover_s:
                    self._go(State.STANDING, t)
            else:
                self._upright_since = None

        return self.state, features

    def reset(self) -> None:
        """Forget everything, as if no frames had been seen."""
        self.state = State.STANDING
        self.state_since = 0.0
        self.alert_time = None
        self.alert_count = 0
        self._fall_start = None
        self._lying_since = None
        self._still_since = None
        self._upright_since = None
        self._prev_t = None
        self._prev_features = None
        self._prev_cx = None
        self._prev_cy = None
        self._prev_angle = math.nan
        self._angle_rate = 0.0

    def _go(self, state: State, t: float) -> None:
        self.state = state
        self.state_since = t
        self._upright_since = None
        if state is State.FALLING:
            self._fall_start = t
        elif state is State.LYING:
            self._lying_since = t
            self._still_since = None
        elif state is State.ALERT:
            self.alert_time = t
            self.alert_count += 1
        else:  # STANDING
            self._fall_start = None
            self._lying_since = None
            self._still_since = None

    def _features(
        self, t: float, landmarks: np.ndarray
    ) -> dict[str, float]:
        lm = np.asarray(landmarks, dtype=np.float32)
        if lm.ndim != 2 or lm.shape[0] < 29 or lm.shape[1] < 4:
            raise ValueError(f"expected landmarks shaped (33, 4), got {lm.shape}")

        width, height = self.params.frame_size
        bbox = _landmark_bbox(lm, width, height)
        dt = 0.0 if self._prev_t is None else t - self._prev_t
        if not math.isfinite(dt) or dt < 0:
            dt = 0.0

        prev = None
        if self._prev_features is not None and dt > 0:
            # extract_features needs the previous bbox centre to measure motion.
            prev = dict(self._prev_features)
            prev["cx"] = self._prev_cx if self._prev_cx is not None else math.nan
            prev["cy"] = self._prev_cy if self._prev_cy is not None else math.nan

        features = extract_features(lm, bbox, width, height, prev=prev, dt=dt)
        angle = features["torso_angle"]
        if dt > 0 and math.isfinite(angle) and math.isfinite(self._prev_angle):
            self._angle_rate = abs(angle - self._prev_angle) / dt
        else:
            self._angle_rate = 0.0

        self._prev_t = t
        self._prev_features = features
        self._prev_cx = float((bbox[0] + bbox[2]) / 2.0)
        self._prev_cy = float((bbox[1] + bbox[3]) / 2.0)
        self._prev_angle = angle
        return features


def _landmark_bbox(
    landmarks: np.ndarray, frame_w: int, frame_h: int
) -> np.ndarray:
    """Axis-aligned bbox in pixels around the visible landmarks."""
    visible = landmarks[:, 3] >= VISIBILITY_THR
    xy = landmarks[visible, :2] if bool(visible.any()) else landmarks[:, :2]
    scale = np.array([frame_w, frame_h], dtype=np.float32)
    low = xy.min(axis=0) * scale
    high = xy.max(axis=0) * scale
    return np.array([low[0], low[1], high[0], high[1]], dtype=np.float32)
