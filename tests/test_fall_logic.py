"""State-machine tests on synthetic skeletons (no video or model needed)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.fall_logic import FallDetector, Params, State

FPS = 25.0
DT = 1.0 / FPS
FRAME_W, FRAME_H = 640, 480


def skeleton(angle: float, hip_y: float, hip_x: float = 0.5) -> np.ndarray:
    """Synthetic skeleton with the torso ``angle`` degrees from vertical.

    0 deg is upright, 90 deg is horizontal. Geometry is built in pixel
    space (640x480) and then normalised, so the angle measured by
    :mod:`src.features` matches the requested one instead of being skewed
    by the frame aspect ratio.
    """
    rad = math.radians(angle)
    up = np.array([math.sin(rad), -math.cos(rad)])  # hip -> shoulder
    side = np.array([math.cos(rad), math.sin(rad)])  # perpendicular
    hip = np.array([hip_x * FRAME_W, hip_y * FRAME_H])
    shoulder = hip + 110.0 * up
    knee = hip - 60.0 * up
    ankle = hip - 120.0 * up
    nose = shoulder + 18.0 * up

    lm = np.zeros((33, 4), dtype=np.float32)

    def put(idx: int, px: np.ndarray, offset: float = 0.0) -> None:
        x, y = px + offset * side
        lm[idx] = [x / FRAME_W, y / FRAME_H, 0.0, 0.9]

    put(0, nose)
    put(11, shoulder, -6.0)
    put(12, shoulder, 6.0)
    put(23, hip, -6.0)
    put(24, hip, 6.0)
    put(25, knee, -6.0)
    put(26, knee, 6.0)
    put(27, ankle, -6.0)
    put(28, ankle, 6.0)
    return lm


def run(
    detector: FallDetector, poses: list[tuple[float, float]], start_t: float = 0.0
) -> tuple[list[State], list[float]]:
    """Feed ``(angle, hip_y)`` poses at 25 fps and collect the states."""
    states: list[State] = []
    times: list[float] = []
    for i, (angle, hip_y) in enumerate(poses):
        t = start_t + i * DT
        state, _ = detector.update(t, skeleton(angle, hip_y))
        states.append(state)
        times.append(t)
    return states, times


def fast_fall() -> list[tuple[float, float]]:
    """Stand, then drop to the floor in about 0.3 s while going horizontal."""
    poses = [(0.0, 0.45)] * 10
    poses += [(min(85.0, 11.0 * i), 0.45 + 0.05 * i) for i in range(1, 9)]
    return poses


class FallLogicTests(unittest.TestCase):
    def test_fast_fall_reaches_alert(self) -> None:
        detector = FallDetector()
        states, _ = run(detector, fast_fall() + [(85.0, 0.85)] * 80)

        self.assertIn(State.FALLING, states)
        self.assertIn(State.LYING, states)
        self.assertIn(State.ALERT, states)
        self.assertIsNotNone(detector.alert_time)
        self.assertEqual(states[-1], State.ALERT)

    def test_alert_after_two_seconds_of_stillness(self) -> None:
        detector = FallDetector()
        states, times = run(detector, fast_fall() + [(85.0, 0.85)] * 80)

        lying_t = times[states.index(State.LYING)]
        self.assertGreaterEqual(detector.alert_time - lying_t, 1.9)
        self.assertLessEqual(detector.alert_time - lying_t, 2.6)

    def test_slow_lying_down_never_alerts(self) -> None:
        # 0.4 torso lengths of hip travel spread over 4 s: far too slow.
        detector = FallDetector()
        steps = 100
        poses = [(0.0, 0.45)] * 10
        poses += [
            (85.0 * i / steps, 0.45 + 0.40 * i / steps)
            for i in range(1, steps + 1)
        ]
        poses += [(85.0, 0.85)] * 100
        states, _ = run(detector, poses)

        self.assertNotIn(State.FALLING, states)
        self.assertNotIn(State.ALERT, states)
        self.assertIsNone(detector.alert_time)
        self.assertEqual(states[-1], State.STANDING)

    def test_fast_sitting_returns_to_standing(self) -> None:
        # Hips drop quickly, but the torso stays upright, so it is not a fall.
        detector = FallDetector()
        poses = [(0.0, 0.45)] * 10
        poses += [(10.0, 0.45 + 0.05 * i) for i in range(1, 7)]
        poses += [(10.0, 0.75)] * 80
        states, _ = run(detector, poses)

        self.assertIn(State.FALLING, states)
        self.assertNotIn(State.LYING, states)
        self.assertNotIn(State.ALERT, states)
        self.assertEqual(states[-1], State.STANDING)

    def test_bending_then_getting_up_never_alerts(self) -> None:
        # A quick bend looks like a fall for a moment (fast hips, torso over
        # 55 deg), but the person is upright again well before 2 s of stillness.
        detector = FallDetector()
        poses = [(0.0, 0.45)] * 10
        poses += [(65.0, 0.45 + 0.025 * i) for i in range(1, 5)]
        poses += [(65.0, 0.55)] * 15
        poses += [(10.0, 0.50)] * 20
        states, _ = run(detector, poses)

        self.assertIn(State.LYING, states)
        self.assertNotIn(State.ALERT, states)
        self.assertIsNone(detector.alert_time)
        self.assertEqual(states[-1], State.STANDING)

    def test_moving_on_the_floor_alerts_at_eight_seconds(self) -> None:
        # Fidgeting keeps the stillness timer from running, so the hard cap
        # applies instead of the 2 s rule.
        detector = FallDetector()
        poses = fast_fall()
        poses += [(75.0 if i % 2 else 85.0, 0.85) for i in range(260)]
        states, times = run(detector, poses)

        lying_t = times[states.index(State.LYING)]
        self.assertIn(State.ALERT, states)
        self.assertGreaterEqual(detector.alert_time - lying_t, 7.8)
        self.assertLessEqual(detector.alert_time - lying_t, 8.4)

    def test_getting_up_after_alert_returns_to_standing(self) -> None:
        detector = FallDetector()
        poses = fast_fall() + [(85.0, 0.85)] * 80
        states, _ = run(detector, poses)
        self.assertEqual(states[-1], State.ALERT)

        start = len(poses) * DT
        states, _ = run(detector, [(5.0, 0.45)] * 20, start_t=start)
        self.assertEqual(states[-1], State.STANDING)
        self.assertEqual(detector.alert_count, 1)

    def test_single_fall_alerts_once(self) -> None:
        detector = FallDetector()
        states, _ = run(detector, fast_fall() + [(85.0, 0.85)] * 200)
        self.assertIn(State.ALERT, states)
        self.assertEqual(detector.alert_count, 1)

    def test_missing_pose_does_not_advance_the_machine(self) -> None:
        detector = FallDetector()
        run(detector, [(0.0, 0.45)] * 10)
        state, features = detector.update(0.5, np.zeros((33, 4), dtype=np.float32))
        self.assertEqual(state, State.STANDING)
        self.assertTrue(math.isnan(features["torso_angle"]))

    def test_bad_landmark_shape_raises(self) -> None:
        detector = FallDetector()
        with self.assertRaises(ValueError):
            detector.update(0.0, np.zeros((10, 4), dtype=np.float32))

    def test_params_are_configurable(self) -> None:
        # A stricter hold makes the same fast fall alert later, not sooner.
        loose = FallDetector(Params(stillness_s=2.0))
        strict = FallDetector(Params(stillness_s=4.0))
        poses = fast_fall() + [(85.0, 0.85)] * 200
        run(loose, poses)
        run(strict, poses)
        self.assertLess(loose.alert_time, strict.alert_time)


if __name__ == "__main__":
    unittest.main()
