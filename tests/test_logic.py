"""Unit tests for the geometry, tracking and fall logic (no video needed)."""

from __future__ import annotations

import unittest

import numpy as np

from src.config import DetectorConfig
from src.fall_detector import FallDetector
from src.features import extract_features
from src.pipeline import target_size
from src.tracker import CentroidTracker


def make_landmarks(
    shoulder_y: float = 0.30,
    hip_y: float = 0.60,
    shoulder_x: float = 0.50,
    hip_x: float = 0.50,
) -> np.ndarray:
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = 0.9
    lm[0] = [shoulder_x, shoulder_y - 0.1, 0.0, 0.99]
    lm[11] = [shoulder_x - 0.05, shoulder_y, 0.0, 0.99]
    lm[12] = [shoulder_x + 0.05, shoulder_y, 0.0, 0.99]
    lm[23] = [hip_x - 0.05, hip_y, 0.0, 0.99]
    lm[24] = [hip_x + 0.05, hip_y, 0.0, 0.99]
    lm[25] = [hip_x - 0.05, hip_y + 0.15, 0.0, 0.99]
    lm[26] = [hip_x + 0.05, hip_y + 0.15, 0.0, 0.99]
    lm[27] = [hip_x - 0.05, hip_y + 0.30, 0.0, 0.99]
    lm[28] = [hip_x + 0.05, hip_y + 0.30, 0.0, 0.99]
    return lm


def frame_features(angle: float, drop: float = 0.0) -> dict[str, float]:
    return {
        "torso_angle": angle,
        "torso_len": 0.30,
        "bbox_aspect": 0.6,
        "bbox_height": 0.6,
        "hip_y": 0.6,
        "shoulder_y": 0.3,
        "head_y": 0.2,
        "knee_angle": 175.0,
        "hip_drop_speed": drop,
        "center_speed": drop,
    }


class FeatureTests(unittest.TestCase):
    def test_upright_torso_angle(self) -> None:
        lm = make_landmarks()
        features = extract_features(
            lm, np.array([100, 100, 500, 400]), 640, 480
        )
        self.assertLess(features["torso_angle"], 5.0)
        self.assertAlmostEqual(features["torso_len"], 0.30, places=2)

    def test_horizontal_torso_angle(self) -> None:
        lm = make_landmarks()
        lm[11] = [0.40, 0.50, 0.0, 0.99]
        lm[12] = [0.40, 0.55, 0.0, 0.99]
        lm[23] = [0.60, 0.50, 0.0, 0.99]
        lm[24] = [0.60, 0.55, 0.0, 0.99]
        features = extract_features(
            lm, np.array([100, 100, 500, 400]), 640, 480
        )
        self.assertGreater(features["torso_angle"], 85.0)

    def test_hip_drop_speed(self) -> None:
        lm = make_landmarks(shoulder_y=0.30, hip_y=0.60)
        prev = {"hip_y": 0.40, "cx": 300.0, "cy": 250.0}
        features = extract_features(
            lm,
            np.array([100, 100, 500, 400]),
            640,
            480,
            prev=prev,
            dt=0.1,
        )
        self.assertGreater(features["hip_drop_speed"], 5.0)


class TrackerTests(unittest.TestCase):
    def test_id_is_stable(self) -> None:
        tracker = CentroidTracker()
        first = tracker.update([np.array([10, 10, 100, 100])])[0]
        second = tracker.update([np.array([12, 12, 102, 102])])[0]
        self.assertEqual(first.track_id, second.track_id)

    def test_new_person_gets_new_id(self) -> None:
        tracker = CentroidTracker()
        first = tracker.update([np.array([10, 10, 100, 100])])[0]
        tracks = tracker.update(
            [
                np.array([12, 12, 102, 102]),
                np.array([300, 300, 400, 400]),
            ]
        )
        ids = {track.track_id for track in tracks}
        self.assertIn(first.track_id, ids)
        self.assertEqual(len(ids), 2)

    def test_fast_motion_keeps_id(self) -> None:
        tracker = CentroidTracker()
        first = tracker.update([np.array([10, 10, 100, 100])])[0]
        # IoU is tiny here, but the centroid distance is still plausible.
        second = tracker.update([np.array([90, 10, 180, 100])])[0]
        self.assertEqual(first.track_id, second.track_id)


class FallDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = DetectorConfig(
            confirm_lying_s=0.35,
            fall_window_s=2.0,
            cooldown_s=4.0,
            drop_speed=2.0,
        )

    def _upright(self, detector: FallDetector, start: float, frames: int) -> None:
        for i in range(frames):
            t = start + i * 0.04
            detector.update(1, frame_features(10.0), t, int(t * 25))

    def test_fast_fall_triggers_once(self) -> None:
        detector = FallDetector(self.cfg)
        self._upright(detector, 0.0, 10)
        events = []
        for i in range(10):
            t = 0.40 + i * 0.04
            _, event = detector.update(1, frame_features(70.0, 5.0), t, int(t * 25))
            if event:
                events.append(event)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source, "rules")
        self.assertGreater(events[0].confidence, 0.5)

        # Staying on the floor must not create duplicate alerts.
        for i in range(30):
            t = 0.80 + i * 0.04
            _, event = detector.update(1, frame_features(70.0, 0.0), t, int(t * 25))
            self.assertIsNone(event)

    def test_slow_lie_does_not_trigger(self) -> None:
        detector = FallDetector(self.cfg)
        self._upright(detector, 0.0, 10)
        for i in range(60):
            t = 0.40 + i * 0.04
            _, event = detector.update(1, frame_features(70.0, 0.2), t, int(t * 25))
            self.assertIsNone(event)

    def test_ml_confirmation_triggers(self) -> None:
        detector = FallDetector(self.cfg)
        self._upright(detector, 0.0, 10)
        events = []
        for i in range(10):
            t = 0.40 + i * 0.04
            _, event = detector.update(
                1, frame_features(70.0, 0.2), t, int(t * 25), ml_prob=0.9
            )
            if event:
                events.append(event)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source, "ml")

    def test_recovery_allows_second_fall(self) -> None:
        detector = FallDetector(self.cfg)
        self._upright(detector, 0.0, 10)
        first = None
        for i in range(10):
            t = 0.40 + i * 0.04
            _, event = detector.update(1, frame_features(70.0, 5.0), t, int(t * 25))
            first = first or event
        self.assertIsNotNone(first)

        # Stand up for more than the 4 s cooldown.
        self._upright(detector, 2.0, 100)
        second = None
        for i in range(10):
            t = 6.0 + i * 0.04
            _, event = detector.update(1, frame_features(70.0, 5.0), t, int(t * 25))
            second = second or event
        self.assertIsNotNone(second)
        self.assertNotEqual(first.timestamp_s, second.timestamp_s)


class ResizeTests(unittest.TestCase):
    def test_1080p_becomes_480p(self) -> None:
        width, height = target_size(1920, 1080, 480, 854)
        self.assertEqual(height, 480)
        self.assertLessEqual(width, 854)
        self.assertEqual(width % 2, 0)

    def test_640x480_is_unchanged(self) -> None:
        self.assertEqual(target_size(640, 480, 480, 854), (640, 480))


if __name__ == "__main__":
    unittest.main()
