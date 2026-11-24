"""Minimal IoU tracker that keeps a stable ID for each person."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    centroid: np.ndarray
    misses: int = 0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _centroid(bbox: np.ndarray) -> np.ndarray:
    return np.array(
        [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
        dtype=np.float32,
    )


def _diagonal(bbox: np.ndarray) -> float:
    return float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))


class CentroidTracker:
    """Greedy IoU matching plus a distance fallback for fast motion."""

    def __init__(
        self,
        iou_threshold: float = 0.2,
        max_missed: int = 15,
        smoothing: float = 0.6,
        max_distance_factor: float = 0.8,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.smoothing = smoothing
        self.max_distance_factor = max_distance_factor
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, bboxes: list[np.ndarray]) -> list[Track]:
        """Return one track per input bbox, in the same order."""
        detections = [np.asarray(b, dtype=np.float32) for b in bboxes]
        if not self._tracks:
            return [self._create(b) for b in detections]

        track_list = list(self._tracks.values())
        ious = np.zeros((len(detections), len(track_list)), dtype=np.float32)
        for i, det in enumerate(detections):
            for j, track in enumerate(track_list):
                ious[i, j] = _iou(det, track.bbox)

        pairs = sorted(
            (
                (float(ious[i, j]), i, j)
                for i in range(len(detections))
                for j in range(len(track_list))
            ),
            reverse=True,
        )
        det_match: dict[int, int] = {}
        track_match: set[int] = set()
        for score, i, j in pairs:
            if score < self.iou_threshold:
                break
            if i in det_match or j in track_match:
                continue
            det_match[i] = j
            track_match.add(j)

        # Fallback: match leftover detections by centroid distance. This keeps
        # the same ID when a fast fall moves the box too much for IoU.
        leftover = [
            (i, j)
            for i in range(len(detections))
            if i not in det_match
            for j in range(len(track_list))
            if j not in track_match
        ]
        distance_candidates = []
        for i, j in leftover:
            distance = float(
                np.linalg.norm(_centroid(detections[i]) - track_list[j].centroid)
            )
            limit = self.max_distance_factor * max(
                _diagonal(detections[i]), _diagonal(track_list[j].bbox)
            )
            if limit > 0 and distance <= limit:
                distance_candidates.append((distance / limit, i, j))
        for _, i, j in sorted(distance_candidates):
            if i in det_match or j in track_match:
                continue
            det_match[i] = j
            track_match.add(j)

        for j, track in enumerate(track_list):
            if j in track_match:
                track.misses = 0
            else:
                track.misses += 1
                if track.misses > self.max_missed:
                    del self._tracks[track.track_id]

        output: list[Track] = []
        for i, det in enumerate(detections):
            j = det_match.get(i)
            if j is None:
                output.append(self._create(det))
            else:
                track = track_list[j]
                track.bbox = self.smoothing * det + (1.0 - self.smoothing) * track.bbox
                track.centroid = _centroid(track.bbox)
                output.append(track)
        return output

    def _create(self, bbox: np.ndarray) -> Track:
        track = Track(self._next_id, bbox.copy(), _centroid(bbox))
        self._tracks[self._next_id] = track
        self._next_id += 1
        return track

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
