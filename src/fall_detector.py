"""Temporal fall state machine (rules, with optional ML confirmation)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping

from .config import DetectorConfig


@dataclass
class FallEvent:
    track_id: int
    frame_idx: int
    timestamp_s: float
    confidence: float
    torso_angle: float
    drop_speed: float
    source: str = "rules"


@dataclass
class TrackState:
    history: deque = field(default_factory=deque)
    posture: str = "unknown"
    last_upright_t: float | None = None
    lying_since: float | None = None
    upright_since: float | None = None
    fallen: bool = False
    last_fall_t: float = -1e9


class FallDetector:
    """Flags a fall when an upright person becomes horizontal quickly."""

    def __init__(self, cfg: DetectorConfig | None = None) -> None:
        self.cfg = cfg or DetectorConfig()
        self._states: dict[int, TrackState] = {}

    def update(
        self,
        track_id: int,
        features: Mapping[str, float],
        timestamp_s: float,
        frame_idx: int,
        ml_prob: float | None = None,
    ) -> tuple[str, FallEvent | None]:
        state = self._states.setdefault(track_id, TrackState())
        state.history.append((timestamp_s, features))
        while (
            state.history
            and timestamp_s - state.history[0][0] > self.cfg.history_s
        ):
            state.history.popleft()

        posture = self._posture(features)
        state.posture = posture
        event: FallEvent | None = None

        if posture == "upright":
            state.last_upright_t = timestamp_s
            state.lying_since = None
            if state.upright_since is None:
                state.upright_since = timestamp_s
            if (
                state.fallen
                and timestamp_s - state.upright_since >= self.cfg.recover_upright_s
            ):
                state.fallen = False
        else:
            state.upright_since = None
            if posture == "lying":
                if state.lying_since is None:
                    state.lying_since = timestamp_s
                event = self._maybe_fall(
                    track_id, state, features, timestamp_s, frame_idx, ml_prob
                )
        return posture, event

    def _posture(self, features: Mapping[str, float]) -> str:
        angle = features.get("torso_angle", math.nan)
        torso_len = features.get("torso_len", 0.0)
        if not math.isfinite(angle) or not math.isfinite(torso_len) or torso_len < 0.05:
            return "unknown"
        if angle >= self.cfg.lying_angle:
            return "lying"
        if angle <= self.cfg.upright_angle:
            return "upright"
        return "transition"

    def _maybe_fall(
        self,
        track_id: int,
        state: TrackState,
        features: Mapping[str, float],
        timestamp_s: float,
        frame_idx: int,
        ml_prob: float | None,
    ) -> FallEvent | None:
        cfg = self.cfg
        if state.fallen or state.last_upright_t is None:
            return None
        if timestamp_s - state.last_upright_t > cfg.fall_window_s:
            return None
        if (
            state.lying_since is None
            or timestamp_s - state.lying_since < cfg.confirm_lying_s
        ):
            return None
        if timestamp_s - state.last_fall_t < cfg.cooldown_s:
            return None

        recent = [
            f for t, f in state.history if t >= state.last_upright_t  # type: ignore[operator]
        ]
        max_drop = max(
            (float(f.get("hip_drop_speed", 0.0)) for f in recent),
            default=0.0,
        )
        drop_ref = max(cfg.drop_speed, 1e-6)
        fast = max_drop >= drop_ref
        ml_ok = ml_prob is not None and ml_prob >= cfg.ml_threshold
        if not (fast or ml_ok):
            return None

        angle = float(features.get("torso_angle", 0.0))
        confidence = 0.5 + 0.25 * min(1.0, max_drop / (2.0 * drop_ref))
        confidence += 0.25 * min(1.0, max(0.0, angle) / 90.0)
        if ml_prob is not None:
            confidence = 0.75 * confidence + 0.25 * float(ml_prob)
        confidence = max(0.5, min(0.99, confidence))
        source = "rules+ml" if fast and ml_ok else ("ml" if ml_ok else "rules")

        state.fallen = True
        state.last_fall_t = timestamp_s
        return FallEvent(
            track_id=track_id,
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            confidence=confidence,
            torso_angle=angle,
            drop_speed=max_drop,
            source=source,
        )

    def is_fallen(self, track_id: int) -> bool:
        state = self._states.get(track_id)
        return bool(state and state.fallen)

    def reset(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._states.clear()
        else:
            self._states.pop(track_id, None)
