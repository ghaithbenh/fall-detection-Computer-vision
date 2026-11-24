"""Threshold baseline: a fall is a horizontal torso held for a while.

This is the reference model the temporal state machine in
:mod:`src.fall_detector` is compared against. It needs nothing but the
torso angle per sample, which makes it easy to reason about but blind to
how fast the person went down.
"""

from __future__ import annotations

import math
from typing import Sequence

# Angles come from src.features (0 deg = upright, 90 deg = horizontal).
DEFAULT_ANGLE_THR = 60.0
DEFAULT_HOLD_S = 1.0

# Frame times computed as i / fps rarely subtract to exactly 1.0 s
# (e.g. 35/25 - 10/25 = 0.9999999999999999), so compare with a small slack.
_EPS = 1e-9


def baseline_detect(
    times: Sequence[float],
    angles: Sequence[float],
    angle_thr: float = DEFAULT_ANGLE_THR,
    hold: float = DEFAULT_HOLD_S,
) -> float | None:
    """Detect a fall as a torso angle above ``angle_thr`` for ``hold`` seconds.

    The run starts at the first sample with ``torso_angle >= angle_thr``.
    Any sample below the threshold, or a non-finite angle (pose not
    visible), ends the run. Timestamps are used as-is, so missing frames are
    harmless: a run is only reported once the observed samples span ``hold``
    seconds.

    Args:
        times: Sample timestamps in seconds, non-decreasing.
        angles: Torso angle in degrees, same length as ``times``.
        angle_thr: Smallest angle treated as lying down, in degrees.
        hold: Seconds the angle must stay at or above ``angle_thr``.

    Returns:
        Timestamp of the first sample that completes the hold window, or
        ``None`` if the person never stayed horizontal that long.

    Raises:
        ValueError: If the sequences differ in length, ``times`` is not
            non-decreasing, or ``hold`` is negative.
    """
    times = [float(t) for t in times]
    angles = [float(a) for a in angles]
    if len(times) != len(angles):
        raise ValueError(
            "times and angles must have the same length "
            f"({len(times)} != {len(angles)})"
        )
    if hold < 0:
        raise ValueError(f"hold must be non-negative, got {hold}")

    run_start: float | None = None
    prev_t: float | None = None
    for t, angle in zip(times, angles):
        if not math.isfinite(t):
            raise ValueError(f"times must be finite, got {t}")
        if prev_t is not None and t < prev_t:
            raise ValueError("times must be non-decreasing")
        prev_t = t

        if math.isfinite(angle) and angle >= angle_thr:
            if run_start is None:
                run_start = t
            # Compare against the start of the run, not the previous frame:
            # sparse frames still measure the full duration above threshold.
            if t - run_start + _EPS >= hold:
                return t
        else:
            run_start = None
    return None
