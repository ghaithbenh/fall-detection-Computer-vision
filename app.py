"""Streamlit demo: fall detection from an uploaded video or the webcam.

Every processed frame goes through MediaPipe PoseLandmarker and the
landmark-only state machine in :mod:`src.fall_logic`, so the app shows the
live skeleton, the current state and a red banner on ALERT.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.annotate import draw_skeleton
from src.config import (
    DEFAULT_FPS,
    DEFAULT_POSE_MODEL,
    FRAME_HEIGHT,
    MAX_FRAME_WIDTH,
)
from src.fall_logic import FallDetector, Params, State
from src.pipeline import target_size
from src.pose_estimator import PoseEstimator

st.set_page_config(page_title="Fall Detection", page_icon="🩺", layout="wide")

STATE_COLORS = {
    State.STANDING: (70, 200, 70),
    State.FALLING: (0, 180, 255),
    State.LYING: (0, 140, 255),
    State.ALERT: (0, 0, 255),
}
ROW_COLUMNS = ("t", "torso_angle", "hip_drop_speed", "state")
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class RunResult:
    """Everything one run produced, kept for the results section."""

    rows: pd.DataFrame
    events: pd.DataFrame
    frames: int
    duration_s: float
    last_state: State
    lying_angle: float
    drop_speed: float
    signature: tuple


def _draw(
    frame: np.ndarray,
    landmarks: np.ndarray | None,
    state: State,
    t: float,
) -> np.ndarray:
    """Skeleton plus a state label, red border while ALERT."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    color = STATE_COLORS[state]

    if landmarks is not None:
        draw_skeleton(canvas, landmarks, color)

    text = f"{state.value.upper()}  t={t:.1f}s"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    cv2.rectangle(canvas, (6, 6), (tw + 18, th + 16), (0, 0, 0), -1)
    cv2.putText(
        canvas, text, (12, th + 9), FONT, 0.6, color, 2, cv2.LINE_AA
    )

    if state is State.ALERT:
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (0, 0, 255), 6)
    return canvas


def _row(t: float, features: dict[str, float], state: State) -> dict:
    return {
        "t": t,
        "torso_angle": features.get("torso_angle", float("nan")),
        "hip_drop_speed": features.get("hip_drop_speed", 0.0),
        "state": state.value,
    }


def _event(t: float, features: dict[str, float]) -> dict:
    return {
        "time_s": round(t, 2),
        "torso_angle_deg": round(features.get("torso_angle", float("nan")), 1),
        "hip_drop_speed": round(features.get("hip_drop_speed", 0.0), 2),
    }


def _result(
    rows: list[dict],
    events: list[dict],
    frames: int,
    duration_s: float,
    last_state: State,
    params: Params,
    signature: tuple,
) -> RunResult:
    return RunResult(
        rows=pd.DataFrame(rows, columns=ROW_COLUMNS),
        events=pd.DataFrame(
            events, columns=("time_s", "torso_angle_deg", "hip_drop_speed")
        ),
        frames=frames,
        duration_s=duration_s,
        last_state=last_state,
        lying_angle=params.lying_angle,
        drop_speed=params.drop_speed,
        signature=signature,
    )


def _run_video(
    source: Path,
    params: Params,
    stride: int,
    frame_ph,
    alert_ph,
    progress,
    signature: tuple,
) -> RunResult:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source.name}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not fps > 0:
        fps = DEFAULT_FPS
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    expected = max(1, total // stride) if total > 0 else 0

    detector = FallDetector(params)
    estimator = PoseEstimator(DEFAULT_POSE_MODEL)
    rows: list[dict] = []
    events: list[dict] = []
    frames = 0
    last_draw = 0.0
    last: tuple[np.ndarray, np.ndarray | None, State, float] | None = None

    try:
        frame_idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            t = frame_idx / fps
            frame = cv2.resize(
                frame, params.frame_size, interpolation=cv2.INTER_AREA
            )
            poses = estimator.detect(frame, int(round(t * 1000)))
            landmarks = poses[0].landmarks if poses else None
            if landmarks is not None:
                state, features = detector.update(t, landmarks)
            else:
                state, features = detector.state, {}

            rows.append(_row(t, features, state))
            if state is State.ALERT and detector.alert_time == t:
                events.append(_event(t, features))
            frames += 1
            last = (frame, landmarks, state, t)

            now = time.perf_counter()
            if now - last_draw >= 0.05:
                last_draw = now
                annotated = _draw(frame, landmarks, state, t)
                frame_ph.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    width="stretch",
                )
                if state is State.ALERT:
                    alert_ph.error("🚨 ALERT — fall detected")
                else:
                    alert_ph.empty()
                if expected:
                    progress.progress(
                        min(1.0, frames / expected),
                        text=f"Frame {frames} / ~{expected}",
                    )
    finally:
        capture.release()
        estimator.close()

    if last is not None:
        annotated = _draw(*last)
        frame_ph.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch"
        )
    duration = frames * stride / fps if fps > 0 else 0.0
    return _result(
        rows,
        events,
        frames,
        duration,
        detector.state,
        params,
        signature,
    )


def _run_webcam(
    index: int,
    max_seconds: float,
    params: Params,
    stride: int,
    frame_ph,
    alert_ph,
) -> RunResult:
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open webcam {index}")

    width, height = target_size(
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        FRAME_HEIGHT,
        MAX_FRAME_WIDTH,
    )
    params = replace(params, frame_size=(width, height))
    detector = FallDetector(params)
    estimator = PoseEstimator(DEFAULT_POSE_MODEL)
    rows: list[dict] = []
    events: list[dict] = []
    frames = 0
    last_draw = 0.0
    started = time.perf_counter()
    state = State.STANDING

    try:
        frame_idx = 0
        while time.perf_counter() - started < max_seconds:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            t = time.perf_counter() - started
            frame = cv2.resize(
                frame, params.frame_size, interpolation=cv2.INTER_AREA
            )
            poses = estimator.detect(frame, int(round(t * 1000)))
            landmarks = poses[0].landmarks if poses else None
            if landmarks is not None:
                state, features = detector.update(t, landmarks)
            else:
                state, features = detector.state, {}

            rows.append(_row(t, features, state))
            if state is State.ALERT and detector.alert_time == t:
                events.append(_event(t, features))
            frames += 1

            now = time.perf_counter()
            if now - last_draw >= 0.05:
                last_draw = now
                annotated = _draw(frame, landmarks, state, t)
                frame_ph.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    width="stretch",
                )
                if state is State.ALERT:
                    alert_ph.error("🚨 ALERT — fall detected")
                else:
                    alert_ph.empty()
    finally:
        capture.release()
        estimator.close()

    duration = time.perf_counter() - started
    return _result(
        rows,
        events,
        frames,
        duration,
        detector.state,
        params,
        ("webcam", index, params, stride, max_seconds),
    )


def _display(result: RunResult) -> None:
    st.subheader("Results")
    cols = st.columns(4)
    cols[0].metric("Processed frames", f"{result.frames}")
    cols[1].metric("Duration", f"{result.duration_s:.1f} s")
    cols[2].metric("Fall events", f"{len(result.events)}")
    cols[3].metric("Final state", result.last_state.value.upper())

    if result.last_state is State.ALERT:
        st.error("🚨 ALERT — a person is down")
    elif not result.events.empty:
        st.warning("A fall was detected earlier in the clip.")
    else:
        st.success("No fall detected.")

    st.markdown("### Event log")
    if result.events.empty:
        st.info("No events.")
    else:
        st.dataframe(result.events, hide_index=True, width="stretch")

    st.markdown("### Signals")
    chart = result.rows.set_index("t")
    left, right = st.columns(2)
    with left:
        st.caption("Torso angle (0° upright, 90° horizontal)")
        angle = chart[["torso_angle"]].copy()
        angle["lying threshold"] = result.lying_angle
        st.line_chart(angle)
    with right:
        st.caption("Hip drop speed (torso lengths / second)")
        speed = chart[["hip_drop_speed"]].copy()
        speed["fall threshold"] = result.drop_speed
        st.line_chart(speed)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Thresholds")
    if not DEFAULT_POSE_MODEL.is_file():
        st.error("Pose model missing.")
        st.code("python -m src.download_model", language="powershell")
        st.stop()

    lying_angle = st.slider("Lying angle (°)", 40, 80, 55)
    drop_speed = st.slider("Hip drop speed (torso/s)", 0.5, 5.0, 2.0, 0.1)
    fall_window = st.slider("Fall window (s)", 0.5, 3.0, 1.5, 0.1)
    stillness = st.slider("Stillness before alert (s)", 1.0, 5.0, 2.0, 0.5)
    recover = st.slider("Recovery time (s)", 0.1, 1.0, 0.3, 0.1)
    stride = st.slider("Process every N frames", 1, 5, 2)
    webcam_seconds = st.slider("Webcam duration (s)", 5, 60, 20, 5)

    st.divider()
    st.caption(
        "A fall starts when the hips drop faster than the threshold and the "
        "torso becomes horizontal within the fall window. The alert fires "
        "after the person stays lying still."
    )

# ------------------------------------------------------------------- main
st.title("Fall Detection")
st.write(
    "Upload a clip or use the webcam. Frames are processed live with "
    "MediaPipe pose landmarks and a rule-based state machine."
)

source = st.radio("Source", ["Upload video", "Webcam"], horizontal=True)
if st.session_state.get("source") != source:
    st.session_state.pop("result", None)
    st.session_state["source"] = source

base_params = Params(
    lying_angle=lying_angle,
    drop_speed=drop_speed,
    fall_window_s=fall_window,
    stillness_s=stillness,
    recover_s=recover,
)

if source == "Upload video":
    uploaded = st.file_uploader(
        "Video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Short clips (10–60 s) process fastest on a laptop CPU.",
    )
    if uploaded is None:
        st.session_state.pop("result", None)
        st.info("Upload a clip to begin. A 480p video of 10–30 s is ideal.")
    else:
        payload = uploaded.getvalue()
        file_hash = hashlib.md5(payload).hexdigest()[:10]
        signature = (file_hash, base_params, stride)

        if st.button("Run detection", type="primary"):
            work_dir = Path(tempfile.mkdtemp(prefix="falldet_in_"))
            suffix = Path(uploaded.name).suffix or ".mp4"
            input_path = work_dir / f"input{suffix}"
            input_path.write_bytes(payload)

            capture = cv2.VideoCapture(str(input_path))
            ok = capture.isOpened()
            width, height = 0, 0
            if ok:
                width, height = target_size(
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                    FRAME_HEIGHT,
                    MAX_FRAME_WIDTH,
                )
            capture.release()
            if not ok:
                st.error("Could not open the uploaded file as a video.")
                st.stop()

            params = replace(base_params, frame_size=(width, height))
            frame_ph = st.empty()
            alert_ph = st.empty()
            progress = st.progress(0.0, text="Starting pose estimation...")
            try:
                with st.spinner("Processing frames..."):
                    result = _run_video(
                        input_path,
                        params,
                        stride,
                        frame_ph,
                        alert_ph,
                        progress,
                        signature,
                    )
            except Exception as exc:  # noqa: BLE001 - show any runtime error
                progress.empty()
                st.error(f"Processing failed: {exc}")
                st.stop()
            progress.progress(1.0, text="Done")
            alert_ph.empty()
            st.session_state["result"] = result

        result = st.session_state.get("result")
        if result is not None:
            if result.signature == signature:
                _display(result)
            else:
                st.info(
                    "Video or settings changed. Click **Run detection** to "
                    "refresh the result."
                )
else:
    index = st.number_input("Camera index", 0, 8, 0, 1)
    if st.button("Start webcam", type="primary"):
        # The loop is blocking, so the run ends by itself after this duration.
        st.caption(
            "Keep the person in frame. The run stops automatically; there is "
            "no stop button because the frame loop blocks the page."
        )
        frame_ph = st.empty()
        alert_ph = st.empty()
        try:
            with st.spinner("Reading from the camera..."):
                result = _run_webcam(
                    int(index),
                    float(webcam_seconds),
                    base_params,
                    stride,
                    frame_ph,
                    alert_ph,
                )
        except Exception as exc:  # noqa: BLE001 - show any runtime error
            st.error(f"Webcam failed: {exc}")
            st.stop()
        alert_ph.empty()
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if result is not None:
        _display(result)
    else:
        st.info("Click **Start webcam** to begin.")
