"""Streamlit demo for fall detection and tracking."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import asdict, fields
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_CLASSIFIER,
    DEFAULT_POSE_MODEL,
    DetectorConfig,
    PipelineConfig,
)
from src.fall_detector import FallEvent
from src.model import FallClassifier
from src.pipeline import FEATURE_ROW_COLUMNS, FallPipeline, VideoReport

st.set_page_config(
    page_title="Fall Detection & Tracking",
    page_icon="🩺",
    layout="wide",
)

EVENT_COLUMNS = [field.name for field in fields(FallEvent)]


def _key_frames(
    video_path: Path,
    events: list[FallEvent],
    max_events: int = 3,
) -> list[tuple[str, np.ndarray]]:
    """Grab a before/at/after frame for each fall event."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    frames: list[tuple[str, np.ndarray]] = []
    for event in events[:max_events]:
        for offset, label in ((-0.6, "Before"), (0.0, "Fall"), (1.0, "After")):
            timestamp = max(0.0, event.timestamp_s + offset)
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if ok:
                frames.append(
                    (
                        f"{label} (t={timestamp:.1f}s)",
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    )
                )
    capture.release()
    return frames


def _run_pipeline(
    input_path: Path,
    cfg: PipelineConfig,
    classifier: FallClassifier | None,
    progress_bar,
    signature: tuple,
) -> dict:
    work_dir = Path(tempfile.mkdtemp(prefix="falldet_out_"))
    video_path = work_dir / "annotated.mp4"

    def progress(done: int, total: int) -> None:
        if total > 0:
            progress_bar.progress(
                min(1.0, done / total),
                text=f"Processing frame {done}/{total}",
            )
        else:
            progress_bar.progress(0.0, text=f"Processing {done} frames")

    pipeline = FallPipeline(cfg, classifier)
    try:
        report: VideoReport = pipeline.process_video(
            input_path, video_path, progress_cb=progress
        )
    finally:
        pipeline.close()

    rows = pd.DataFrame(report.rows, columns=FEATURE_ROW_COLUMNS)
    events = pd.DataFrame(
        [asdict(event) for event in report.events], columns=EVENT_COLUMNS
    )
    features_csv = work_dir / "features.csv"
    events_csv = work_dir / "events.csv"
    rows.to_csv(features_csv, index=False)
    events.to_csv(events_csv, index=False)

    return {
        "report": report,
        "rows": rows,
        "events": events,
        "video": video_path,
        "features_csv": features_csv,
        "events_csv": events_csv,
        "key_frames": _key_frames(video_path, report.events),
        "signature": signature,
        "lying_angle": cfg.detector.lying_angle,
        "drop_speed": cfg.detector.drop_speed,
    }


def _display(result: dict) -> None:
    report: VideoReport = result["report"]
    events: pd.DataFrame = result["events"]
    rows: pd.DataFrame = result["rows"]

    st.subheader("Results")
    cols = st.columns(4)
    cols[0].metric("Duration", f"{report.duration_s:.1f} s")
    cols[1].metric("Processed frames", f"{report.frames_processed}")
    cols[2].metric("Processing speed", f"{report.processing_fps:.1f} FPS")
    cols[3].metric("Fall events", f"{len(report.events)}")

    if report.events:
        st.error(f"⚠️ {len(report.events)} fall event(s) detected")
    else:
        st.success("No fall detected in this clip.")

    video_col, download_col = st.columns([2, 1])
    with video_col:
        st.video(str(result["video"]))
        st.caption(
            "If the browser cannot preview the codec, use the download button."
        )
    with download_col:
        st.download_button(
            "⬇️ Annotated video (MP4)",
            data=result["video"].read_bytes(),
            file_name="annotated.mp4",
            mime="video/mp4",
            width="stretch",
        )
        st.download_button(
            "⬇️ Events (CSV)",
            data=result["events_csv"].read_bytes(),
            file_name="fall_events.csv",
            mime="text/csv",
            width="stretch",
        )
        st.download_button(
            "⬇️ Features (CSV)",
            data=result["features_csv"].read_bytes(),
            file_name="features.csv",
            mime="text/csv",
            width="stretch",
        )

    if not events.empty:
        st.markdown("### Fall events")
        table = events.copy()
        table["timestamp_s"] = table["timestamp_s"].map(lambda v: f"{v:.2f} s")
        table["confidence"] = table["confidence"].map(lambda v: f"{v:.2f}")
        table["torso_angle"] = table["torso_angle"].map(lambda v: f"{v:.0f}°")
        table["drop_speed"] = table["drop_speed"].map(lambda v: f"{v:.2f}")
        st.dataframe(table, hide_index=True, width="stretch")

        if result["key_frames"]:
            st.markdown("### Key frames")
            frames = result["key_frames"]
            for start in range(0, len(frames), 3):
                image_cols = st.columns(3)
                for col, (caption, image) in zip(
                    image_cols, frames[start : start + 3]
                ):
                    col.image(image, caption=caption, width="stretch")

    if not rows.empty and "timestamp_s" in rows.columns:
        st.markdown("### Signals")
        chart_data = rows.dropna(subset=["timestamp_s"]).set_index("timestamp_s")
        left, right = st.columns(2)
        with left:
            st.caption("Torso angle (0° upright, 90° horizontal)")
            angle = chart_data[["torso_angle"]].copy()
            angle["lying threshold"] = result["lying_angle"]
            st.line_chart(angle)
        with right:
            st.caption("Hip drop speed (torso lengths / second)")
            speed = chart_data[["hip_drop_speed"]].copy()
            speed["fall threshold"] = result["drop_speed"]
            st.line_chart(speed)
        if "ml_prob" in chart_data.columns and chart_data["ml_prob"].notna().any():
            st.caption("RandomForest fall probability")
            st.line_chart(chart_data[["ml_prob"]])

    with st.expander("How the detector works"):
        st.markdown(
            """
1. Each frame is resized to 480p and passed to **MediaPipe PoseLandmarker (lite)**
   to get 33 body keypoints.
2. A lightweight IoU tracker keeps a stable ID and bounding box per person.
3. Geometry features are computed: torso angle, bounding-box aspect, hip drop
   speed, knee angle, etc.
4. A state machine raises a fall when an upright person becomes horizontal
   within 2 s **and** the hips drop fast (≥ 2 torso lengths/s).
5. An optional RandomForest, trained on 15-frame windows, can confirm the rule.
            """
        )


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    if DEFAULT_POSE_MODEL.is_file():
        st.caption("Pose model: MediaPipe lite (5.7 MB, CPU)")
    else:
        st.error("Pose model missing.")
        st.code("python -m src.download_model", language="powershell")
        st.stop()

    ml_available = DEFAULT_CLASSIFIER.is_file()
    use_ml = st.toggle(
        "RandomForest confirmation",
        value=False,
        disabled=not ml_available,
        help="Requires a model trained with python -m src.train",
    )
    if not ml_available:
        st.caption("Train a model to enable ML: `python -m src.train`")

    lying_angle = st.slider("Lying angle (degrees)", 45, 75, 55)
    drop_speed = st.slider(
        "Hip drop speed (torso/s)", 1.0, 4.0, 2.0, 0.1
    )
    confirm = st.slider("Lying confirmation (s)", 0.2, 1.0, 0.35, 0.05)
    stride = st.slider("Process every N frames", 1, 3, 1)
    num_poses = st.selectbox("Maximum people", [1, 2], index=0)
    st.divider()
    st.caption(
        "Fall = upright → horizontal within 2 s with a fast hip drop. "
        "Lower the thresholds for more sensitive detection."
    )

# ------------------------------------------------------------------- main
st.title("Fall Detection & Tracking")
st.write(
    "Upload a video to detect and track falls with MediaPipe PoseLandmarker "
    "(lite model) and a temporal rule-based detector."
)

uploaded = st.file_uploader(
    "Video file",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    help="Short clips (10–60 s) process fastest on a laptop CPU.",
)

if uploaded is not None:
    payload = uploaded.getvalue()
    file_hash = hashlib.md5(payload).hexdigest()[:10]
    signature = (
        file_hash,
        use_ml,
        lying_angle,
        drop_speed,
        confirm,
        stride,
        num_poses,
    )

    run = st.button("Run detection", type="primary")
    if run:
        work_dir = Path(tempfile.mkdtemp(prefix="falldet_in_"))
        suffix = Path(uploaded.name).suffix or ".mp4"
        input_path = work_dir / f"input{suffix}"
        input_path.write_bytes(payload)

        detector_cfg = DetectorConfig(
            lying_angle=lying_angle,
            drop_speed=drop_speed,
            confirm_lying_s=confirm,
        )
        cfg = PipelineConfig(
            model_path=DEFAULT_POSE_MODEL,
            num_poses=num_poses,
            stride=stride,
            detector=detector_cfg,
        )
        classifier = None
        if use_ml:
            try:
                classifier = FallClassifier.load(DEFAULT_CLASSIFIER)
            except Exception as exc:  # noqa: BLE001 - show any load error
                st.error(f"Could not load classifier: {exc}")
                st.stop()

        progress_bar = st.progress(0.0, text="Starting pose estimation...")
        try:
            with st.spinner("Detecting poses and falls..."):
                result = _run_pipeline(
                    input_path, cfg, classifier, progress_bar, signature
                )
        except Exception as exc:  # noqa: BLE001 - display pipeline errors
            progress_bar.empty()
            st.error(f"Processing failed: {exc}")
            st.stop()
        progress_bar.progress(1.0, text="Done")
        st.session_state["result"] = result
        st.session_state["signature"] = signature

    result = st.session_state.get("result")
    if result is not None:
        if result.get("signature") == signature:
            _display(result)
        else:
            st.info(
                "Video or settings changed. Click **Run detection** to refresh "
                "the result."
            )
else:
    st.session_state.pop("result", None)
    st.info("Upload a clip to begin. A 480p video of 10–30 s is ideal.")

