# Fall Detection & Tracking

University project: detect and track human falls in video using **MediaPipe
PoseLandmarker (Tasks API, lite model)**, lightweight geometry features and a
temporal state machine, with an optional scikit-learn RandomForest confirmation
and a Streamlit demo.

Designed for a weak CPU laptop (i5-4200U, 6 GB RAM, no GPU): frames are resized
to **480p**, only the 5.7 MB `pose_landmarker_lite.task` model is used, and the
ML model is a small RandomForest.

> This is **not** a medical or safety-certified device. It is a computer-vision
> coursework prototype.

---

## Project structure

```
fall_detection/
├── app.py                  # Streamlit demo UI
├── requirements.txt
├── models/
│   └── pose_landmarker_lite.task
├── data/
│   ├── videos/             # your clips (create this)
│   ├── labels.example.csv  # label template for supervised training
│   └── features/           # cached per-frame features (generated)
├── outputs/                # annotated videos + CSV exports (generated)
├── src/
│   ├── config.py           # paths, frame size, detector thresholds
│   ├── pose_estimator.py   # MediaPipe Tasks PoseLandmarker wrapper
│   ├── tracker.py          # lightweight IoU tracker (stable IDs)
│   ├── features.py         # torso angle, hip drop speed, etc.
│   ├── fall_detector.py    # temporal fall state machine
│   ├── annotate.py         # OpenCV skeleton/bbox/alert drawing
│   ├── pipeline.py         # video processing pipeline
│   ├── model.py            # optional RandomForest wrapper
│   ├── build_dataset.py    # features -> windowed training CSV
│   ├── train.py            # train/evaluate RandomForest
│   ├── run_video.py        # command-line runner
│   └── download_model.py   # fetch the .task model
└── tests/
    └── test_logic.py       # unit tests (no video required)
```

The project uses the **Tasks API** (`mediapipe.tasks.python.vision.PoseLandmarker`),
not the removed/deprecated `mp.solutions.pose` API.

---

## Requirements

- **Python 3.12** (MediaPipe 0.10.x does not support 3.13/3.14)
- OpenCV, MediaPipe, NumPy, pandas, scikit-learn, Streamlit, joblib
- Webcam not required; a short MP4/AVI clip is enough

## Setup (Windows PowerShell)

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.download_model
```

`python -m src.download_model` downloads
`models/pose_landmarker_lite.task` (about 5.7 MB). The file is already present
in this workspace.

---

## Quick start

### 1. Command line

```powershell
python -m src.run_video --input data/videos/clip.mp4 --show
```

Outputs:

- `outputs/clip_annotated.mp4` – skeleton, bounding box, track ID, alert banner
- `outputs/clip_features.csv` – one row per detected person per frame
- `outputs/clip_events.csv` – one row per detected fall

Useful options:

```powershell
python -m src.run_video `
  --input clip.mp4 `
  --stride 2 `                 # process every 2nd frame (faster)
  --lying-angle 55 `           # torso angle considered "lying"
  --drop-speed 2.0 `           # hip drop speed threshold (torso lengths/s)
  --classifier models/fall_rf.joblib   # optional ML confirmation
```

You can also set custom export paths with `--csv features.csv` and
`--events events.csv`.

Press `q` in the preview window to stop early.

### 2. Streamlit demo

```powershell
streamlit run app.py
```

Upload a clip, adjust the thresholds in the sidebar, run detection, then watch
or download the annotated video, events CSV and features CSV. The page also
shows torso-angle / hip-drop signals and before/at/after key frames for each
fall.

---

## How it works (5 lines for the professor)

1. Every frame is resized to 480p and passed to **MediaPipe PoseLandmarker
   (lite)** in VIDEO mode, which returns 33 body keypoints per person.
2. A lightweight **IoU tracker** matches detections between frames, assigns a
   stable track ID and smooths the bounding box.
3. **Geometry features** are computed from the keypoints: torso angle from
   vertical, bounding-box aspect ratio, hip height and **hip drop speed** in
   torso-lengths per second.
4. A **state machine** raises a fall when the person was upright and becomes
   horizontal within 2 s, stays horizontal for 0.35 s, and the hips drop faster
   than 2 torso-lengths/s; a 4 s cooldown prevents duplicate alerts.
5. The optional **RandomForest** (trained on 15-frame feature windows) can
   confirm a lying posture, and all results are drawn on the video and exported
   as MP4/CSV.

### Detection logic

| Feature | Meaning | Why it helps |
|---|---|---|
| `torso_angle` | Angle between the hip→shoulder vector and vertical (0° upright, 90° lying) | Main posture cue |
| `hip_drop_speed` | Downward hip motion in torso lengths / second | Separates a fall from slowly lying down |
| `bbox_aspect` | Bounding-box width / height | Lying bodies are usually wider than tall |
| `knee_angle` | Average hip–knee–ankle angle | Distinguishes sitting/crouching |
| `center_speed` | Body-centre motion | Detects fast whole-body movement |

A fall is reported only when **all** of these hold:

1. the person was `upright` (torso ≤ 35°) within the last 2 s,
2. the torso is now `lying` (≥ 55°) for at least 0.35 s,
3. the maximum hip drop speed in that interval is ≥ 2 torso-lengths/s,
4. no fall was reported for that track in the last 4 s.

This makes a fast fall trigger, while sitting down or lying down slowly does
not. A person must stand upright for 1 s before the track can trigger again.

---

## Optional RandomForest training

The rule-based detector works without training. The ML part is optional and is
useful for the report/viva.

### Data layout

```
data/
├── videos/
│   ├── fall_01.mp4
│   ├── fall_02.mp4
│   ├── normal_01.mp4
│   └── normal_02.mp4
└── labels.csv          # optional but recommended
```

`data/labels.csv`:

```csv
video,fall_start_s,fall_end_s
fall_01.mp4,4.2,7.5
fall_02.mp4,2.8,5.0
```

- `fall_start_s` is the moment the fall begins (person starts losing balance).
- `fall_end_s` is optional; if omitted, the window `start … start+2 s` is used.

### Build the dataset

```powershell
python -m src.build_dataset
```

- Extracts features from every clip (cached in `data/features/`).
- Slides a **15-frame window** with step 5 and summarizes each feature with
  mean/std/min/max.
- Labels a window positive if it overlaps a labelled fall.
- Without `labels.csv`, rule-based detections are used as **weak labels**
  (`--weak-labels` forces this). For a real evaluation, label the clips.
- Writes `data/windows.csv` and `data/windows_meta.json`.

### Train and evaluate

```powershell
python -m src.train
```

The script:

- splits **by video** when possible (so windows from the same clip never leak
  into both train and test); with too few labelled videos it falls back to a
  stratified split and prints a warning,
- trains a `RandomForestClassifier` (150 trees, balanced classes),
- prints accuracy, precision, recall, F1, ROC AUC, confusion matrix and the
  top-10 important features,
- saves `models/fall_rf.joblib` and `models/fall_rf.metrics.json`.

Use it at runtime with:

```powershell
python -m src.run_video --input clip.mp4 --classifier models/fall_rf.joblib
```

or enable **RandomForest confirmation** in the Streamlit sidebar.

> The RandomForest only confirms a **lying** posture; it cannot trigger while
> the person is upright. This keeps false positives low. With weak labels the
> metrics measure how well the forest imitates the rules, not true fall
> accuracy — use manually labelled clips for the report.

---

## Configuration

Main thresholds live in `src/config.py` (`DetectorConfig`):

| Parameter | Default | Effect |
|---|---|---|
| `upright_angle` | 35° | Maximum torso angle considered upright |
| `lying_angle` | 55° | Minimum torso angle considered lying |
| `fall_window_s` | 2.0 s | Maximum time for upright → lying |
| `confirm_lying_s` | 0.35 s | How long the person must stay down |
| `drop_speed` | 2.0 | Minimum hip drop speed (torso/s) |
| `cooldown_s` | 4.0 s | No duplicate alerts |
| `recover_upright_s` | 1.0 s | Upright time needed to reset |
| `ml_threshold` | 0.60 | RandomForest probability threshold |

Lower `drop_speed` / `lying_angle` → more sensitive, more false positives.
For a camera high on a wall, a standing person can look shorter, so test the
thresholds on your own clips.

---

## Performance on a weak laptop

- 480p frames, lite model, CPU delegate; no GPU required.
- `--stride 2` roughly halves processing time (and the output FPS).
- `--num-poses 1` (default) is fastest; use 2 only for two-person clips.
- Keep clips short (10–60 s) in the Streamlit demo.
- Processing speed is printed as **processed FPS**; playback is independent of
  processing speed because the annotated video keeps the source timing.

Typical results on an i5-4200U are in the 10–25 processed-FPS range, depending
on clip resolution and background load. The exact number is shown after each
run.

---

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover torso-angle computation, hip-drop speed, tracker ID stability,
fast/slow fall transitions, duplicate suppression, recovery and 480p resizing.
They do **not** need the MediaPipe model or a video.

---

## Limitations

- Single RGB camera; occlusion, poor lighting and unusual camera angles reduce
  landmark quality.
- A person already lying at the start of the clip cannot produce a transition,
  so no fall is reported.
- Rapid sitting or crouching can look like a fall; tune thresholds and/or use
  the ML confirmation.
- No face recognition or personal identity; track IDs are anonymous.
- The model is trained for a general pose; it is not a medical fall detector.

---

## Files you will submit

- All `src/` modules and `app.py`
- `requirements.txt`
- `models/pose_landmarker_lite.task`
- `tests/test_logic.py`
- `data/labels.example.csv`
- Generated `outputs/*_annotated.mp4`, `*_events.csv`, `*_features.csv`
  for your demo clips
