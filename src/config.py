"""Central configuration: paths, frame sizes and detection thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"
DEFAULT_CLASSIFIER = MODELS_DIR / "fall_rf.joblib"

FRAME_HEIGHT = 480  # 480p keeps CPU load low
MAX_FRAME_WIDTH = 854
DEFAULT_FPS = 25.0


@dataclass(frozen=True)
class DetectorConfig:
    """Thresholds for the rule-based fall state machine."""

    upright_angle: float = 35.0  # degrees from vertical
    lying_angle: float = 55.0  # degrees from vertical
    fall_window_s: float = 2.0  # max upright -> lying transition time
    confirm_lying_s: float = 0.35  # how long the person must stay horizontal
    drop_speed: float = 2.0  # hip drop speed in torso lengths / second
    cooldown_s: float = 4.0  # ignore repeated alerts for this long
    recover_upright_s: float = 1.0  # upright time needed to reset a fall
    history_s: float = 2.5  # feature history kept per track
    ml_threshold: float = 0.60  # RandomForest probability threshold


@dataclass(frozen=True)
class PipelineConfig:
    """Video processing settings."""

    model_path: Path = DEFAULT_POSE_MODEL
    num_poses: int = 1
    frame_height: int = FRAME_HEIGHT
    max_width: int = MAX_FRAME_WIDTH
    stride: int = 1
    min_visibility: float = 0.5
    detector: DetectorConfig = field(default_factory=DetectorConfig)
