"""Download the lightweight MediaPipe pose model (about 5.7 MB)."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from .config import DEFAULT_POSE_MODEL

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing file")
    args = parser.parse_args()

    target = DEFAULT_POSE_MODEL
    if target.exists() and not args.force:
        print(f"Model already exists: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, target)
    print(f"Saved to {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
