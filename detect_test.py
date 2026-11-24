"""Quick visual test for src/detector.py on a video file.

Example:
    python detect_test.py data/videos/clip.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.detector import PoseDetector, draw_skeleton


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="video file to test")
    parser.add_argument(
        "--model", type=Path, default=None, help="optional .task model path"
    )
    args = parser.parse_args()

    if not args.video.is_file():
        print(f"Video not found: {args.video}")
        return 1

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print(f"Cannot open video: {args.video}")
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    window = "Pose detection - press q to quit"
    detected = 0
    frames = 0
    started = time.perf_counter()

    with PoseDetector(args.model) as detector:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            landmarks = detector.process(frame, frames / fps)
            if landmarks is not None:
                detected += 1
                draw_skeleton(frame, landmarks)

            person = "yes" if landmarks is not None else "no"
            status = f"frame {frames + 1}/{total or '?'} | person: {person}"
            cv2.putText(
                frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow(window, frame)
            frames += 1
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - started, 1e-6)
    print(
        f"Frames: {frames} | with person: {detected} | "
        f"{frames / elapsed:.1f} FPS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
