"""Build a windowed feature dataset from labelled videos.

Expected layout:
    data/videos/*.mp4
    data/labels.csv   (optional: video,fall_start_s,fall_end_s)

Without labels.csv, rule-based detections are used as weak labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, PipelineConfig
from .features import FEATURE_NAMES
from .model import aggregate_window
from .pipeline import FEATURE_ROW_COLUMNS, FallPipeline

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MIN_VALID_RATIO = 0.6


def _find_videos(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _cache_path(
    video: Path,
    root: Path,
    cfg: PipelineConfig,
    cache_dir: Path,
) -> Path:
    rel = str(video.relative_to(root)).replace("\\", "/")
    key = f"{root}|{rel}|{cfg.detector}|{cfg.frame_height}|{cfg.stride}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"{video.stem}_{digest}.csv"


def _extract_rows(
    video: Path,
    root: Path,
    cfg: PipelineConfig,
    cache_dir: Path,
    rebuild: bool,
) -> pd.DataFrame:
    cache = _cache_path(video, root, cfg, cache_dir)
    if (
        cache.is_file()
        and not rebuild
        and cache.stat().st_mtime >= video.stat().st_mtime
    ):
        return pd.read_csv(cache)

    print(f"  extracting features: {video.relative_to(root)}")
    with FallPipeline(cfg) as pipeline:
        report = pipeline.process_video(video, draw=False)
    rows = pd.DataFrame(report.rows, columns=FEATURE_ROW_COLUMNS)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(cache, index=False)
    return rows


def _load_labels(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    labels = pd.read_csv(path)
    if "video" not in labels.columns or "fall_start_s" not in labels.columns:
        raise ValueError(
            "labels.csv needs columns: video, fall_start_s, fall_end_s"
        )
    return labels


def _true_mask(
    rows: pd.DataFrame,
    video: Path,
    labels: pd.DataFrame,
) -> np.ndarray:
    names = (
        labels["video"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[-1]
        .str.lower()
    )
    subset = labels[names == video.name.lower()]
    mask = np.zeros(len(rows), dtype=bool)
    if subset.empty:
        print(f"  warning: no labels for {video.name}; using all-negative windows")
        return mask
    for _, label in subset.iterrows():
        start = float(label["fall_start_s"])
        end = label.get("fall_end_s")
        end = float(end) if pd.notna(end) else start + 2.0
        mask |= (
            (rows["timestamp_s"] >= start - 0.5)
            & (rows["timestamp_s"] <= end + 1.0)
        ).to_numpy()
    return mask


def _weak_mask(rows: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(rows), dtype=bool)
    event_times = rows.loc[rows["event"] == 1, "timestamp_s"].to_numpy()
    for t in event_times:
        mask |= (
            (rows["timestamp_s"] >= t - 1.0)
            & (rows["timestamp_s"] <= t + 2.0)
        ).to_numpy()
    return mask


def _windows(
    rows: pd.DataFrame,
    positive: np.ndarray,
    video_name: str,
    window: int,
    step: int,
) -> list[dict]:
    windows: list[dict] = []
    if rows.empty:
        return windows
    for track_id, group in rows.groupby("track_id", sort=False):
        group = group.sort_values("timestamp_s")
        if len(group) < window:
            continue
        original = group.index.to_numpy()
        torso_len = group["torso_len"].to_numpy(dtype=float)
        valid = np.isfinite(torso_len) & (torso_len >= 0.05)
        for start in range(0, len(group) - window + 1, step):
            if valid[start : start + window].mean() < MIN_VALID_RATIO:
                continue
            chunk = group.iloc[start : start + window]
            label = int(positive[original[start : start + window]].any())
            summary = aggregate_window(
                chunk[list(FEATURE_NAMES)].to_dict("records")
            )
            windows.append(
                {
                    "video": video_name,
                    "track_id": int(track_id),
                    "t_start": float(chunk["timestamp_s"].iloc[0]),
                    "t_end": float(chunk["timestamp_s"].iloc[-1]),
                    "label": label,
                    **summary,
                }
            )
    return windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=DATA_DIR / "videos")
    parser.add_argument("--labels", type=Path, default=DATA_DIR / "labels.csv")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "windows.csv")
    parser.add_argument("--cache", type=Path, default=DATA_DIR / "features")
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--weak-labels", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if not args.videos.is_dir():
        print(f"Video folder not found: {args.videos}")
        print("Create it and add clips, e.g. data/videos/fall_01.mp4")
        return 1
    videos = _find_videos(args.videos)
    if not videos:
        print(f"No videos found in {args.videos}")
        return 1

    labels = None if args.weak_labels else _load_labels(args.labels)
    if labels is None and not args.weak_labels:
        print("labels.csv not found: using rule-based weak labels")
    cfg = PipelineConfig(stride=max(1, args.stride))

    all_windows: list[dict] = []
    for video in videos:
        rows = _extract_rows(video, args.videos, cfg, args.cache, args.rebuild)
        if labels is not None:
            mask = _true_mask(rows, video, labels)
            source = "labels.csv"
        else:
            mask = _weak_mask(rows)
            source = "weak-rules"
        name = str(video.relative_to(args.videos)).replace("\\", "/")
        all_windows.extend(
            _windows(rows, mask, name, args.window, args.step)
        )

    if not all_windows:
        print("No valid windows were produced. Check the videos and labels.")
        return 1

    dataset = pd.DataFrame(all_windows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.out, index=False)

    meta = {
        "window": args.window,
        "step": args.step,
        "base_features": list(FEATURE_NAMES),
        "label_source": source,
        "videos": [
            str(v.relative_to(args.videos)).replace("\\", "/") for v in videos
        ],
    }
    args.out.with_name("windows_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    positives = int(dataset["label"].sum())
    print(f"\nSaved {len(dataset)} windows to {args.out}")
    print(f"  videos: {len(videos)} | positive: {positives} | negative: {len(dataset) - positives}")
    print(f"  label source: {meta['label_source']}")
    if positives == 0:
        print("  warning: no positive windows; add fall clips or labels.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
