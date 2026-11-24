"""Build data/labels.csv from data/raw/falls and data/raw/nonfalls.

Output columns:

    clip,label,split,fall_start

* ``clip``: path relative to ``data/raw``, e.g. ``falls/fall-01-cam0.mp4``
* ``label``: 1 = fall, 0 = no fall (taken from the source folder)
* ``split``: train / val / test, assigned per clip, 60/20/20, stratified by
  label and reproducible from the seed. Frames of one clip never straddle
  two splits.
* ``fall_start``: seconds from the start of the clip. Read from
  ``data/raw/fall_starts.csv`` (columns ``clip,fall_start``) when present,
  otherwise left empty for you to fill in. Non-fall clips are always empty.

``data/unseen`` is never scanned: it is a held-out set, not training data.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "labels.csv"
FALL_STARTS_PATH = RAW_DIR / "fall_starts.csv"

FALL_DIR = "falls"
NONFALL_DIR = "nonfalls"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SPLITS = ("train", "val", "test")
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2
DEFAULT_SEED = 42


def find_clips(raw_dir: Path, folder: str) -> list[str]:
    """Video files under ``raw_dir/folder`` as raw-relative posix paths."""
    root = raw_dir / folder
    if not root.is_dir():
        return []
    return sorted(
        path.relative_to(raw_dir).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and "unseen" not in path.relative_to(raw_dir).parts
    )


def split_sizes(n: int) -> tuple[int, int, int]:
    """Split ``n`` clips into 60/20/20, rounding to whole clips.

    Uses the largest-remainder method so the totals are exact, then makes
    sure all three splits are non-empty once there are at least 3 clips.
    Smaller groups cannot be split three ways; the clips go to train/test
    so that a class is at least represented in the held-out set.
    """
    if n <= 0:
        return (0, 0, 0)
    if n == 1:
        return (1, 0, 0)
    if n == 2:
        return (1, 0, 1)

    ideal = [n * TRAIN_FRAC, n * VAL_FRAC, n * TEST_FRAC]
    counts = [int(value) for value in ideal]
    short = n - sum(counts)
    order = sorted(
        range(3), key=lambda i: (ideal[i] - counts[i], -i), reverse=True
    )
    for index in order[:short]:
        counts[index] += 1

    for index in range(3):
        if counts[index] == 0:
            donor = max(range(3), key=lambda i: counts[i])
            counts[donor] -= 1
            counts[index] = 1
    return counts[0], counts[1], counts[2]


def assign_splits(
    clips_by_label: dict[int, list[str]], seed: int
) -> dict[str, list[str]]:
    """Shuffle each label group separately and deal it into the three splits."""
    rng = random.Random(seed)
    splits = {name: [] for name in SPLITS}
    for label in sorted(clips_by_label):
        items = sorted(clips_by_label[label])
        rng.shuffle(items)
        n_train, n_val, _ = split_sizes(len(items))
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])
    return splits


def load_fall_starts(path: Path) -> dict[str, float]:
    """Map clip identifiers to fall start times from an optional CSV."""
    if not path.is_file():
        return {}
    table: dict[str, float] = {}
    # utf-8-sig so annotations saved by Excel/Notepad (which add a BOM) work.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "clip" not in fields or "fall_start" not in fields:
            raise SystemExit(
                f"{path} needs columns clip,fall_start (found: {fields})"
            )
        for row in reader:
            clip = (row.get("clip") or "").strip().replace("\\", "/")
            raw = (row.get("fall_start") or "").strip()
            if not clip or not raw:
                continue
            value = float(raw)
            names = {clip, clip.lower(), Path(clip).name.lower()}
            names.add(Path(clip).stem.lower())
            for name in names:
                table[name] = value
    return table


def lookup_fall_start(clip: str, table: dict[str, float]) -> float | None:
    for key in (clip.lower(), Path(clip).name.lower(), Path(clip).stem.lower()):
        if key in table:
            return table[key]
    return None


def format_seconds(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="raw clip root")
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="output CSV")
    parser.add_argument(
        "--fall-starts",
        type=Path,
        default=FALL_STARTS_PATH,
        help="optional CSV with clip,fall_start",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    falls = find_clips(args.raw, FALL_DIR)
    nonfalls = find_clips(args.raw, NONFALL_DIR)
    if not falls and not nonfalls:
        print(f"No clips found under {args.raw / FALL_DIR} or {args.raw / NONFALL_DIR}")
        print("Expected layout: data/raw/falls/*.mp4 and data/raw/nonfalls/*.mp4")
        return 1
    for folder, clips in ((FALL_DIR, falls), (NONFALL_DIR, nonfalls)):
        if not clips:
            print(f"warning: no clips in {args.raw / folder}")

    clips_by_label = {1: falls, 0: nonfalls}
    splits = assign_splits(clips_by_label, args.seed)
    labels_by_clip = {
        clip: label for label, clips in clips_by_label.items() for clip in clips
    }
    fall_starts = load_fall_starts(args.fall_starts)

    rows = []
    missing_fall_start = 0
    for split in SPLITS:
        for clip in sorted(splits[split]):
            label = labels_by_clip[clip]
            start = lookup_fall_start(clip, fall_starts) if label == 1 else None
            if label == 1 and start is None:
                missing_fall_start += 1
            rows.append(
                {
                    "clip": clip,
                    "label": label,
                    "split": split,
                    "fall_start": format_seconds(start),
                }
            )
    rows.sort(key=lambda row: row["clip"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["clip", "label", "split", "fall_start"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} clips to {args.out}")
    print(f"{'':6} {'fall':>6} {'no-fall':>8}")
    for split in SPLITS:
        part = [row for row in rows if row["split"] == split]
        positives = sum(row["label"] for row in part)
        print(f"{split:6} {positives:6} {len(part) - positives:8}")
    if missing_fall_start:
        print(
            f"warning: {missing_fall_start} fall clip(s) have no fall_start; "
            f"add them to {args.fall_starts}"
        )
    if not falls or not nonfalls:
        print("warning: need both classes for a stratified split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
