"""Train the optional RandomForest fall classifier."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import DATA_DIR, DEFAULT_CLASSIFIER
from .features import FEATURE_NAMES


def _split(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if groups.nunique() >= 3:
        # Try a few seeds so the holdout contains both classes when possible.
        for offset in range(20):
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=0.3, random_state=seed + offset
            )
            train_idx, test_idx = next(splitter.split(x, y, groups))
            if (
                y.iloc[train_idx].nunique() == 2
                and y.iloc[test_idx].nunique() == 2
            ):
                return train_idx, test_idx, "grouped by video"
        print(
            "warning: could not make a grouped split with both classes; "
            "falling back to a random stratified split (label more videos "
            "for a clean evaluation)"
        )
    indices = np.arange(len(y))
    try:
        train_idx, test_idx = train_test_split(
            indices, test_size=0.3, random_state=seed, stratify=y
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            indices, test_size=0.3, random_state=seed
        )
    return train_idx, test_idx, "random (add more labelled videos)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=DATA_DIR / "windows.csv")
    parser.add_argument(
        "--meta", type=Path, default=DATA_DIR / "windows_meta.json"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--trees", type=int, default=150)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.windows.is_file():
        print(f"Dataset not found: {args.windows}")
        print("Run: python -m src.build_dataset")
        return 1

    df = pd.read_csv(args.windows)
    if df.empty:
        print("Dataset is empty.")
        return 1
    meta = {}
    meta_path = args.meta
    if not meta_path.is_file():
        candidate = args.windows.with_name("windows_meta.json")
        if candidate.is_file():
            meta_path = candidate
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    feature_cols = [
        col
        for col in df.columns
        if col.endswith(("_mean", "_std", "_min", "_max"))
    ]
    if not feature_cols:
        print("No aggregated feature columns found.")
        return 1

    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["label"].astype(int)
    groups = df["video"].astype(str)
    if y.nunique() < 2:
        print("Need both fall and normal windows. Add fall clips or labels.csv.")
        return 1

    train_idx, test_idx, split_desc = _split(x, y, groups, args.seed)
    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=2,
    )
    model.fit(
        x.iloc[train_idx].to_numpy(dtype=np.float64),
        y.iloc[train_idx].to_numpy(),
    )

    proba = model.predict_proba(
        x.iloc[test_idx].to_numpy(dtype=np.float64)
    )[:, 1]
    y_test = y.iloc[test_idx]
    y_pred = (proba >= args.threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_test, proba))
            if y_test.nunique() > 1
            else None
        ),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ).tolist(),
        "split": split_desc,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_positives": int(y_test.sum()),
        "threshold": float(args.threshold),
    }

    print(f"Split: {split_desc}")
    print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["normal", "fall"],
        zero_division=0,
    ))
    print("Confusion matrix [[TN, FP], [FN, TP]]:", metrics["confusion_matrix"])
    if metrics["roc_auc"] is not None:
        print(f"ROC AUC: {metrics['roc_auc']:.3f}")

    importances = np.argsort(model.feature_importances_)[::-1][:10]
    print("\nTop features:")
    for index in importances:
        print(f"  {feature_cols[index]}: {model.feature_importances_[index]:.3f}")

    artifact = {
        "model": model,
        "feature_names": feature_cols,
        "base_features": meta.get("base_features") or list(FEATURE_NAMES),
        "window": int(meta.get("window", 15)),
        "threshold": float(args.threshold),
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out)
    args.out.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"\nSaved model: {args.out}")
    print(f"Metrics: {args.out.with_suffix('.metrics.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
