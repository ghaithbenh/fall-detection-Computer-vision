"""Windowed RandomForest classifier used as an optional second opinion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from .features import FEATURE_NAMES


def aggregate_window(
    rows: Sequence[Mapping[str, float]],
    base_features: Sequence[str] = FEATURE_NAMES,
) -> dict[str, float]:
    """Summarize a short window with mean/std/min/max per feature."""
    out: dict[str, float] = {}
    for name in base_features:
        values = np.array([float(r.get(name, np.nan)) for r in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            values = np.zeros(1, dtype=np.float64)
        out[f"{name}_mean"] = float(values.mean())
        out[f"{name}_std"] = float(values.std())
        out[f"{name}_min"] = float(values.min())
        out[f"{name}_max"] = float(values.max())
    return out


class FallClassifier:
    """Loads a trained artifact and scores the last N frames."""

    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        base_features: Sequence[str],
        window: int,
        threshold: float = 0.60,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.base_features = list(base_features)
        self.window = int(window)
        self.threshold = float(threshold)
        self.metrics = metrics or {}

    @classmethod
    def load(cls, path: str | Path) -> "FallClassifier":
        data = joblib.load(Path(path))
        required = {"model", "feature_names", "base_features", "window"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"Invalid classifier artifact, missing: {sorted(missing)}")
        return cls(
            model=data["model"],
            feature_names=data["feature_names"],
            base_features=data["base_features"],
            window=data["window"],
            threshold=data.get("threshold", 0.60),
            metrics=data.get("metrics"),
        )

    def probability(self, rows: Sequence[Mapping[str, float]]) -> float | None:
        """Return P(fall) for the latest window, or None if not enough frames."""
        if len(rows) < self.window:
            return None
        window = list(rows)[-self.window :]
        agg = aggregate_window(window, self.base_features)
        x = np.array(
            [[agg[name] for name in self.feature_names]],
            dtype=np.float64,
        )
        proba = self.model.predict_proba(x)[0]
        positive = np.where(self.model.classes_ == 1)[0]
        index = int(positive[0]) if positive.size else len(proba) - 1
        return float(proba[index])
