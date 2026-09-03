"""Validation-only selection and application of ensemble aggregation rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..benchmarks import correlation_fidelity, mae, rmse


@dataclass(frozen=True)
class EnsembleAggregation:
    """A fixed rule selected on synthetic validation data."""

    method: str = "mean"
    weights: tuple[float, ...] | None = None
    selection_metric: str = "validation_mae"

    def __post_init__(self) -> None:
        if self.method not in {"mean", "median", "validation_weighted"}:
            raise ValueError("aggregation method must be mean, median, or validation_weighted")
        if self.method == "validation_weighted":
            if not self.weights:
                raise ValueError("validation_weighted aggregation requires weights")
            values = np.asarray(self.weights, dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
                raise ValueError("aggregation weights must be finite, non-negative, and nonzero")
        elif self.weights is not None:
            raise ValueError(f"{self.method} aggregation does not use weights")

    def aggregate(self, predictions: ArrayLike) -> NDArray[np.float32]:
        values = np.asarray(predictions, dtype=np.float32)
        if values.ndim < 2:
            raise ValueError("predictions must start with a model axis")
        if self.method == "mean":
            combined = np.mean(values, axis=0)
        elif self.method == "median":
            combined = np.median(values, axis=0)
        else:
            weights = np.asarray(self.weights, dtype=np.float32)
            if weights.size != values.shape[0]:
                raise ValueError("aggregation weights must match the number of models")
            weights = weights / weights.sum()
            combined = np.tensordot(weights, values, axes=(0, 0))
        return np.asarray(combined, dtype=np.float32)


def select_ensemble_aggregation(
    predictions: ArrayLike,
    expected: ArrayLike,
    *,
    candidates: Sequence[str] = ("mean", "median", "validation_weighted"),
) -> tuple[EnsembleAggregation, dict[str, dict[str, Any]]]:
    """Select the lowest-MAE rule using validation predictions only."""
    values = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(expected, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != targets.shape:
        raise ValueError("predictions must have shape (models, examples, targets)")
    if not candidates:
        raise ValueError("at least one aggregation candidate is required")

    per_model_mae = np.asarray([mae(item, targets) for item in values], dtype=float)
    inverse = 1.0 / np.maximum(per_model_mae, 1e-8)
    inverse /= inverse.sum()
    rules = {
        "mean": EnsembleAggregation("mean"),
        "median": EnsembleAggregation("median"),
        "validation_weighted": EnsembleAggregation(
            "validation_weighted", tuple(float(item) for item in inverse)
        ),
    }
    unknown = set(candidates) - set(rules)
    if unknown:
        raise ValueError(f"unknown aggregation candidates: {sorted(unknown)}")

    metrics: dict[str, dict[str, Any]] = {}
    for name in candidates:
        rule = rules[name]
        combined = rule.aggregate(values)
        metrics[name] = {
            "mae": mae(combined, targets),
            "rmse": rmse(combined, targets),
            "correlation_fidelity": correlation_fidelity(combined, targets),
            "weights": list(rule.weights) if rule.weights is not None else None,
        }
    selected = min(candidates, key=lambda name: (metrics[name]["mae"], candidates.index(name)))
    return rules[selected], metrics
