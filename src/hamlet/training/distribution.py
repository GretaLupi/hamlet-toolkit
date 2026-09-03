"""Compact training-distribution profiles for experimental compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class NearestTrainingExamples:
    distances: NDArray[np.float32]
    inputs: NDArray[np.float32]
    targets: NDArray[np.float32]


@dataclass(frozen=True)
class TrainingDistributionProfile:
    """Diagonal standardized-distance profile plus representative examples."""

    mean: NDArray[np.float32]
    standard_deviation: NDArray[np.float32]
    score_quantiles: NDArray[np.float32]
    quantile_levels: NDArray[np.float32]
    reference_inputs: NDArray[np.float32]
    reference_targets: NDArray[np.float32]

    @classmethod
    def fit(
        cls,
        inputs: ArrayLike,
        targets: ArrayLike,
        *,
        max_references: int = 256,
        seed: int = 42,
    ) -> "TrainingDistributionProfile":
        features = np.asarray(inputs, dtype=np.float32)
        labels = np.asarray(targets, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError("inputs must have shape (examples, features) with at least 2 rows")
        if labels.ndim != 2 or labels.shape[0] != features.shape[0]:
            raise ValueError("targets must match the input examples")
        if max_references < 1:
            raise ValueError("max_references must be positive")
        mean = features.mean(axis=0)
        deviation = features.std(axis=0)
        floor = max(float(np.median(deviation)) * 1e-3, 1e-6)
        deviation = np.maximum(deviation, floor)
        scores = _rms_z_score(features, mean, deviation)
        levels = np.asarray([0.5, 0.9, 0.95, 0.99], dtype=np.float32)
        quantiles = np.quantile(scores, levels).astype(np.float32)
        count = min(max_references, features.shape[0])
        indices = np.random.default_rng(seed).choice(features.shape[0], count, replace=False)
        return cls(
            mean.astype(np.float32),
            deviation.astype(np.float32),
            quantiles,
            levels,
            features[indices].copy(),
            labels[indices].copy(),
        )

    @property
    def threshold_95(self) -> float:
        index = int(np.argmin(np.abs(self.quantile_levels - 0.95)))
        return float(self.score_quantiles[index])

    def score(self, inputs: ArrayLike) -> NDArray[np.float32]:
        features = np.asarray(inputs, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.mean.size:
            raise ValueError(f"inputs must have shape (examples, {self.mean.size})")
        return _rms_z_score(features, self.mean, self.standard_deviation)

    def nearest(self, inputs: ArrayLike) -> NearestTrainingExamples:
        features = np.asarray(inputs, dtype=np.float32)
        self.score(features)  # shape validation
        scaled_query = (features - self.mean) / self.standard_deviation
        scaled_reference = (
            self.reference_inputs - self.mean
        ) / self.standard_deviation
        squared = np.mean(
            (scaled_query[:, None, :] - scaled_reference[None, :, :]) ** 2,
            axis=2,
        )
        indices = np.argmin(squared, axis=1)
        return NearestTrainingExamples(
            np.sqrt(squared[np.arange(features.shape[0]), indices]).astype(np.float32),
            self.reference_inputs[indices].copy(),
            self.reference_targets[indices].copy(),
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            mean=self.mean,
            standard_deviation=self.standard_deviation,
            score_quantiles=self.score_quantiles,
            quantile_levels=self.quantile_levels,
            reference_inputs=self.reference_inputs,
            reference_targets=self.reference_targets,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainingDistributionProfile":
        with np.load(path, allow_pickle=False) as data:
            return cls(*(np.asarray(data[name], dtype=np.float32) for name in (
                "mean",
                "standard_deviation",
                "score_quantiles",
                "quantile_levels",
                "reference_inputs",
                "reference_targets",
            )))

    def to_metadata(self) -> dict[str, object]:
        return {
            "method": "diagonal_rms_z_score",
            "feature_count": int(self.mean.size),
            "reference_count": int(self.reference_inputs.shape[0]),
            "quantile_levels": self.quantile_levels.astype(float).tolist(),
            "score_quantiles": self.score_quantiles.astype(float).tolist(),
            "threshold_95": self.threshold_95,
        }


def _rms_z_score(
    features: NDArray[np.float32],
    mean: NDArray[np.float32],
    deviation: NDArray[np.float32],
) -> NDArray[np.float32]:
    standardized = (features - mean) / deviation
    return np.sqrt(np.mean(standardized**2, axis=1)).astype(np.float32)
