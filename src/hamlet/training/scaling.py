"""Serializable scaling for physical regression targets."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class MinMaxTargetScaler:
    minimum: NDArray[np.float32]
    maximum: NDArray[np.float32]

    @classmethod
    def fit(cls, targets: ArrayLike) -> "MinMaxTargetScaler":
        values = np.asarray(targets, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError("targets must have shape (examples, parameters)")
        return cls(values.min(axis=0), values.max(axis=0))

    def transform(self, targets: ArrayLike) -> NDArray[np.float32]:
        values = np.asarray(targets, dtype=np.float32)
        return (values - self.minimum) / (self.maximum - self.minimum + 1e-12)

    def inverse_transform(self, scaled: ArrayLike) -> NDArray[np.float32]:
        values = np.asarray(scaled, dtype=np.float32)
        return values * (self.maximum - self.minimum) + self.minimum

    def to_metadata(self) -> dict[str, list[float]]:
        return {
            "minimum": self.minimum.astype(float).tolist(),
            "maximum": self.maximum.astype(float).tolist(),
        }

