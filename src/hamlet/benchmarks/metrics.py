import numpy as np
from numpy.typing import ArrayLike


def mae(predicted: ArrayLike, expected: ArrayLike) -> float:
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(expected))))


def rmse(predicted: ArrayLike, expected: ArrayLike) -> float:
    error = np.asarray(predicted) - np.asarray(expected)
    return float(np.sqrt(np.mean(error**2)))


def correlation_fidelity(predicted: ArrayLike, expected: ArrayLike) -> float:
    """Absolute Pearson correlation used as fidelity in the reference project."""
    predicted_array = np.asarray(predicted, dtype=float).ravel()
    expected_array = np.asarray(expected, dtype=float).ravel()
    if predicted_array.shape != expected_array.shape:
        raise ValueError("predicted and expected must have matching shapes")
    denominator = np.std(predicted_array) * np.std(expected_array)
    if denominator == 0:
        return 0.0
    return float(abs(np.mean(
        (predicted_array - predicted_array.mean())
        * (expected_array - expected_array.mean())
    )) / denominator)

