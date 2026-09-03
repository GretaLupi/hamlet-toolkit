"""Shared spectroscopy preprocessing for training and experiment inference."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SpectralPreprocessor:
    output_points: int = 200
    bias_range_mev: tuple[float, float] = (0.0, 50.0)
    baseline_range_mev: tuple[float, float] = (0.0, 3.0)
    scale_range_mev: tuple[float, float] = (40.0, 50.0)
    epsilon: float = 1e-12
    clip: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.output_points < 2:
            raise ValueError("output_points must be at least 2")
        lo, hi = self.bias_range_mev
        if not lo < hi:
            raise ValueError("bias_range_mev must satisfy low < high")
        for name, interval in (
            ("baseline_range_mev", self.baseline_range_mev),
            ("scale_range_mev", self.scale_range_mev),
        ):
            start, stop = interval
            if start < lo or stop > hi or not start < stop:
                raise ValueError(f"{name} must be a non-empty subset of bias_range_mev")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    @property
    def output_bias_mev(self) -> NDArray[np.float64]:
        return np.linspace(*self.bias_range_mev, self.output_points)

    def transform_spectrum(
        self, spectrum: ArrayLike, bias_mev: ArrayLike
    ) -> NDArray[np.float32]:
        values = np.asarray(spectrum, dtype=float)
        bias = np.asarray(bias_mev, dtype=float)
        if values.ndim != 1 or bias.ndim != 1 or values.size != bias.size:
            raise ValueError("spectrum and bias_mev must be one-dimensional and equally sized")
        if values.size < 2 or not np.all(np.isfinite(values)) or not np.all(np.isfinite(bias)):
            raise ValueError("spectrum and bias_mev need at least two finite points")
        if np.any(np.diff(bias) <= 0):
            raise ValueError("bias_mev must be strictly increasing")

        target_bias = self.output_bias_mev
        lo, hi = self.bias_range_mev
        if bias[0] > lo or bias[-1] < hi:
            raise ValueError("bias_mev does not cover the configured output range")
        transformed = np.interp(target_bias, bias, values)

        baseline_mask = self._range_mask(target_bias, self.baseline_range_mev)
        scale_mask = self._range_mask(target_bias, self.scale_range_mev)
        transformed = transformed - np.mean(transformed[baseline_mask])
        scale = float(np.mean(np.abs(transformed[scale_mask])))
        # ``epsilon`` is a relative degeneracy tolerance, not an additive
        # floor in signal units.  An absolute 1e-12 floor corrupts ordinary
        # lock-in data measured in amperes, whose complete dI/dV signal can
        # naturally be of that order.  Dividing by the measured scale keeps
        # preprocessing invariant under a change of signal units.
        amplitude = max(float(np.max(np.abs(transformed))), np.finfo(float).tiny)
        if scale <= self.epsilon * amplitude:
            raise ValueError(
                "normalization scale is effectively zero in scale_range_mev"
            )
        transformed = transformed / scale
        if self.clip is not None:
            transformed = np.clip(transformed, *self.clip)
        return np.nan_to_num(transformed).astype(np.float32)

    def transform_map(
        self, spectra: ArrayLike, bias_mev: ArrayLike
    ) -> NDArray[np.float32]:
        array = np.asarray(spectra, dtype=float)
        if array.ndim == 2:
            return np.stack([self.transform_spectrum(row, bias_mev) for row in array])
        if array.ndim == 3:
            return np.stack([self.transform_map(item, bias_mev) for item in array])
        raise ValueError("spectra must have shape (sites, bias) or (samples, sites, bias)")

    @staticmethod
    def _range_mask(bias: NDArray[np.float64], interval: tuple[float, float]):
        start, stop = interval
        mask = (bias >= start) & (bias <= stop)
        if not np.any(mask):
            raise ValueError(f"No output grid points fall inside range {interval}")
        return mask
