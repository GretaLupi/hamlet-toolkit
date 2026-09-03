"""Adapter from local ML models to arbitrary-length chain predictions."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..preprocessing.windows import make_local_windows, reconstruct_chain


class PredictsWindows(Protocol):
    def predict(self, inputs: NDArray[np.float32], **kwargs: object) -> ArrayLike: ...


@dataclass(frozen=True)
class ChainPrediction:
    couplings_mev: NDArray[np.float32]
    window_predictions: NDArray[np.float32]


class LocalChainEstimator:
    """Apply a three-site/two-bond model to any chain of at least three sites."""

    def __init__(
        self,
        model: PredictsWindows,
        coupling_range_mev: tuple[float, float] | None = None,
        window_sites: int = 3,
    ) -> None:
        if window_sites != 3:
            raise ValueError("the current two-bond reconstruction requires window_sites=3")
        if coupling_range_mev is not None and not coupling_range_mev[0] < coupling_range_mev[1]:
            raise ValueError("coupling_range_mev must satisfy low < high")
        self.model = model
        self.coupling_range_mev = coupling_range_mev
        self.window_sites = window_sites

    def predict(self, processed_spectra: ArrayLike) -> ChainPrediction:
        windows = make_local_windows(processed_spectra, self.window_sites)
        if windows.ndim != 2:
            raise ValueError("predict currently accepts one chain with shape (sites, bias)")
        try:
            raw = self.model.predict(windows, verbose=0)
        except TypeError:
            raw = self.model.predict(windows)
        window_predictions = np.asarray(raw, dtype=np.float32)
        expected = (windows.shape[0], 2)
        if window_predictions.shape != expected:
            raise ValueError(f"model returned {window_predictions.shape}; expected {expected}")
        couplings = reconstruct_chain(window_predictions)
        if self.coupling_range_mev is not None:
            low, high = self.coupling_range_mev
            couplings = couplings * (high - low) + low
        return ChainPrediction(couplings, window_predictions)

