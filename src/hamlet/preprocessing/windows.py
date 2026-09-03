"""Length-independent local windows and chain reconstruction."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def make_local_windows(
    spectra: ArrayLike, window_sites: int = 3
) -> NDArray[np.float32]:
    """Return flattened consecutive-site windows.

    Accepts ``(sites, bias)`` and returns ``(windows, window_sites * bias)``;
    accepts ``(samples, sites, bias)`` and returns
    ``(samples, windows, window_sites * bias)``.
    """
    array = np.asarray(spectra, dtype=np.float32)
    if window_sites < 2:
        raise ValueError("window_sites must be at least 2")
    if array.ndim not in (2, 3):
        raise ValueError("spectra must have shape (sites, bias) or (samples, sites, bias)")
    n_sites = array.shape[-2]
    if n_sites < window_sites:
        raise ValueError(f"at least {window_sites} sites are required")
    windows = [array[..., i : i + window_sites, :] for i in range(n_sites - window_sites + 1)]
    stacked = np.stack(windows, axis=-3)
    return stacked.reshape(*stacked.shape[:-2], -1)


def reconstruct_chain(window_predictions: ArrayLike) -> NDArray[np.float32]:
    """Average overlapping two-bond predictions into a full open chain.

    Input shape is ``(windows, 2)`` or ``(samples, windows, 2)``. A three-site
    window produces two bonds; adjacent estimates of an interior bond receive
    equal weight.
    """
    predictions = np.asarray(window_predictions, dtype=np.float32)
    if predictions.ndim not in (2, 3) or predictions.shape[-1] != 2:
        raise ValueError("window_predictions must have shape (windows, 2) or (samples, windows, 2)")
    n_windows = predictions.shape[-2]
    if n_windows < 1:
        raise ValueError("at least one window prediction is required")

    output = np.zeros((*predictions.shape[:-2], n_windows + 1), dtype=np.float32)
    counts = np.zeros(n_windows + 1, dtype=np.float32)
    output[..., :n_windows] += predictions[..., :, 0]
    output[..., 1:] += predictions[..., :, 1]
    counts[:n_windows] += 1
    counts[1:] += 1
    return output / counts

