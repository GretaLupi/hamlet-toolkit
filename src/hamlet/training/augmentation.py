"""Reproducible experimental-like perturbations for simulated spectroscopy."""

from dataclasses import replace

import numpy as np
from scipy.signal import convolve

from ..data import SpectroscopyDataset


def augment_experimental_like(
    dataset: SpectroscopyDataset,
    *,
    seed: int = 42,
    noise: float = 0.002,
    offset_range: tuple[float, float] = (0.009, 0.016),
    broadening_points: tuple[float, float] = (0.5, 2.0),
    energy_shift_mev: tuple[float, float] = (0.0, 0.0),
    energy_stretch: tuple[float, float] = (1.0, 1.0),
    background_quadratic: tuple[float, float] = (0.0, 0.0),
    amplitude_range: tuple[float, float] = (1.0, 1.0),
) -> SpectroscopyDataset:
    """Apply the original project's broadening, noise, offset, and drift.

    This operates once on the broad raw simulation dataset, before cutoff
    cropping. Consequently every model in a cutoff bank sees the same physical
    chains and the same perturbation realization.
    """
    if noise < 0:
        raise ValueError("noise must be non-negative")
    if not offset_range[0] <= offset_range[1]:
        raise ValueError("offset_range must satisfy low <= high")
    if broadening_points[0] <= 0 or broadening_points[0] > broadening_points[1]:
        raise ValueError("broadening_points must be positive and ordered")
    for name, interval in (
        ("energy_shift_mev", energy_shift_mev),
        ("energy_stretch", energy_stretch),
        ("background_quadratic", background_quadratic),
        ("amplitude_range", amplitude_range),
    ):
        if interval[0] > interval[1]:
            raise ValueError(f"{name} must be ordered")
    if energy_stretch[0] <= 0 or amplitude_range[0] <= 0:
        raise ValueError("energy_stretch and amplitude_range must be positive")
    rng = np.random.default_rng(seed)
    spectra = np.abs(np.asarray(dataset.spectra, dtype=np.float32)).copy()
    weight = np.linspace(0.6, 1.2, spectra.shape[-1], dtype=np.float32)
    bias = np.asarray(dataset.bias_mev, dtype=np.float32)
    normalized_bias = (bias - bias[0]) / max(float(bias[-1] - bias[0]), 1e-12)
    for sample in spectra:
        # Energy calibration is an instrument/chain-level nuisance and is
        # therefore shared by all sites in a simulated chain.
        shift = rng.uniform(*energy_shift_mev)
        stretch = rng.uniform(*energy_stretch)
        source_bias = (bias - shift) / stretch
        for row in sample:
            row *= rng.uniform(*amplitude_range)
            if shift != 0.0 or stretch != 1.0:
                row[:] = np.interp(source_bias, bias, row, left=row[0], right=row[-1])
            row += rng.uniform(*offset_range)
            gamma = rng.uniform(*broadening_points)
            kernel_size = max(5, int(np.ceil(gamma * 12)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            positions = np.arange(kernel_size) - (kernel_size - 1) / 2
            kernel = 1.0 / (1.0 + (positions / gamma) ** 2)
            kernel /= kernel.sum()
            row[:] = convolve(row, kernel.astype(np.float32), mode="same")
            row += rng.uniform(*background_quadratic) * normalized_bias**2
            if noise:
                row += rng.normal(size=row.size).astype(np.float32) * noise * weight
            threshold = float(np.mean(row)) * rng.uniform(0.1, 1.2)
            above = np.flatnonzero(row > threshold)
            start = int(above[0]) if above.size else row.size // 3
            slope = rng.uniform(3.2e-5, 5.1e-5) * rng.uniform(1.5, 3.0)
            row[start:] += np.arange(row.size - start, dtype=np.float32) * slope

    metadata = dict(dataset.metadata)
    metadata["experimental_like_augmentation"] = {
        "seed": seed,
        "noise": noise,
        "offset_range": list(offset_range),
        "broadening_points": list(broadening_points),
        "energy_shift_mev": list(energy_shift_mev),
        "energy_stretch": list(energy_stretch),
        "background_quadratic": list(background_quadratic),
        "amplitude_range": list(amplitude_range),
        "linear_drift": True,
    }
    return replace(dataset, spectra=spectra, metadata=metadata)
