"""Import the synthetic NPZ format used by the original research project."""

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ..data import SpectroscopyDataset


def load_reference_heisenberg_datasets(
    paths: Sequence[str | Path],
) -> SpectroscopyDataset:
    """Convert reference ``X/Z/J`` files into the package's meV dataset.

    The reference files store susceptibility on the DMRGPy energy grid and
    exchange couplings in DMRGPy units. Both axes and targets are multiplied by
    10, following the project convention. Susceptibility is integrated into
    the dI/dV-like observable used by the original training script.
    """
    if not paths:
        raise ValueError("at least one reference dataset path is required")
    all_spectra = []
    all_targets = []
    source_files = []
    common_bias = None
    common_sites = None

    for raw_path in paths:
        path = Path(raw_path)
        with np.load(path, allow_pickle=False) as data:
            required = {"X", "Z", "J", "n_sites", "n_bias"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            n_sites = int(data["n_sites"])
            n_bias = int(data["n_bias"])
            x = np.asarray(data["X"], dtype=np.float32).reshape(-1, n_sites, n_bias)
            z = np.asarray(data["Z"], dtype=np.float32).reshape(-1, n_sites, n_bias)
            targets_mev = np.asarray(data["J"], dtype=np.float32) * 10.0
        bias_mev = np.asarray(x[0, 0], dtype=np.float64) * 10.0
        if not np.allclose(x * 10.0, bias_mev[None, None, :], atol=1e-5):
            raise ValueError(f"{path} contains nonuniform site/sample bias grids")
        if common_bias is not None and not np.allclose(common_bias, bias_mev):
            raise ValueError("reference datasets use different bias grids")
        if common_sites is not None and common_sites != n_sites:
            raise ValueError("reference datasets use different chain lengths")
        # Match the original training code: integrate on the backend 0--10
        # grid, while exposing the corresponding 0--100 meV grid publicly.
        spectra = cumulative_trapezoid(z, x[0, 0], axis=2, initial=0.0).astype(np.float32)
        all_spectra.append(spectra)
        all_targets.append(targets_mev)
        source_files.append(str(path))
        common_bias = bias_mev
        common_sites = n_sites

    assert common_bias is not None and common_sites is not None
    return SpectroscopyDataset(
        spectra=np.concatenate(all_spectra),
        targets_mev=np.concatenate(all_targets),
        bias_mev=common_bias,
        target_names=tuple(f"J{i + 1}" for i in range(common_sites - 1)),
        system_type="inhomogeneous_heisenberg",
        metadata={
            "source_format": "Inhomogeneous-Heisenberg-HL X/Z/J NPZ",
            "source_files": source_files,
            "observable": "integrated local susceptibility (dI/dV-like)",
            "dmrgpy_energy_unit_mev": 10.0,
        },
    )
