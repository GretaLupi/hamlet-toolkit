"""Load the long-form STM CSV schema used by the reference project."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..measurements import Measurement


def load_measurement_csv(path: str | Path) -> Measurement:
    """Load the long-form spectroscopy CSV as a canonical measurement."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on installation extras
        raise ImportError("CSV loading requires: pip install 'hamlet-toolkit[io]'") from exc

    frame = pd.read_csv(path)
    required = {"bias_meV", "site", "didv_A"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    # Keep first-occurrence order: chain position is physical and must not be
    # silently changed by lexicographic sorting of labels such as site_2/site_10.
    sites = frame["site"].drop_duplicates().to_numpy()
    reference_bias: NDArray[np.float64] | None = None
    channels: dict[str, list[NDArray[np.float64]]] = {
        "didv": [],
    }
    auxiliary_columns = {
        column.removesuffix("_A"): column
        for column in frame.columns
        if column.endswith("_A") and column != "didv_A"
    }
    channels.update({name: [] for name in auxiliary_columns})
    for site in sites:
        subset = frame.loc[frame["site"] == site].sort_values("bias_meV")
        bias = subset["bias_meV"].to_numpy(dtype=float)
        if reference_bias is None:
            reference_bias = bias
        elif not np.array_equal(reference_bias, bias):
            raise ValueError("all sites must share the same bias grid")
        channels["didv"].append(subset["didv_A"].to_numpy(dtype=float))
        for name, column in auxiliary_columns.items():
            channels[name].append(subset[column].to_numpy(dtype=float))
    if reference_bias is None or not channels["didv"]:
        raise ValueError("CSV contains no spectroscopy rows")
    return Measurement(
        axes={"site": sites, "bias": reference_bias},
        channels={name: np.stack(rows) for name, rows in channels.items()},
        axis_units={"site": "index", "bias": "meV"},
        channel_units={name: "A" for name in channels},
        primary_channel="didv",
        metadata={"source": str(path), "schema": "long_form_spectroscopy_csv_v1"},
    )


def load_spectroscopy_csv(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load columns ``bias_meV``, ``site``, and ``didv_A``.

    Returns ``(bias_mev, didv)`` where ``didv`` has shape ``(sites, bias)``.
    Pandas is an optional dependency so the numerical core stays lightweight.
    """
    measurement = load_measurement_csv(path)
    return measurement.site_spectra()
