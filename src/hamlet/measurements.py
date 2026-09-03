"""Canonical, laboratory-independent measurement containers."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .branding import brand_manifest


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class Measurement:
    """A named-axis measurement with one or more aligned observable channels.

    The container is deliberately independent of an instrument or physical
    system.  For site-resolved spectroscopy the conventional axes are
    ``site`` and ``bias`` and the inference channel is ``didv``.  Additional
    channels, such as a second derivative, remain attached for visualization
    and quality control.
    """

    axes: Mapping[str, ArrayLike]
    channels: Mapping[str, ArrayLike]
    axis_units: Mapping[str, str] = field(default_factory=dict)
    channel_units: Mapping[str, str] = field(default_factory=dict)
    primary_channel: str = "didv"
    masks: Mapping[str, ArrayLike] | None = None
    uncertainties: Mapping[str, ArrayLike] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        axes = {str(name): np.asarray(values) for name, values in self.axes.items()}
        if not axes:
            raise ValueError("a measurement requires at least one axis")
        for name, values in axes.items():
            if values.ndim != 1 or values.size == 0:
                raise ValueError(f"axis {name!r} must be a non-empty 1D array")
        shape = tuple(len(values) for values in axes.values())

        channels = {
            str(name): np.asarray(values, dtype=float) for name, values in self.channels.items()
        }
        if not channels:
            raise ValueError("a measurement requires at least one channel")
        if self.primary_channel not in channels:
            raise ValueError(f"primary channel {self.primary_channel!r} is not present")
        for name, values in channels.items():
            if values.shape != shape:
                raise ValueError(
                    f"channel {name!r} has shape {values.shape}; expected {shape} from axes"
                )

        supplied_masks = self.masks or {}
        masks: dict[str, NDArray[np.bool_]] = {}
        for name, values in channels.items():
            mask = np.asarray(supplied_masks.get(name, np.isfinite(values)), dtype=bool)
            if mask.shape != shape:
                raise ValueError(f"mask for {name!r} must have shape {shape}")
            masks[name] = mask & np.isfinite(values)

        uncertainties: dict[str, NDArray[np.float64]] = {}
        for name, values in (self.uncertainties or {}).items():
            array = np.asarray(values, dtype=float)
            if name not in channels:
                raise ValueError(f"uncertainty channel {name!r} is not present")
            if array.shape != shape:
                raise ValueError(f"uncertainty for {name!r} must have shape {shape}")
            uncertainties[name] = array

        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "uncertainties", uncertainties)
        object.__setattr__(self, "axis_units", dict(self.axis_units))
        object.__setattr__(self, "channel_units", dict(self.channel_units))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def axis_order(self) -> tuple[str, ...]:
        return tuple(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return next(iter(self.channels.values())).shape

    @property
    def is_primary_complete(self) -> bool:
        return bool(np.all(self.masks[self.primary_channel]))

    def site_spectra(
        self,
        *,
        site_axis: str = "site",
        energy_axis: str = "bias",
        channel: str | None = None,
        require_complete: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(energy, spectra)`` with spectra shaped ``(sites, energy)``."""
        selected = channel or self.primary_channel
        if set(self.axes) != {site_axis, energy_axis} or len(self.axes) != 2:
            raise ValueError(
                f"site spectroscopy requires exactly axes {site_axis!r}, {energy_axis!r}"
            )
        if selected not in self.channels:
            raise KeyError(f"measurement has no channel {selected!r}")
        order = self.axis_order
        values = self.channels[selected]
        mask = self.masks[selected]
        if order == (energy_axis, site_axis):
            values = values.T
            mask = mask.T
        elif order != (site_axis, energy_axis):  # pragma: no cover - protected by set check
            raise ValueError("unsupported axis order")
        if require_complete and not np.all(mask):
            missing = int(mask.size - np.count_nonzero(mask))
            raise ValueError(
                f"channel {selected!r} has {missing} missing values; choose an import "
                "missing-data policy before inference"
            )
        energy = np.asarray(self.axes[energy_axis], dtype=float)
        return energy, np.asarray(values, dtype=float)

    def to_spectroscopy_csv(
        self,
        path: str | Path,
        *,
        include_auxiliary: bool = True,
        allow_missing: bool = False,
    ) -> Path:
        """Write the package's long-form ``site,bias_meV,didv_A`` schema."""
        energy, primary = self.site_spectra(require_complete=not allow_missing)
        sites = np.asarray(self.axes["site"])
        channel_names = [self.primary_channel]
        if include_auxiliary:
            channel_names.extend(name for name in self.channels if name != self.primary_channel)
        column_names = ["site", "bias_meV"] + [
            _spectroscopy_column(name, self.channel_units.get(name, "arbitrary"))
            for name in channel_names
        ]
        channel_values: dict[str, NDArray[np.float64]] = {}
        channel_masks: dict[str, NDArray[np.bool_]] = {}
        for name in channel_names:
            _, values = self.site_spectra(channel=name, require_complete=False)
            mask = self.masks[name]
            if self.axis_order == ("bias", "site"):
                mask = mask.T
            channel_values[name] = values
            channel_masks[name] = mask

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(column_names)
            for site_index, site in enumerate(sites):
                for energy_index, bias in enumerate(energy):
                    row: list[Any] = [_json_safe(site), float(bias)]
                    for name in channel_names:
                        if channel_masks[name][site_index, energy_index]:
                            row.append(float(channel_values[name][site_index, energy_index]))
                        else:
                            row.append("")
                    writer.writerow(row)
        return destination

    def save(self, path: str | Path) -> Path:
        """Save a portable compressed NPZ without Python object pickles."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        for index, (name, values) in enumerate(self.axes.items()):
            payload[f"axis_{index}"] = values.astype(str) if values.dtype.kind == "O" else values
        for index, name in enumerate(self.channels):
            payload[f"channel_{index}"] = self.channels[name]
            payload[f"mask_{index}"] = self.masks[name]
            if name in self.uncertainties:
                payload[f"uncertainty_{index}"] = self.uncertainties[name]
        manifest = {
            "format": "hamiltonian-learning-measurement-v1",
            "toolkit": brand_manifest(),
            "axes": list(self.axes),
            "channels": list(self.channels),
            "axis_units": self.axis_units,
            "channel_units": self.channel_units,
            "primary_channel": self.primary_channel,
            "uncertainty_channels": list(self.uncertainties),
            "metadata": _json_safe(self.metadata),
        }
        payload["manifest_json"] = np.asarray(json.dumps(manifest))
        np.savez_compressed(destination, **payload)
        return destination

    def plot_spectroscopy(self, channels: list[str] | tuple[str, ...] | None = None):
        """Plot site/bias maps for the primary and optional auxiliary channels."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("plotting requires: pip install 'hamlet-toolkit[io]'") from exc
        selected = tuple(channels or self.channels.keys())
        if not selected:
            raise ValueError("at least one channel must be selected")
        figure, axes = plt.subplots(
            1, len(selected), figsize=(6 * len(selected), 4), squeeze=False, constrained_layout=True
        )
        for axis, name in zip(axes[0], selected):
            bias, values = self.site_spectra(channel=name, require_complete=False)
            masked = np.ma.masked_where(
                ~np.isfinite(values), values
            )
            image = axis.pcolormesh(
                bias, np.arange(values.shape[0]), masked, shading="auto"
            )
            role = "inference" if name == self.primary_channel else "plot/QC only"
            axis.set(
                title=f"{name} ({role})",
                xlabel=f"bias [{self.axis_units.get('bias', '')}]",
                ylabel="site order",
            )
            figure.colorbar(
                image,
                ax=axis,
                label=f"{name} [{self.channel_units.get(name, 'arbitrary')}]",
            )
        return figure

    @classmethod
    def load(cls, path: str | Path) -> "Measurement":
        with np.load(path, allow_pickle=False) as payload:
            manifest = json.loads(str(payload["manifest_json"]))
            if manifest.get("format") != "hamiltonian-learning-measurement-v1":
                raise ValueError("unsupported measurement file format")
            axes = {
                name: payload[f"axis_{index}"]
                for index, name in enumerate(manifest["axes"])
            }
            channels = {
                name: payload[f"channel_{index}"]
                for index, name in enumerate(manifest["channels"])
            }
            masks = {
                name: payload[f"mask_{index}"]
                for index, name in enumerate(manifest["channels"])
            }
            uncertainties = {
                name: payload[f"uncertainty_{manifest['channels'].index(name)}"]
                for name in manifest.get("uncertainty_channels", [])
            }
        return cls(
            axes=axes,
            channels=channels,
            masks=masks,
            uncertainties=uncertainties,
            axis_units=manifest.get("axis_units", {}),
            channel_units=manifest.get("channel_units", {}),
            primary_channel=manifest["primary_channel"],
            metadata=manifest.get("metadata", {}),
        )


def _spectroscopy_column(channel: str, unit: str) -> str:
    if channel == "didv":
        return "didv_A" if unit == "A" else f"didv_{unit}"
    if channel == "d2idv2":
        return "d2idv2_A" if unit == "A" else f"d2idv2_{unit}"
    return f"{channel}_{unit}"
