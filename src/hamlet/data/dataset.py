"""Portable dataset container with physical metadata."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..branding import brand_manifest


@dataclass(frozen=True)
class SpectroscopyDataset:
    spectra: ArrayLike
    targets_mev: ArrayLike
    bias_mev: ArrayLike
    target_names: tuple[str, ...]
    system_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spectra = np.asarray(self.spectra, dtype=np.float32)
        targets = np.asarray(self.targets_mev, dtype=np.float32)
        bias = np.asarray(self.bias_mev, dtype=np.float64)
        if spectra.ndim != 3:
            raise ValueError("spectra must have shape (samples, sites, bias)")
        if targets.ndim != 2 or targets.shape[0] != spectra.shape[0]:
            raise ValueError("targets_mev must have shape (samples, parameters)")
        if bias.ndim != 1 or spectra.shape[2] != bias.size:
            raise ValueError("bias_mev must match the last spectra dimension")
        if len(self.target_names) != targets.shape[1]:
            raise ValueError("target_names must name every target column")
        if not self.system_type:
            raise ValueError("system_type cannot be empty")
        if not np.all(np.isfinite(spectra)) or not np.all(np.isfinite(targets)):
            raise ValueError("spectra and targets must contain only finite values")
        object.__setattr__(self, "spectra", spectra.copy())
        object.__setattr__(self, "targets_mev", targets.copy())
        object.__setattr__(self, "bias_mev", bias.copy())
        object.__setattr__(self, "target_names", tuple(self.target_names))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_samples(self) -> int:
        return int(np.asarray(self.spectra).shape[0])

    @property
    def n_sites(self) -> int:
        return int(np.asarray(self.spectra).shape[1])

    def save(self, path: str | Path) -> None:
        payload = {
            "toolkit": brand_manifest(),
            "system_type": self.system_type,
            "target_names": list(self.target_names),
            "metadata": self.metadata,
        }
        np.savez_compressed(
            path,
            spectra=self.spectra,
            targets_mev=self.targets_mev,
            bias_mev=self.bias_mev,
            manifest_json=np.array(json.dumps(payload)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SpectroscopyDataset":
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["manifest_json"]))
            return cls(
                spectra=data["spectra"],
                targets_mev=data["targets_mev"],
                bias_mev=data["bias_mev"],
                target_names=tuple(payload["target_names"]),
                system_type=payload["system_type"],
                metadata=payload["metadata"],
            )
