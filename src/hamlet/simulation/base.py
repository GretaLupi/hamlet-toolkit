"""Backend-independent simulation contracts."""

from dataclasses import dataclass
from typing import Any
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from ..systems.heisenberg import (
    HomogeneousHeisenbergChain,
    HomogeneousXXZLongRangeChain,
    HomogeneousXXZDMILongRangeChain,
    InhomogeneousHeisenbergChain,
)

HeisenbergSystem = (
    InhomogeneousHeisenbergChain
    | HomogeneousHeisenbergChain
    | HomogeneousXXZLongRangeChain
    | HomogeneousXXZDMILongRangeChain
)


@dataclass(frozen=True)
class SpectroscopyProtocol:
    bias_mev: ArrayLike
    broadening_mev: float = 0.5
    observable: str = "Sz"
    observable_weights: tuple[float, float, float] | None = None
    output_quantity: str = "spectral_function"

    def __post_init__(self) -> None:
        bias = np.asarray(self.bias_mev, dtype=float)
        if bias.ndim != 1 or bias.size < 2 or np.any(np.diff(bias) <= 0):
            raise ValueError("bias_mev must be a strictly increasing 1D grid")
        if not np.all(np.isfinite(bias)):
            raise ValueError("bias_mev must contain only finite values")
        if self.broadening_mev <= 0:
            raise ValueError("broadening_mev must be positive")
        if self.observable not in {"Sz", "total_spin"}:
            raise ValueError("observable must be 'Sz' or 'total_spin'")
        if self.observable == "Sz" and self.observable_weights is not None:
            raise ValueError("observable_weights apply only to observable='total_spin'")
        if self.observable_weights is not None:
            weights = np.asarray(self.observable_weights, dtype=float)
            if weights.shape != (3,) or not np.all(np.isfinite(weights)):
                raise ValueError("observable_weights must contain finite (Sxx, Syy, Szz) weights")
            if np.any(weights < 0.0) or not np.any(weights > 0.0):
                raise ValueError("observable_weights must be non-negative and not all zero")
            object.__setattr__(self, "observable_weights", tuple(float(x) for x in weights))
        if self.output_quantity not in ("spectral_function", "didv"):
            raise ValueError("output_quantity must be 'spectral_function' or 'didv'")
        object.__setattr__(self, "bias_mev", bias.copy())

    @property
    def resolved_observable_weights(self) -> tuple[float, float, float]:
        if self.observable == "Sz":
            return (0.0, 0.0, 1.0)
        return self.observable_weights or (1.0, 1.0, 1.0)

    def observable_contract(self) -> dict[str, Any]:
        components = ("Szz",) if self.observable == "Sz" else ("Sxx", "Syy", "Szz")
        weights = (
            (1.0,)
            if self.observable == "Sz"
            else self.resolved_observable_weights
        )
        return {
            "kind": self.observable,
            "components": list(components),
            "weights": list(weights),
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "broadening_mev": self.broadening_mev,
            "observable": self.observable,
            "observable_contract": self.observable_contract(),
            "output_quantity": self.output_quantity,
        }

    @classmethod
    def uniform(
        cls,
        bias_range_mev: tuple[float, float] = (0.0, 100.0),
        points: int = 200,
        **kwargs: object,
    ) -> "SpectroscopyProtocol":
        if points < 2:
            raise ValueError("points must be at least 2")
        low, high = bias_range_mev
        if not low < high:
            raise ValueError("bias_range_mev must satisfy low < high")
        return cls(np.linspace(low, high, points), **kwargs)


@dataclass(frozen=True)
class SpectroscopyResult:
    bias_mev: ArrayLike
    spectral_map: ArrayLike

    def __post_init__(self) -> None:
        bias = np.asarray(self.bias_mev, dtype=float)
        spectra = np.asarray(self.spectral_map, dtype=float)
        if bias.ndim != 1 or spectra.ndim != 2 or spectra.shape[1] != bias.size:
            raise ValueError("spectral_map must have shape (sites, len(bias_mev))")
        if not np.all(np.isfinite(bias)) or not np.all(np.isfinite(spectra)):
            raise ValueError("simulation result must contain only finite values")
        object.__setattr__(self, "bias_mev", bias.copy())
        object.__setattr__(self, "spectral_map", spectra.copy())


@runtime_checkable
class SpectroscopySimulator(Protocol):
    def simulate(
        self, system: HeisenbergSystem, protocol: SpectroscopyProtocol
    ) -> SpectroscopyResult: ...
