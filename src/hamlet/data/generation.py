"""Reproducible generation independent of the physics backend."""

from typing import Callable, Protocol, TypeAlias

import numpy as np

from .dataset import SpectroscopyDataset
from ..simulation.base import HeisenbergSystem, SpectroscopyProtocol, SpectroscopySimulator

ProgressCallback: TypeAlias = Callable[[int, int], None]


class SystemFamily(Protocol):
    n_sites: int
    system_type: str

    @property
    def parameter_names(self) -> tuple[str, ...]: ...

    def sample(self, rng: np.random.Generator) -> HeisenbergSystem: ...


def generate_dataset(
    family: SystemFamily,
    simulator: SpectroscopySimulator,
    protocol: SpectroscopyProtocol,
    n_samples: int,
    seed: int | None = None,
    progress: ProgressCallback | None = None,
) -> SpectroscopyDataset:
    """Sample Hamiltonians and simulate their site-resolved spectra."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    spectra = []
    targets = []
    expected_bias = np.asarray(protocol.bias_mev)
    system_type = ""
    for index in range(n_samples):
        system = family.sample(rng)
        system_type = getattr(family, "system_type", type(system).__name__)
        result = simulator.simulate(system, protocol)
        if result.spectral_map.shape[0] != family.n_sites:
            raise ValueError("simulator returned the wrong number of sites")
        if not np.allclose(result.bias_mev, expected_bias):
            raise ValueError("simulator returned a bias grid different from the protocol")
        spectra.append(result.spectral_map)
        targets.append(system.as_array())
        if progress is not None:
            progress(index + 1, n_samples)

    return SpectroscopyDataset(
        spectra=np.stack(spectra),
        targets_mev=np.stack(targets),
        bias_mev=expected_bias,
        target_names=family.parameter_names,
        system_type=system_type,
        metadata={
            "seed": seed,
            "n_samples": n_samples,
            "protocol": protocol.to_metadata(),
            "simulator": type(simulator).__name__,
            "family": type(family).__name__,
        },
    )
