"""Optional DMRGPy spectroscopy backend."""

from dataclasses import dataclass

import numpy as np

from .base import HeisenbergSystem, SpectroscopyProtocol, SpectroscopyResult
from ..systems.heisenberg import (
    HomogeneousHeisenbergChain,
    HomogeneousXXZLongRangeChain,
    HomogeneousXXZDMIFieldChain,
    HomogeneousXXZDMILongRangeChain,
    InhomogeneousHeisenbergChain,
)

DMRGPY_ENERGY_UNIT_MEV = 10.0


def mev_to_dmrgpy_energy(values):
    """Convert physical meV to the DMRGPy convention used by this project."""
    return np.asarray(values, dtype=float) / DMRGPY_ENERGY_UNIT_MEV


def dmrgpy_energy_to_mev(values):
    """Convert DMRGPy energy units to physical meV."""
    return np.asarray(values, dtype=float) * DMRGPY_ENERGY_UNIT_MEV


@dataclass(frozen=True)
class DmrgpySimulator:
    max_bond_dimension: int = 20
    kpm_max_bond_dimension: int = 20
    max_relative_imaginary_residue: float = 1e-6
    dynamics_mode: str = "DMRG"

    def __post_init__(self) -> None:
        if self.max_bond_dimension < 1 or self.kpm_max_bond_dimension < 1:
            raise ValueError("DMRGPy bond dimensions must be positive")
        if not 0.0 < self.max_relative_imaginary_residue < 1.0:
            raise ValueError("max_relative_imaginary_residue must lie between zero and one")
        if self.dynamics_mode not in {"DMRG", "ED"}:
            raise ValueError("dynamics_mode must be 'DMRG' or 'ED'")

    def simulate(
        self, system: HeisenbergSystem, protocol: SpectroscopyProtocol
    ) -> SpectroscopyResult:
        try:
            from dmrgpy import spinchain
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DMRGPy simulation requires the 'simulation' optional dependency"
            ) from exc

        chain = spinchain.Spin_Chain(["S=1/2"] * system.n_sites)
        hamiltonian = 0
        if isinstance(system, InhomogeneousHeisenbergChain):
            interactions = (
                (i, i + 1, float(mev_to_dmrgpy_energy(coupling)))
                for i, coupling in enumerate(system.as_array())
            )
        elif isinstance(system, HomogeneousHeisenbergChain):
            interactions = (
                (i, i + distance, float(mev_to_dmrgpy_energy(coupling)))
                for distance, coupling in enumerate(system.as_array(), start=1)
                for i in range(system.n_sites - distance)
            )
        elif isinstance(system, HomogeneousXXZLongRangeChain):
            j1_xy, j2, j3, jz = mev_to_dmrgpy_energy(system.as_array())
            for i in range(system.n_sites - 1):
                hamiltonian += j1_xy * (
                    chain.Sx[i] * chain.Sx[i + 1]
                    + chain.Sy[i] * chain.Sy[i + 1]
                )
                hamiltonian += jz * chain.Sz[i] * chain.Sz[i + 1]
            for distance, coupling in ((2, j2), (3, j3)):
                for i in range(system.n_sites - distance):
                    hamiltonian += coupling * (
                        chain.Sx[i] * chain.Sx[i + distance]
                        + chain.Sy[i] * chain.Sy[i + distance]
                        + chain.Sz[i] * chain.Sz[i + distance]
                    )
            interactions = ()
        elif isinstance(system, HomogeneousXXZDMILongRangeChain):
            j1_xy, j2, j3, jz, d_z = mev_to_dmrgpy_energy(system.as_array())
            for i in range(system.n_sites - 1):
                hamiltonian += j1_xy * (
                    chain.Sx[i] * chain.Sx[i + 1]
                    + chain.Sy[i] * chain.Sy[i + 1]
                )
                hamiltonian += jz * chain.Sz[i] * chain.Sz[i + 1]
                hamiltonian += d_z * (
                    chain.Sx[i] * chain.Sy[i + 1]
                    - chain.Sy[i] * chain.Sx[i + 1]
                )
            for distance, coupling in ((2, j2), (3, j3)):
                for i in range(system.n_sites - distance):
                    hamiltonian += coupling * (
                        chain.Sx[i] * chain.Sx[i + distance]
                        + chain.Sy[i] * chain.Sy[i + distance]
                        + chain.Sz[i] * chain.Sz[i + distance]
                    )
            interactions = ()
        elif isinstance(system, HomogeneousXXZDMIFieldChain):
            j1_xy, j2, j3, jz, d_z = mev_to_dmrgpy_energy(system.as_array())
            b_x = float(mev_to_dmrgpy_energy(system.transverse_field_mev))
            for i in range(system.n_sites - 1):
                hamiltonian += j1_xy * (
                    chain.Sx[i] * chain.Sx[i + 1]
                    + chain.Sy[i] * chain.Sy[i + 1]
                )
                hamiltonian += jz * chain.Sz[i] * chain.Sz[i + 1]
                hamiltonian += d_z * (
                    chain.Sx[i] * chain.Sy[i + 1]
                    - chain.Sy[i] * chain.Sx[i + 1]
                )
            for distance, coupling in ((2, j2), (3, j3)):
                for i in range(system.n_sites - distance):
                    hamiltonian += coupling * (
                        chain.Sx[i] * chain.Sx[i + distance]
                        + chain.Sy[i] * chain.Sy[i + distance]
                        + chain.Sz[i] * chain.Sz[i + distance]
                    )
            # Zeeman term -B_x sum_i S^x_i, with g and mu_B absorbed into a
            # field already expressed as an energy in meV. The axis is x, i.e.
            # transverse to the DM vector along z, which is the whole point: a
            # field along z is invariant under the rotation that removes a
            # uniform D_z and would leave it just as unmeasurable as at zero
            # field.
            if b_x:
                for i in range(system.n_sites):
                    hamiltonian += -b_x * chain.Sx[i]
            interactions = ()
        else:
            raise TypeError(f"unsupported system type: {type(system).__name__}")

        for left, right, coupling in interactions:
            hamiltonian += coupling * (
                chain.Sx[left] * chain.Sx[right]
                + chain.Sy[left] * chain.Sy[right]
                + chain.Sz[left] * chain.Sz[right]
            )
        chain.set_hamiltonian(hamiltonian)
        chain.maxm = self.max_bond_dimension
        chain.kpmmaxm = self.kpm_max_bond_dimension

        spectra = []
        output_bias = np.asarray(protocol.bias_mev, dtype=float)
        backend_bias = mev_to_dmrgpy_energy(output_bias)
        backend_broadening = float(mev_to_dmrgpy_energy(protocol.broadening_mev))
        for site in range(system.n_sites):
            operators = (chain.Sx[site], chain.Sy[site], chain.Sz[site])
            values = np.zeros_like(backend_bias, dtype=float)
            for weight, operator in zip(
                protocol.resolved_observable_weights, operators
            ):
                if weight == 0.0:
                    continue
                x, component = chain.get_dynamical_correlator(
                    name=(operator, operator),
                    es=backend_bias,
                    delta=backend_broadening,
                    mode=self.dynamics_mode,
                )
                component = np.asarray(component)
                if np.iscomplexobj(component):
                    imaginary_scale = float(np.max(np.abs(component.imag)))
                    real_scale = max(float(np.max(np.abs(component.real))), 1.0)
                    # Finite-MPS/KPM truncation leaves a small imaginary residue in
                    # Hermitian self-correlators. Research runs keep the strict
                    # default; deliberately coarse pilots may opt into a looser limit.
                    if imaginary_scale > self.max_relative_imaginary_residue * real_scale:
                        raise ValueError(
                            "DMRGPy returned a materially complex spectral function "
                            f"(relative imaginary residue {imaginary_scale / real_scale:.3g})"
                        )
                    component = component.real
                x = np.asarray(x, dtype=float)
                component = np.asarray(component, dtype=float)
                if not np.array_equal(x, backend_bias):
                    component = np.interp(backend_bias, x, component)
                values += float(weight) * component
            if protocol.output_quantity == "didv":
                # Match the reference pipeline, which integrates in DMRGPy units.
                increments = 0.5 * (values[1:] + values[:-1]) * np.diff(backend_bias)
                values = np.concatenate(([0.0], np.cumsum(increments)))
            spectra.append(values)
        return SpectroscopyResult(output_bias, np.stack(spectra))
