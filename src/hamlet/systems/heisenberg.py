"""Physical specifications for nearest-neighbor Heisenberg chains."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class InhomogeneousHeisenbergChain:
    """Open spin-1/2 chain with one isotropic exchange per bond.

    The represented Hamiltonian is
    ``H = sum_i J[i] (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})``.
    This class describes the physics without depending on a simulator backend.
    """

    couplings_mev: ArrayLike

    def __post_init__(self) -> None:
        couplings = np.asarray(self.couplings_mev, dtype=float)
        if couplings.ndim != 1 or couplings.size < 1:
            raise ValueError("couplings_mev must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(couplings)):
            raise ValueError("couplings_mev must contain only finite values")
        object.__setattr__(self, "couplings_mev", couplings.copy())

    @property
    def n_sites(self) -> int:
        return int(np.asarray(self.couplings_mev).size + 1)

    @property
    def n_bonds(self) -> int:
        return self.n_sites - 1

    @classmethod
    def sample(
        cls,
        n_sites: int,
        coupling_range_mev: tuple[float, float],
        rng: np.random.Generator | None = None,
    ) -> "InhomogeneousHeisenbergChain":
        if n_sites < 2:
            raise ValueError("n_sites must be at least 2")
        low, high = coupling_range_mev
        if not low < high:
            raise ValueError("coupling_range_mev must satisfy low < high")
        generator = rng if rng is not None else np.random.default_rng()
        return cls(generator.uniform(low, high, n_sites - 1))

    def as_array(self) -> NDArray[np.float64]:
        return np.asarray(self.couplings_mev, dtype=np.float64).copy()

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"J_bond_{i}" for i in range(self.n_bonds))


@dataclass(frozen=True)
class HomogeneousHeisenbergChain:
    """Open J1-J2-... chain with each coupling shared at a distance.

    ``couplings_by_distance_mev[r - 1]`` is used for every pair separated by
    distance ``r``. Thus ``[J1, J2]`` means homogeneous nearest- and
    next-nearest-neighbor exchange, not two particular bonds.
    """

    n_sites: int
    couplings_by_distance_mev: ArrayLike

    def __post_init__(self) -> None:
        couplings = np.asarray(self.couplings_by_distance_mev, dtype=float)
        if self.n_sites < 2:
            raise ValueError("n_sites must be at least 2")
        if couplings.ndim != 1 or couplings.size < 1:
            raise ValueError("couplings_by_distance_mev must be a non-empty 1D array")
        if couplings.size >= self.n_sites:
            raise ValueError("interaction distance must be smaller than n_sites")
        if not np.all(np.isfinite(couplings)):
            raise ValueError("couplings must contain only finite values")
        object.__setattr__(self, "couplings_by_distance_mev", couplings.copy())

    @property
    def max_interaction_distance(self) -> int:
        return int(np.asarray(self.couplings_by_distance_mev).size)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"J{i}" for i in range(1, self.max_interaction_distance + 1))

    def as_array(self) -> NDArray[np.float64]:
        return np.asarray(self.couplings_by_distance_mev, dtype=np.float64).copy()


@dataclass(frozen=True)
class InhomogeneousHeisenbergFamily:
    """Sampler for independently distributed nearest-neighbor bond couplings."""

    system_type: ClassVar[str] = "inhomogeneous_heisenberg"
    n_sites: int
    coupling_range_mev: tuple[float, float]

    def __post_init__(self) -> None:
        low, high = self.coupling_range_mev
        if self.n_sites < 3:
            raise ValueError("local inhomogeneous learning requires at least 3 sites")
        if not low < high:
            raise ValueError("coupling_range_mev must satisfy low < high")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"J_bond_{i}" for i in range(self.n_sites - 1))

    def sample(self, rng: np.random.Generator) -> InhomogeneousHeisenbergChain:
        return InhomogeneousHeisenbergChain.sample(
            self.n_sites, self.coupling_range_mev, rng
        )


@dataclass(frozen=True)
class HomogeneousHeisenbergFamily:
    """Sampler for global J1-J2-... couplings shared across a chain."""

    system_type: ClassVar[str] = "homogeneous_heisenberg"
    n_sites: int
    coupling_ranges_mev: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.n_sites < 2:
            raise ValueError("n_sites must be at least 2")
        if not self.coupling_ranges_mev:
            raise ValueError("at least one coupling range is required")
        if len(self.coupling_ranges_mev) >= self.n_sites:
            raise ValueError("interaction distance must be smaller than n_sites")
        if any(not low < high for low, high in self.coupling_ranges_mev):
            raise ValueError("every coupling range must satisfy low < high")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"J{i}" for i in range(1, len(self.coupling_ranges_mev) + 1))

    def sample(self, rng: np.random.Generator) -> HomogeneousHeisenbergChain:
        couplings = [rng.uniform(low, high) for low, high in self.coupling_ranges_mev]
        return HomogeneousHeisenbergChain(self.n_sites, couplings)


@dataclass(frozen=True)
class HomogeneousXXZLongRangeChain:
    """Open homogeneous XXZ nearest-neighbour chain with isotropic J2 and J3.

    The parameter order is ``(J1_xy, J2, J3, Jz)`` and the Hamiltonian is
    ``J1_xy*(SxSx+SySy) + Jz*SzSz`` on nearest neighbours plus isotropic
    ``J2`` and ``J3`` exchange at distances two and three.
    """

    n_sites: int
    parameters_mev: ArrayLike

    def __post_init__(self) -> None:
        values = np.asarray(self.parameters_mev, dtype=float)
        if self.n_sites < 4:
            raise ValueError("J3 interactions require at least 4 sites")
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("parameters_mev must contain finite (J1_xy, J2, J3, Jz)")
        object.__setattr__(self, "parameters_mev", values.copy())

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("J1_xy", "J2", "J3", "Jz")

    def as_array(self) -> NDArray[np.float64]:
        return np.asarray(self.parameters_mev, dtype=np.float64).copy()


@dataclass(frozen=True)
class HomogeneousXXZLongRangeFamily:
    """Sampler for the fixed four-parameter XXZ+J2+J3 pilot model."""

    system_type: ClassVar[str] = "homogeneous_xxz_j1j2j3"
    n_sites: int
    parameter_ranges_mev: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.n_sites < 4:
            raise ValueError("J3 interactions require at least 4 sites")
        if len(self.parameter_ranges_mev) != 4:
            raise ValueError("ranges must be ordered as J1_xy, J2, J3, Jz")
        if any(not low < high for low, high in self.parameter_ranges_mev):
            raise ValueError("every parameter range must satisfy low < high")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("J1_xy", "J2", "J3", "Jz")

    def sample(self, rng: np.random.Generator) -> HomogeneousXXZLongRangeChain:
        values = [rng.uniform(low, high) for low, high in self.parameter_ranges_mev]
        return HomogeneousXXZLongRangeChain(self.n_sites, values)


@dataclass(frozen=True)
class HomogeneousXXZDMILongRangeChain:
    """XXZ+J2+J3 chain with uniform nearest-neighbour z-axis DMI.

    Parameters are ``(J1_xy, J2, J3, Jz, D_z)``. ``D_z`` is represented as a
    non-negative magnitude because unpolarized autocorrelations do not provide
    a reliable handedness/sign contract.
    """

    n_sites: int
    parameters_mev: ArrayLike

    def __post_init__(self) -> None:
        values = np.asarray(self.parameters_mev, dtype=float)
        if self.n_sites < 4:
            raise ValueError("J3 interactions require at least 4 sites")
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "parameters_mev must contain finite (J1_xy, J2, J3, Jz, D_z)"
            )
        if values[4] < 0.0:
            raise ValueError("D_z is a magnitude and must be non-negative")
        object.__setattr__(self, "parameters_mev", values.copy())

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("J1_xy", "J2", "J3", "Jz", "D_z_magnitude")

    def as_array(self) -> NDArray[np.float64]:
        return np.asarray(self.parameters_mev, dtype=np.float64).copy()


@dataclass(frozen=True)
class HomogeneousXXZDMILongRangeFamily:
    """Sampler for the five-target total-spin DMI research model."""

    system_type: ClassVar[str] = "homogeneous_xxz_j1j2j3_dmi"
    n_sites: int
    parameter_ranges_mev: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.n_sites < 4:
            raise ValueError("J3 interactions require at least 4 sites")
        if len(self.parameter_ranges_mev) != 5:
            raise ValueError("ranges must be ordered as J1_xy, J2, J3, Jz, D_z")
        if any(not low < high for low, high in self.parameter_ranges_mev):
            raise ValueError("every parameter range must satisfy low < high")
        if self.parameter_ranges_mev[4][0] < 0.0:
            raise ValueError("D_z magnitude range cannot include negative values")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("J1_xy", "J2", "J3", "Jz", "D_z_magnitude")

    def sample(self, rng: np.random.Generator) -> HomogeneousXXZDMILongRangeChain:
        values = [rng.uniform(low, high) for low, high in self.parameter_ranges_mev]
        return HomogeneousXXZDMILongRangeChain(self.n_sites, values)


@dataclass(frozen=True)
class HomogeneousXXZDMIFieldChain:
    """XXZ+J2+J3 chain with uniform z-axis DMI in a transverse field.

    Parameters are ``(J1_xy, J2, J3, Jz, D_z)`` as in
    :class:`HomogeneousXXZDMILongRangeChain`, plus a **fixed** transverse field
    ``B_x`` that is a known experimental condition rather than something to
    infer.

    The field exists to make ``D_z`` measurable at all. A uniform DM vector
    along z can be removed exactly from a nearest-neighbour XXZ chain by the
    site-dependent rotation ``U = prod_j exp(-i j alpha S^z_j)`` with
    ``tan(alpha) = D_z / J1_xy``, which leaves a plain XXZ chain of
    ``J'_xy = sqrt(J1_xy^2 + D_z^2)``. Because that rotation is about z, and
    because U(1) magnetisation conservation kills the terms that would not be
    invariant, every on-site autocorrelator ``<S^a_j S^a_j>`` is unchanged by
    it. Zero-field spectra of this observable therefore constrain only
    ``J'_xy``: measured directly, a pure nearest-neighbour gauge pair agrees to
    one part in 10^13.

    A field along z would not help, being itself invariant under that rotation.
    ``B_x`` is transverse on purpose: it breaks the U(1) symmetry, so the
    rotation no longer leaves the Hamiltonian invariant and ``D_z`` acquires an
    imprint on the same spectra the package already simulates.
    """

    n_sites: int
    parameters_mev: ArrayLike
    transverse_field_mev: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(self.parameters_mev, dtype=float)
        if self.n_sites < 4:
            raise ValueError("J3 interactions require at least 4 sites")
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "parameters_mev must contain finite (J1_xy, J2, J3, Jz, D_z)"
            )
        if values[4] < 0.0:
            raise ValueError("D_z is a magnitude and must be non-negative")
        if not np.isfinite(self.transverse_field_mev):
            raise ValueError("transverse_field_mev must be finite")
        if self.transverse_field_mev < 0.0:
            raise ValueError(
                "transverse_field_mev is a magnitude along +x and must be non-negative"
            )
        object.__setattr__(self, "parameters_mev", values.copy())
        object.__setattr__(self, "transverse_field_mev", float(self.transverse_field_mev))

    @property
    def parameter_names(self) -> tuple[str, ...]:
        # The field is a known condition recorded with the dataset, not a
        # target, so it does not appear here.
        return ("J1_xy", "J2", "J3", "Jz", "D_z_magnitude")

    @property
    def gauge_invariant_j1_mev(self) -> float:
        """``sqrt(J1_xy^2 + D_z^2)``, the only combination a zero-field
        on-site autocorrelation measurement can constrain."""
        values = np.asarray(self.parameters_mev, dtype=float)
        return float(np.hypot(values[0], values[4]))

    def as_array(self) -> NDArray[np.float64]:
        return np.asarray(self.parameters_mev, dtype=np.float64).copy()


@dataclass(frozen=True)
class HomogeneousXXZDMIFieldFamily:
    """Sampler for the DMI research model at one fixed transverse field.

    One family, and therefore one dataset and one artifact, corresponds to a
    single field strength. This mirrors how the bias cutoff is handled: a model
    trained under one experimental condition is not reused under another, so
    the field belongs to the contract rather than to the target vector.
    """

    system_type: ClassVar[str] = "homogeneous_xxz_j1j2j3_dmi_field"
    n_sites: int
    parameter_ranges_mev: tuple[tuple[float, float], ...]
    transverse_field_mev: float = 0.0

    def __post_init__(self) -> None:
        if self.n_sites < 4:
            raise ValueError("J3 interactions require at least 4 sites")
        if len(self.parameter_ranges_mev) != 5:
            raise ValueError("ranges must be ordered as J1_xy, J2, J3, Jz, D_z")
        if any(not low < high for low, high in self.parameter_ranges_mev):
            raise ValueError("every parameter range must satisfy low < high")
        if self.parameter_ranges_mev[4][0] < 0.0:
            raise ValueError("D_z magnitude range cannot include negative values")
        if not np.isfinite(self.transverse_field_mev) or self.transverse_field_mev < 0.0:
            raise ValueError("transverse_field_mev must be finite and non-negative")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("J1_xy", "J2", "J3", "Jz", "D_z_magnitude")

    def sample(self, rng: np.random.Generator) -> HomogeneousXXZDMIFieldChain:
        values = [rng.uniform(low, high) for low, high in self.parameter_ranges_mev]
        return HomogeneousXXZDMIFieldChain(
            self.n_sites, values, transverse_field_mev=self.transverse_field_mev
        )
