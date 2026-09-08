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


_SPIN_MAGNITUDES = {
    "S=1/2": 0.5,
    "S=1": 1.0,
    "S=3/2": 1.5,
    "S=2": 2.0,
    "S=5/2": 2.5,
}


def spin_multiplicity(label: str) -> int:
    """States per site, ``2S + 1``.

    Exposed because the product over sites is the exact-diagonalisation cost,
    and choosing between ED and DMRG has to be based on it rather than on the
    site count: eight spin-1/2 sites is a 256-state problem, but eight sites
    with three spin-1 impurities is 3456.
    """
    try:
        magnitude = _SPIN_MAGNITUDES[label]
    except KeyError:
        raise ValueError(
            f"unknown spin {label!r}; expected one of {sorted(_SPIN_MAGNITUDES)}"
        ) from None
    return int(round(2.0 * magnitude + 1.0))


@dataclass(frozen=True)
class SiteImpurity:
    """One substituted site with its own spin and single-ion anisotropy.

    ``transverse_mev`` is the ``E`` coefficient of an anisotropy whose hard/easy
    axes lie in the xy-plane, oriented at ``transverse_angle_rad`` from ``x``::

        E [cos(2 phi) ((S^x)^2 - (S^y)^2) + sin(2 phi) (S^x S^y + S^y S^x)]

    The angle is the physical orientation of the impurity's anisotropy in the
    lattice, and *relative* angles between impurities are what matter for
    exposing ``D_z`` -- see :class:`HomogeneousXXZDMIImpurityChain`.

    ``axial_mev`` is the ``D`` coefficient of ``(S^z)^2``. It commutes with
    total ``S^z`` and so can never expose ``D_z`` by itself, but it is included
    because real adatoms have it and it shifts the spectrum.
    """

    site: int
    spin: str = "S=1"
    axial_mev: float = 0.0
    transverse_mev: float = 0.0
    transverse_angle_rad: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.site, bool) or not isinstance(self.site, (int, np.integer)):
            raise ValueError("impurity site must be an integer")
        if self.spin not in _SPIN_MAGNITUDES:
            raise ValueError(f"spin must be one of {sorted(_SPIN_MAGNITUDES)}")
        for name in ("axial_mev", "transverse_mev", "transverse_angle_rad"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        # Single-ion anisotropy is a constant for S=1/2, so requesting it there
        # is a physics error rather than a harmless no-op: it would silently
        # fail to break the symmetry the impurity is present to break.
        if _SPIN_MAGNITUDES[self.spin] < 1.0 and (self.axial_mev or self.transverse_mev):
            raise ValueError(
                "single-ion anisotropy vanishes for a spin-1/2 impurity; "
                "use S=1 or higher to break the U(1) symmetry that hides D_z"
            )
        object.__setattr__(self, "site", int(self.site))
        for name in ("axial_mev", "transverse_mev", "transverse_angle_rad"):
            object.__setattr__(self, name, float(getattr(self, name)))


@dataclass(frozen=True)
class HomogeneousXXZDMIImpurityChain:
    """XXZ+J2+J3 chain with uniform z-axis DMI and substituted impurity sites.

    Parameters are ``(J1_xy, J2, J3, Jz, D_z)`` as in
    :class:`HomogeneousXXZDMIFieldChain`. The impurities replace the transverse
    field as the mechanism that makes ``D_z`` measurable, and like the field
    they are **known conditions**: an isolated adatom's anisotropy is measured
    by single-atom spectroscopy before the chain is assembled, so it belongs to
    the dataset contract rather than to the target vector.

    Why most impurities do not help
    -------------------------------
    The rotation that hides a DM vector along z is generated by ``S^z``::

        U = prod_j exp(-i theta_j S^z_j),      S^±_j -> exp(-i theta_j) S^±_j

    For collinear DMI the nearest-neighbour transverse exchange is a complex
    hopping ``t = (J1_xy - i D_z) / 2``, and on an *open* chain the phases
    ``theta_j`` can be chosen to remove that phase bond by bond -- for
    arbitrary per-bond values, and for any spin magnitude, since ``S^±``
    transforms identically at every site. Hence:

    * Breaking translational symmetry does **not** help. Bond-disordered
      ``D_z``, a vacancy, or a chain cut into segments all remain gaugeable.
    * A **different spin magnitude** does not help, nor does axial anisotropy
      ``D (S^z)^2``, which commutes with total ``S^z``.

    Transverse anisotropy ``E`` does break U(1). But **one** such impurity is
    still not enough, which is measured rather than assumed: ``U`` rotates that
    impurity's anisotropy axis in the xy-plane by the single angle
    ``theta_s``, and the global rotation ``R_z(-theta_s)`` rotates it back
    while leaving the XY exchange, ``Jz`` and ``D_z`` untouched. So
    ``H(J1_xy, D_z, E)`` is unitarily equivalent to ``H(J'_xy, 0, E)`` and
    ``D_z`` stays hidden from any xy-isotropic observable.

    What does work
    --------------
    ``U`` rotates each impurity by its **own** angle ``theta_s = s * alpha``, so
    with two impurities at different sites their *relative* in-plane
    orientation changes, and no single global rotation can restore both. Two
    transverse-anisotropy impurities at distinct sites therefore expose ``D_z``
    to the ordinary unpolarised observable. This is why the transverse field
    works too: it is rotated into a spiral, which is likewise not globally
    undoable.
    """

    n_sites: int
    parameters_mev: ArrayLike
    impurities: tuple[SiteImpurity, ...] = ()
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
        impurities = tuple(self.impurities)
        sites = [impurity.site for impurity in impurities]
        if any(not 0 <= site < self.n_sites for site in sites):
            raise ValueError("every impurity site must index a site of the chain")
        if len(set(sites)) != len(sites):
            raise ValueError("impurity sites must be distinct")
        object.__setattr__(self, "parameters_mev", values.copy())
        object.__setattr__(self, "impurities", impurities)
        object.__setattr__(
            self, "transverse_field_mev", float(self.transverse_field_mev)
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        # The impurities are known conditions recorded with the dataset.
        return ("J1_xy", "J2", "J3", "Jz", "D_z_magnitude")

    @property
    def site_spins(self) -> tuple[str, ...]:
        """Per-site spin magnitudes, spin-1/2 except at substituted sites."""
        spins = ["S=1/2"] * self.n_sites
        for impurity in self.impurities:
            spins[impurity.site] = impurity.spin
        return tuple(spins)

    @property
    def n_transverse_impurities(self) -> int:
        """How many impurities actually break the U(1) symmetry."""
        return sum(1 for imp in self.impurities if imp.transverse_mev)

    @property
    def exposes_dmi(self) -> bool:
        """Whether this configuration can constrain ``D_z`` at all.

        Either mechanism suffices, and they are independent: a transverse field
        is rotated into a spiral, while two transverse impurities at distinct
        sites have their relative in-plane orientation changed. A single
        transverse impurity with no field is *not* enough, because one
        in-plane axis is restored by a global rotation about z; see the class
        docstring.
        """
        return bool(self.transverse_field_mev) or self.n_transverse_impurities >= 2

    @property
    def gauge_invariant_j1_mev(self) -> float:
        """``sqrt(J1_xy^2 + D_z^2)``, all that a measurement blind to ``D_z``
        can constrain."""
        values = np.asarray(self.parameters_mev, dtype=float)
        return float(np.hypot(values[0], values[4]))

    def as_array(self) -> NDArray[np.float64]:
        return np.asarray(self.parameters_mev, dtype=np.float64).copy()


@dataclass(frozen=True)
class HomogeneousXXZDMIImpurityFamily:
    """Sampler for the DMI research model with a fixed impurity configuration.

    As with the transverse-field family, one configuration means one dataset
    and one artifact: a model trained with a given impurity species and
    placement is not reused for another.
    """

    system_type: ClassVar[str] = "homogeneous_xxz_j1j2j3_dmi_impurity"
    n_sites: int
    parameter_ranges_mev: tuple[tuple[float, float], ...]
    impurities: tuple[SiteImpurity, ...] = ()
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
        # Build one chain so the impurity configuration is validated up front
        # rather than partway through a long generation run.
        HomogeneousXXZDMIImpurityChain(
            self.n_sites,
            [low for low, _ in self.parameter_ranges_mev],
            impurities=tuple(self.impurities),
            transverse_field_mev=self.transverse_field_mev,
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("J1_xy", "J2", "J3", "Jz", "D_z_magnitude")

    def sample(self, rng: np.random.Generator) -> HomogeneousXXZDMIImpurityChain:
        values = [rng.uniform(low, high) for low, high in self.parameter_ranges_mev]
        return HomogeneousXXZDMIImpurityChain(
            self.n_sites,
            values,
            impurities=tuple(self.impurities),
            transverse_field_mev=self.transverse_field_mev,
        )
