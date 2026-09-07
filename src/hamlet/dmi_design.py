"""Screen sample designs for whether they can expose DMI at all.

A DM vector along ``z`` is unidentifiable in any Hamiltonian conserving total
``S^z``: the transverse exchange is a complex hopping ``(J1_xy - i D_z) / 2``
whose phase a site-dependent rotation about ``z`` removes bond by bond on an
open chain. Only ``sqrt(J1_xy^2 + D_z^2)`` is measurable, never ``D_z`` alone.

Some perturbation must therefore break that symmetry, and whether a given one
breaks it *enough* is a quantitative question rather than a matter of judgement.
This module answers it directly for an arbitrary design, by simulating a **gauge
pair**: two chains that share ``sqrt(J1_xy^2 + D_z^2)`` and differ only in how
it splits between exchange and DMI. If a design cannot separate that pair, no
model trained on it can recover ``D_z``, however much data is generated.

Use this before committing simulation time or beam time. It costs two
simulations per candidate, against thousands of chains for a training set.

The verdict thresholds are calibrated against trained models rather than chosen
arbitrarily; see :data:`IMPRINT_CALIBRATION`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .simulation import SpectroscopyProtocol
from .systems import HomogeneousXXZDMIImpurityChain, SiteImpurity

# Measured reference points, each an imprint paired with the D_z skill a model
# trained on that design actually reached. These anchor the verdicts below.
IMPRINT_CALIBRATION: tuple[tuple[float, str], ...] = (
    (1e-14, "a symmetric chain: exactly degenerate, D_z unrecoverable"),
    (4.4e-02, "J2/J3-mediated breaking: trained to ~0.00 D_z skill"),
    (1.18e-01, "a 1 meV transverse field: trained to ~0.19 D_z skill, and flat with more data"),
    (2.29e-01, "three transverse-anisotropy impurities: trained to 0.54 D_z skill"),
)

# Below this the pair is degenerate to numerical precision, i.e. the symmetry is
# unbroken and the design is hopeless no matter how much data is generated.
HIDDEN_IMPRINT = 1e-8
# At or below the level that was measured to train to zero skill.
TOO_WEAK_IMPRINT = 5e-02
# At or above the level that trained to ~0.19, which plateaued.
PROMISING_IMPRINT = 1.1e-01
# At or above the level that trained to 0.54.
STRONG_IMPRINT = 2.0e-01


@dataclass(frozen=True)
class DmiDesign:
    """One candidate sample design to screen.

    ``impurities`` and ``transverse_field_mev`` are the symmetry-breaking
    mechanisms; everything else describes the host chain. ``label`` is free text
    used for reporting.
    """

    n_sites: int
    j_eff_mev: float
    d_z_mev: float
    jz_mev: float
    impurities: tuple[SiteImpurity, ...] = ()
    transverse_field_mev: float = 0.0
    j2_mev: float = 0.0
    j3_mev: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.d_z_mev <= self.j_eff_mev:
            # The pair is built by trading D_z against exchange at fixed
            # hypotenuse, so D_z cannot exceed it.
            raise ValueError(
                "d_z_mev must be positive and no greater than j_eff_mev, which "
                "is the shared sqrt(J1_xy^2 + D_z^2) of the gauge pair"
            )

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        sites = ",".join(str(imp.site) for imp in self.impurities) or "none"
        return f"L{self.n_sites} imp[{sites}] B={self.transverse_field_mev:g}"

    def gauge_pair(self) -> tuple[HomogeneousXXZDMIImpurityChain, ...]:
        """The two chains that a DMI-blind measurement cannot tell apart.

        Both share ``sqrt(J1_xy^2 + D_z^2) = j_eff_mev``; one puts all of it in
        the exchange, the other splits off ``d_z_mev`` into DMI.
        """
        j1_alt = float(np.sqrt(self.j_eff_mev**2 - self.d_z_mev**2))
        return tuple(
            HomogeneousXXZDMIImpurityChain(
                self.n_sites,
                [j1, self.j2_mev, self.j3_mev, self.jz_mev, d_z],
                impurities=self.impurities,
                transverse_field_mev=self.transverse_field_mev,
            )
            for j1, d_z in ((self.j_eff_mev, 0.0), (j1_alt, self.d_z_mev))
        )

    @property
    def breaks_symmetry(self) -> bool:
        """Whether the symmetry argument alone predicts any imprint.

        This is free -- no simulation -- and rules out hopeless designs
        immediately. It cannot say whether a surviving design is *strong
        enough*, which is what :func:`measure_dmi_imprint` measures.
        """
        return self.gauge_pair()[0].exposes_dmi


@dataclass(frozen=True)
class DmiImprint:
    """How well one design separates a gauge pair."""

    design: DmiDesign
    imprint: float
    verdict: str
    predicted_to_break_symmetry: bool
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def can_expose_dmi(self) -> bool:
        """True when the pair is separated beyond numerical precision.

        Necessary, not sufficient: a separated pair may still be too weak to
        learn from. Check :attr:`verdict`.
        """
        return self.imprint >= HIDDEN_IMPRINT


def classify_imprint(imprint: float) -> str:
    """Turn an imprint into a verdict calibrated on trained models."""
    if not np.isfinite(imprint):
        raise ValueError("imprint must be finite")
    if imprint < HIDDEN_IMPRINT:
        return "hidden"
    if imprint < TOO_WEAK_IMPRINT:
        return "too weak"
    if imprint < PROMISING_IMPRINT:
        return "marginal"
    if imprint < STRONG_IMPRINT:
        return "promising"
    return "strong"


def measure_dmi_imprint(
    design: DmiDesign,
    simulator: Any,
    protocol: SpectroscopyProtocol,
) -> DmiImprint:
    """Simulate a design's gauge pair and report how far apart it is.

    The imprint is the largest absolute difference between the two spectral
    maps, relative to the scale of the first, so it is comparable across
    designs and to :data:`IMPRINT_CALIBRATION`.
    """
    first, second = design.gauge_pair()
    a = np.asarray(simulator.simulate(first, protocol).spectral_map, dtype=float)
    b = np.asarray(simulator.simulate(second, protocol).spectral_map, dtype=float)
    if a.shape != b.shape:
        raise ValueError("gauge-pair simulations returned different shapes")
    scale = float(np.abs(a).max())
    if not scale > 0.0:
        raise ValueError("reference simulation is identically zero")
    imprint = float(np.abs(a - b).max() / scale)
    return DmiImprint(
        design=design,
        imprint=imprint,
        verdict=classify_imprint(imprint),
        predicted_to_break_symmetry=design.breaks_symmetry,
        detail={
            "j1_xy_mev": (design.j_eff_mev, float(second.as_array()[0])),
            "d_z_mev": (0.0, design.d_z_mev),
            "shared_gauge_invariant_mev": design.j_eff_mev,
        },
    )


def screen_dmi_designs(
    designs: Iterable[DmiDesign],
    simulator: Any,
    protocol: SpectroscopyProtocol,
    *,
    skip_symmetric: bool = True,
    progress: Callable[[DmiImprint], None] | None = None,
) -> tuple[DmiImprint, ...]:
    """Rank candidate designs by how well each exposes ``D_z``, best first.

    ``skip_symmetric`` avoids simulating designs the symmetry argument already
    rules out, which is the common case when sweeping impurity counts and
    positions: those are reported with a zero imprint and the ``hidden``
    verdict without touching the simulator. Set it False to verify the
    prediction rather than trust it.
    """
    results: list[DmiImprint] = []
    for design in designs:
        if skip_symmetric and not design.breaks_symmetry:
            result = DmiImprint(
                design=design,
                imprint=0.0,
                verdict="hidden",
                predicted_to_break_symmetry=False,
                detail={"skipped": "no U(1)-breaking mechanism; not simulated"},
            )
        else:
            result = measure_dmi_imprint(design, simulator, protocol)
        results.append(result)
        if progress is not None:
            progress(result)
    return tuple(sorted(results, key=lambda item: -item.imprint))


def transverse_impurities(
    sites: Sequence[int],
    transverse_mev: float,
    *,
    spin: str = "S=1",
    axial_mev: float = 0.0,
) -> tuple[SiteImpurity, ...]:
    """Convenience builder for a set of identical transverse-anisotropy sites.

    Transverse anisotropy is what breaks the symmetry; ``axial_mev`` is carried
    through because real adatoms have it and it changes the spectra, not
    because it can expose ``D_z`` on its own.
    """
    return tuple(
        SiteImpurity(
            site, spin, axial_mev=axial_mev, transverse_mev=transverse_mev
        )
        for site in sites
    )


def format_screening_table(results: Sequence[DmiImprint]) -> str:
    """Render screening results, best first, as a fixed-width table."""
    lines = [
        f"{'design':>34s}{'imprint':>11s}  verdict",
        "-" * 62,
    ]
    for item in results:
        lines.append(
            f"{item.design.name:>34s}{item.imprint:11.3e}  {item.verdict}"
        )
    lines.append("")
    lines.append("calibration (imprint -> D_z skill actually achieved):")
    for value, note in IMPRINT_CALIBRATION:
        lines.append(f"  {value:9.2e}  {note}")
    return "\n".join(lines)
