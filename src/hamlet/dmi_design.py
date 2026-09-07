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
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

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


SCREENING_SCHEMA_VERSION = 1

_ALLOWED_IMPURITY_KEYS = {
    "site",
    "spin",
    "axial_mev",
    "transverse_mev",
    "transverse_angle_rad",
}
_ALLOWED_TOP_KEYS = {
    "screening_schema_version",
    "name",
    "chain",
    "protocol",
    "candidates",
    "sweep",
}
_ALLOWED_CHAIN_KEYS = {
    "n_sites",
    "j_eff_mev",
    "d_z_mev",
    "jz_mev",
    "j2_mev",
    "j3_mev",
}
_ALLOWED_PROTOCOL_KEYS = {
    "bias_range_mev",
    "bias_points",
    "broadening_mev",
    "observable",
    "observable_weights",
    "output_quantity",
}
_ALLOWED_SWEEP_KEYS = {
    "sites",
    "transverse_mev",
    "spin",
    "axial_mev",
    "transverse_angle_rad",
    "transverse_field_mev",
}


def _require_mapping(value: Any, what: str) -> "Mapping[str, Any]":
    from collections.abc import Mapping as _Mapping

    if not isinstance(value, _Mapping):
        raise ValueError(f"{what} must be a mapping")
    return value


def _reject_unknown(payload: "Mapping[str, Any]", allowed: set[str], what: str) -> None:
    # Unknown keys are rejected rather than ignored: a typo in an impurity
    # position or anisotropy would otherwise silently screen a different design
    # from the one the user wrote down.
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"{what} has unknown fields: {sorted(unknown)}")


def _build_impurity(payload: Any, what: str) -> SiteImpurity:
    mapping = _require_mapping(payload, what)
    _reject_unknown(mapping, _ALLOWED_IMPURITY_KEYS, what)
    if "site" not in mapping:
        raise ValueError(f"{what} requires site")
    return SiteImpurity(**mapping)


def load_screening_config(
    path: str | Path,
) -> tuple[tuple[DmiDesign, ...], SpectroscopyProtocol]:
    """Read candidate designs and a measurement protocol from YAML or JSON.

    Screening is a design decision taken before any code is written, so it is
    available from a configuration file rather than only the Python API. The
    file declares the host chain once, the measurement protocol once, and then
    either explicit ``candidates`` or a ``sweep`` that expands into them.

    See ``examples/dmi_screening.yaml``.
    """
    import json
    from pathlib import Path as _Path

    config_path = _Path(path).resolve()
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("YAML configuration requires PyYAML") from exc
        payload = yaml.safe_load(text)

    payload = _require_mapping(payload, "screening configuration")
    _reject_unknown(payload, _ALLOWED_TOP_KEYS, "screening configuration")

    version = int(payload.get("screening_schema_version", SCREENING_SCHEMA_VERSION))
    if version != SCREENING_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported screening_schema_version {version}; "
            f"this build reads {SCREENING_SCHEMA_VERSION}"
        )

    chain = _require_mapping(payload.get("chain", {}), "chain")
    _reject_unknown(chain, _ALLOWED_CHAIN_KEYS, "chain")
    for required in ("n_sites", "j_eff_mev", "d_z_mev", "jz_mev"):
        if required not in chain:
            raise ValueError(f"chain requires {required}")

    protocol_payload = _require_mapping(payload.get("protocol", {}), "protocol")
    _reject_unknown(protocol_payload, _ALLOWED_PROTOCOL_KEYS, "protocol")
    bias_range = tuple(protocol_payload.get("bias_range_mev", (0.0, 20.0)))
    if len(bias_range) != 2:
        raise ValueError("protocol.bias_range_mev must contain low and high values")
    weights = protocol_payload.get("observable_weights")
    protocol = SpectroscopyProtocol.uniform(
        bias_range,
        points=int(protocol_payload.get("bias_points", 81)),
        broadening_mev=float(protocol_payload.get("broadening_mev", 0.25)),
        observable=str(protocol_payload.get("observable", "total_spin")),
        observable_weights=tuple(weights) if weights is not None else None,
        output_quantity=str(protocol_payload.get("output_quantity", "didv")),
    )

    if "candidates" not in payload and "sweep" not in payload:
        raise ValueError("screening configuration requires candidates or sweep")

    base = dict(
        n_sites=int(chain["n_sites"]),
        j_eff_mev=float(chain["j_eff_mev"]),
        d_z_mev=float(chain["d_z_mev"]),
        jz_mev=float(chain["jz_mev"]),
        j2_mev=float(chain.get("j2_mev", 0.0)),
        j3_mev=float(chain.get("j3_mev", 0.0)),
    )

    designs: list[DmiDesign] = []

    listed = payload.get("candidates") or []
    if not isinstance(listed, (list, tuple)):
        raise ValueError("candidates must be a list")
    for index, entry in enumerate(listed):
        mapping = _require_mapping(entry, f"candidates[{index}]")
        _reject_unknown(
            mapping,
            {"label", "impurities", "transverse_field_mev"},
            f"candidates[{index}]",
        )
        impurities = tuple(
            _build_impurity(item, f"candidates[{index}].impurities[{position}]")
            for position, item in enumerate(mapping.get("impurities") or [])
        )
        designs.append(
            DmiDesign(
                **base,
                impurities=impurities,
                transverse_field_mev=float(mapping.get("transverse_field_mev", 0.0)),
                label=str(mapping.get("label", "")),
            )
        )

    sweep = payload.get("sweep")
    if sweep is not None:
        sweep = _require_mapping(sweep, "sweep")
        _reject_unknown(sweep, _ALLOWED_SWEEP_KEYS, "sweep")
        site_groups = sweep.get("sites")
        if not site_groups:
            raise ValueError("sweep requires sites, a list of site lists")
        transverse_values = sweep.get("transverse_mev", [2.0])
        if not isinstance(transverse_values, (list, tuple)):
            transverse_values = [transverse_values]
        field_values = sweep.get("transverse_field_mev", [0.0])
        if not isinstance(field_values, (list, tuple)):
            field_values = [field_values]
        spin = str(sweep.get("spin", "S=1"))
        axial = float(sweep.get("axial_mev", 0.0))
        angle = float(sweep.get("transverse_angle_rad", 0.0))
        for sites in site_groups:
            if not isinstance(sites, (list, tuple)):
                raise ValueError("sweep.sites must be a list of site lists")
            for transverse in transverse_values:
                for field in field_values:
                    impurities = tuple(
                        SiteImpurity(
                            int(site), spin,
                            axial_mev=axial,
                            transverse_mev=float(transverse),
                            transverse_angle_rad=angle,
                        )
                        for site in sites
                    )
                    label = (
                        f"{len(sites)} imp at {list(sites)}, E={float(transverse):g}"
                    )
                    if float(field):
                        label += f", B={float(field):g}"
                    designs.append(
                        DmiDesign(
                            **base,
                            impurities=impurities,
                            transverse_field_mev=float(field),
                            label=label,
                        )
                    )

    if not designs:
        raise ValueError("screening configuration produced no candidate designs")
    return tuple(designs), protocol
