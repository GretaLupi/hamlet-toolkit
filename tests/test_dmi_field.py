"""The transverse-field DMI system, and why it exists.

A uniform DM vector along z can be removed from a nearest-neighbour XXZ chain
by the site-dependent rotation ``U = prod_j exp(-i j alpha S^z_j)`` with
``tan(alpha) = D_z / J1_xy``, leaving a plain XXZ chain of
``J'_xy = sqrt(J1_xy^2 + D_z^2)``. Every on-site autocorrelator is invariant
under that rotation once U(1) magnetisation conservation removes the terms that
would not be, so zero-field ``total_spin`` spectra cannot see ``D_z`` at all --
measured directly, a gauge pair agrees to one part in 10^13, which is why
``D_z`` scored zero skill at 32, 500 and 3000 training chains alike.

A transverse field breaks that symmetry. The integration test at the bottom is
the claim that motivates the whole system, so it is written down rather than
left in a notebook.
"""

import numpy as np
import pytest

from hamlet.systems import (
    HomogeneousXXZDMIFieldChain,
    HomogeneousXXZDMIFieldFamily,
    HomogeneousXXZDMILongRangeFamily,
)

RANGES = ((25.0, 45.0), (-8.0, 8.0), (-5.0, 5.0), (25.0, 45.0), (0.0, 6.0))


def test_field_is_a_condition_not_a_target():
    """The applied field is measured in the laboratory, never inferred."""
    family = HomogeneousXXZDMIFieldFamily(8, RANGES, transverse_field_mev=10.0)

    assert family.parameter_names == ("J1_xy", "J2", "J3", "Jz", "D_z_magnitude")
    assert "B" not in " ".join(family.parameter_names)

    chain = family.sample(np.random.default_rng(0))
    assert chain.as_array().shape == (5,)
    assert chain.transverse_field_mev == 10.0


def test_field_reaches_every_sampled_chain():
    family = HomogeneousXXZDMIFieldFamily(8, RANGES, transverse_field_mev=3.5)
    rng = np.random.default_rng(1)

    fields = {family.sample(rng).transverse_field_mev for _ in range(5)}

    assert fields == {3.5}


def test_system_type_is_distinct_from_the_zero_field_dmi_system():
    """An artifact trained in a field must never be reused without one."""
    assert (
        HomogeneousXXZDMIFieldFamily.system_type
        != HomogeneousXXZDMILongRangeFamily.system_type
    )


def test_gauge_invariant_combination_is_reported():
    chain = HomogeneousXXZDMIFieldChain(
        8, [34.4819, 0.0, 0.0, 38.0, 6.0], transverse_field_mev=0.0
    )

    # sqrt(34.4819^2 + 6^2) == 35 to the precision of the stored constant.
    assert chain.gauge_invariant_j1_mev == pytest.approx(35.0, abs=1e-3)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"parameters_mev": [1.0, 2.0, 3.0, 4.0]}, "J1_xy, J2, J3, Jz, D_z"),
        ({"parameters_mev": [35.0, 0.0, 0.0, 38.0, -1.0]}, "non-negative"),
        (
            {"parameters_mev": [35.0, 0.0, 0.0, 38.0, 1.0], "transverse_field_mev": -1.0},
            "non-negative",
        ),
        (
            {"parameters_mev": [35.0, 0.0, 0.0, 38.0, 1.0], "transverse_field_mev": np.nan},
            "finite",
        ),
    ],
)
def test_invalid_configurations_are_refused(kwargs, message):
    payload = {"n_sites": 8, **kwargs}
    with pytest.raises(ValueError, match=message):
        HomogeneousXXZDMIFieldChain(**payload)


def test_family_refuses_a_negative_field():
    with pytest.raises(ValueError, match="non-negative"):
        HomogeneousXXZDMIFieldFamily(8, RANGES, transverse_field_mev=-2.0)


@pytest.mark.integration
def test_transverse_field_makes_a_gauge_pair_distinguishable():
    """Zero field cannot see D_z; a transverse field can.

    Both members of the pair share ``sqrt(J1_xy^2 + D_z^2)`` and differ only in
    how that is split between exchange and DMI. J2 and J3 are zero so the field
    is the only symmetry-breaking mechanism in play.
    """
    pytest.importorskip("dmrgpy")
    from hamlet.simulation import DmrgpySimulator, SpectroscopyProtocol

    protocol = SpectroscopyProtocol.uniform(
        (0.0, 60.0),
        points=41,
        broadening_mev=1.0,
        observable="total_spin",
        observable_weights=(1.0, 1.0, 1.0),
        output_quantity="didv",
    )
    simulator = DmrgpySimulator(dynamics_mode="ED")

    j_eff, d_z = 35.0, 6.0
    j1_alt = float(np.sqrt(j_eff**2 - d_z**2))

    def relative_difference(field_mev: float) -> float:
        maps = [
            np.asarray(
                simulator.simulate(
                    HomogeneousXXZDMIFieldChain(
                        6, params, transverse_field_mev=field_mev
                    ),
                    protocol,
                ).spectral_map,
                dtype=float,
            )
            for params in (
                [j_eff, 0.0, 0.0, 38.0, 0.0],
                [j1_alt, 0.0, 0.0, 38.0, d_z],
            )
        ]
        return float(np.abs(maps[0] - maps[1]).max() / np.abs(maps[0]).max())

    without_field = relative_difference(0.0)
    with_field = relative_difference(10.0)

    # Gauge equivalence at zero field is exact, not approximate.
    assert without_field < 1e-8, (
        f"expected an exact degeneracy at zero field, got {without_field:.3e}"
    )
    # The field must lift it by orders of magnitude, otherwise the whole
    # transverse-field approach does not deliver an identifiable D_z.
    assert with_field > 1e-3, (
        f"a transverse field did not expose D_z: {with_field:.3e}"
    )
