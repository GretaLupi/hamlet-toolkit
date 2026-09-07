"""Screening sample designs for whether they can expose DMI.

The point of this module is to answer "can this chain measure D_z, and well
enough to bother?" for an arbitrary user design, before any dataset is
generated. These tests cover the gauge-pair construction, the free symmetry
pre-filter, and the calibrated verdicts.

Most tests use a stub simulator so the logic is exercised without DMRGPy; the
integration test at the end checks the whole thing against the real backend.
"""

import numpy as np
import pytest

from hamlet.dmi_design import (
    HIDDEN_IMPRINT,
    DmiDesign,
    classify_imprint,
    format_screening_table,
    measure_dmi_imprint,
    screen_dmi_designs,
    transverse_impurities,
)
from hamlet.simulation import SpectroscopyProtocol

PROTOCOL = SpectroscopyProtocol.uniform(
    (0.0, 20.0), points=21, broadening_mev=0.25,
    observable="total_spin", observable_weights=(1.0, 1.0, 1.0),
    output_quantity="didv",
)


class StubResult:
    def __init__(self, spectral_map):
        self.spectral_map = spectral_map


class StubSimulator:
    """Returns a map whose amplitude depends on D_z by a controllable amount."""

    def __init__(self, sensitivity=0.0):
        self.sensitivity = sensitivity
        self.calls = 0

    def simulate(self, system, protocol):
        self.calls += 1
        base = np.ones((system.n_sites, np.asarray(protocol.bias_mev).size))
        d_z = float(system.as_array()[4])
        return StubResult(base * (1.0 + self.sensitivity * d_z))


def design(**overrides):
    params = dict(n_sites=8, j_eff_mev=5.0, d_z_mev=1.5, jz_mev=5.5)
    params.update(overrides)
    return DmiDesign(**params)


def test_gauge_pair_shares_the_invariant_combination():
    """Both members must be indistinguishable to a DMI-blind measurement."""
    first, second = design().gauge_pair()
    assert first.gauge_invariant_j1_mev == pytest.approx(5.0)
    assert second.gauge_invariant_j1_mev == pytest.approx(5.0)
    # One puts everything in exchange, the other splits off D_z.
    assert first.as_array()[4] == 0.0
    assert second.as_array()[4] == pytest.approx(1.5)
    assert second.as_array()[0] == pytest.approx(np.sqrt(5.0**2 - 1.5**2))


def test_dmi_cannot_exceed_the_shared_hypotenuse():
    with pytest.raises(ValueError, match="no greater than j_eff_mev"):
        design(d_z_mev=6.0)
    with pytest.raises(ValueError, match="must be positive"):
        design(d_z_mev=0.0)


@pytest.mark.parametrize(
    "impurities, field, expected",
    [
        ((), 0.0, False),
        (transverse_impurities([3], 2.0), 0.0, False),
        (transverse_impurities([1, 6], 2.0), 0.0, True),
        (transverse_impurities([1, 4, 6], 2.0), 0.0, True),
        ((), 1.0, True),
        (transverse_impurities([3], 2.0), 1.0, True),
        # Axial anisotropy conserves S^z, so it cannot help however large.
        (transverse_impurities([1, 6], 0.0, axial_mev=5.0), 0.0, False),
    ],
)
def test_symmetry_prefilter_identifies_hopeless_designs(impurities, field, expected):
    assert design(impurities=impurities, transverse_field_mev=field).breaks_symmetry is expected


def test_hopeless_designs_are_not_simulated():
    """The pre-filter is free; screening a wide sweep must exploit that."""
    simulator = StubSimulator(sensitivity=1.0)
    designs = [
        design(label="bare"),
        design(label="one impurity", impurities=transverse_impurities([3], 2.0)),
        design(label="two impurities", impurities=transverse_impurities([1, 6], 2.0)),
    ]
    results = screen_dmi_designs(designs, simulator, PROTOCOL)
    # Only the viable design reaches the simulator, two calls for its pair.
    assert simulator.calls == 2
    by_name = {item.design.label: item for item in results}
    assert by_name["bare"].verdict == "hidden"
    assert by_name["one impurity"].verdict == "hidden"
    assert not by_name["bare"].can_expose_dmi
    assert by_name["two impurities"].can_expose_dmi


def test_prefilter_can_be_disabled_to_verify_the_prediction():
    simulator = StubSimulator(sensitivity=0.0)
    screen_dmi_designs(
        [design(label="bare")], simulator, PROTOCOL, skip_symmetric=False
    )
    assert simulator.calls == 2


def test_results_are_ranked_best_first():
    simulator = StubSimulator(sensitivity=0.1)
    designs = [
        design(label="weak", d_z_mev=0.5, impurities=transverse_impurities([1, 6], 2.0)),
        design(label="strong", d_z_mev=4.0, impurities=transverse_impurities([1, 6], 2.0)),
        design(label="middle", d_z_mev=2.0, impurities=transverse_impurities([1, 6], 2.0)),
    ]
    ranked = screen_dmi_designs(designs, simulator, PROTOCOL)
    assert [item.design.label for item in ranked] == ["strong", "middle", "weak"]
    assert ranked[0].imprint > ranked[-1].imprint


def test_an_insensitive_measurement_reports_hidden():
    """A simulator blind to D_z must not be reported as exposing it."""
    result = measure_dmi_imprint(
        design(impurities=transverse_impurities([1, 6], 2.0)),
        StubSimulator(sensitivity=0.0),
        PROTOCOL,
    )
    assert result.imprint < HIDDEN_IMPRINT
    assert result.verdict == "hidden"
    # The symmetry argument still predicts breaking; the measurement overrides.
    assert result.predicted_to_break_symmetry
    assert not result.can_expose_dmi


@pytest.mark.parametrize(
    "imprint, verdict",
    [
        (1e-14, "hidden"),
        (1e-2, "too weak"),
        (4.4e-2, "too weak"),
        (8.0e-2, "marginal"),
        (1.18e-1, "promising"),
        (2.29e-1, "strong"),
    ],
)
def test_verdicts_match_the_measured_calibration(imprint, verdict):
    """Thresholds are anchored to D_z skill actually achieved, not invented.

    4.4e-2 trained to ~0 skill so it must not read better than "too weak";
    1.18e-1 trained to 0.19 and plateaued; 2.29e-1 trained to 0.54.
    """
    assert classify_imprint(imprint) == verdict


def test_screening_table_reports_designs_and_calibration():
    simulator = StubSimulator(sensitivity=0.1)
    results = screen_dmi_designs(
        [design(label="candidate", impurities=transverse_impurities([1, 6], 2.0))],
        simulator,
        PROTOCOL,
    )
    table = format_screening_table(results)
    assert "candidate" in table
    assert "calibration" in table
    assert "0.54" in table


def test_transverse_impurity_builder_requires_a_capable_spin():
    with pytest.raises(ValueError, match="spin-1/2"):
        transverse_impurities([1, 6], 2.0, spin="S=1/2")


@pytest.mark.integration
def test_screening_reproduces_the_measured_ranking_on_the_real_backend():
    """The ranking this module exists to produce, against DMRGPy.

    Three transverse impurities must beat one, and a bare chain must come back
    exactly degenerate. These are the measurements the recommended sample
    design rests on.
    """
    from hamlet.simulation import DmrgpySimulator

    protocol = SpectroscopyProtocol.uniform(
        (0.0, 20.0), points=41, broadening_mev=0.25,
        observable="total_spin", observable_weights=(1.0, 1.0, 1.0),
        output_quantity="didv",
    )
    designs = [
        design(label="bare"),
        design(label="1 impurity", impurities=transverse_impurities([3], 2.0)),
        design(label="3 impurities", impurities=transverse_impurities([1, 4, 6], 2.0)),
    ]
    ranked = screen_dmi_designs(
        designs, DmrgpySimulator(dynamics_mode="ED"), protocol, skip_symmetric=False
    )
    by_name = {item.design.label: item for item in ranked}
    assert by_name["bare"].imprint < HIDDEN_IMPRINT
    assert by_name["1 impurity"].imprint < HIDDEN_IMPRINT
    assert by_name["3 impurities"].imprint > 1e-2
    assert ranked[0].design.label == "3 impurities"
