"""The impurity route to measuring DMI, and the symmetry rule that governs it.

These tests pin down which impurity configurations can constrain ``D_z`` and
which cannot. The rule is that a DM vector along ``z`` is unidentifiable in any
Hamiltonian conserving total ``S^z``, *and* that a single U(1)-breaking site is
still not enough, because the gauge rotation turns one impurity's in-plane
anisotropy axis by an angle that a global rotation about ``z`` undoes.

The integration test at the bottom measures that directly against the
simulator, since the claim is the entire justification for the system existing.
"""

import numpy as np
import pytest

from hamlet.systems import (
    HomogeneousXXZDMIImpurityChain,
    HomogeneousXXZDMIImpurityFamily,
    SiteImpurity,
)

COUPLINGS_MEV = [5.0, 0.0, 0.0, 5.5, 1.5]
RANGES_MEV = ((2.0, 8.0), (-1.5, 1.5), (-1.0, 1.0), (2.0, 8.0), (0.0, 2.0))


def transverse(site, e_mev=1.0, angle=0.0):
    return SiteImpurity(site, "S=1", transverse_mev=e_mev, transverse_angle_rad=angle)


def test_impurities_are_conditions_not_targets():
    chain = HomogeneousXXZDMIImpurityChain(
        8, COUPLINGS_MEV, impurities=(transverse(2), transverse(5))
    )
    assert chain.parameter_names == ("J1_xy", "J2", "J3", "Jz", "D_z_magnitude")
    assert chain.as_array().shape == (5,)


def test_site_spins_substitute_only_the_impurity_sites():
    chain = HomogeneousXXZDMIImpurityChain(
        8, COUPLINGS_MEV, impurities=(transverse(2), SiteImpurity(5, "S=5/2"))
    )
    assert chain.site_spins == (
        "S=1/2", "S=1/2", "S=1", "S=1/2", "S=1/2", "S=5/2", "S=1/2", "S=1/2",
    )


def test_a_bare_chain_has_no_mechanism_to_expose_dmi():
    chain = HomogeneousXXZDMIImpurityChain(8, COUPLINGS_MEV)
    assert chain.site_spins == ("S=1/2",) * 8
    assert not chain.exposes_dmi


@pytest.mark.parametrize(
    "impurities",
    [
        # A different spin magnitude alone conserves S^z.
        (SiteImpurity(3, "S=1"),),
        # Axial anisotropy commutes with S^z.
        (SiteImpurity(3, "S=1", axial_mev=2.0),),
        # One transverse impurity is undone by a global rotation about z.
        (transverse(3),),
        # Two sites, but only one breaks U(1).
        (transverse(2), SiteImpurity(5, "S=1", axial_mev=2.0)),
    ],
)
def test_configurations_that_cannot_expose_dmi(impurities):
    chain = HomogeneousXXZDMIImpurityChain(8, COUPLINGS_MEV, impurities=impurities)
    assert not chain.exposes_dmi


def test_two_transverse_impurities_expose_dmi():
    chain = HomogeneousXXZDMIImpurityChain(
        8, COUPLINGS_MEV, impurities=(transverse(1), transverse(6))
    )
    assert chain.exposes_dmi


def test_gauge_invariant_combination_is_the_hypotenuse():
    chain = HomogeneousXXZDMIImpurityChain(8, COUPLINGS_MEV)
    assert chain.gauge_invariant_j1_mev == pytest.approx(np.hypot(5.0, 1.5))


def test_spin_half_impurity_rejects_single_ion_anisotropy():
    # Not a no-op but a physics error: it would silently fail to break the
    # symmetry the impurity exists to break.
    with pytest.raises(ValueError, match="spin-1/2"):
        SiteImpurity(3, "S=1/2", transverse_mev=1.0)


def test_impurity_sites_must_be_distinct_and_in_range():
    with pytest.raises(ValueError, match="distinct"):
        HomogeneousXXZDMIImpurityChain(
            8, COUPLINGS_MEV, impurities=(transverse(2), transverse(2))
        )
    with pytest.raises(ValueError, match="index a site"):
        HomogeneousXXZDMIImpurityChain(8, COUPLINGS_MEV, impurities=(transverse(9),))


def test_negative_dmi_magnitude_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        HomogeneousXXZDMIImpurityChain(8, [5.0, 0.0, 0.0, 5.5, -0.1])


def test_family_has_a_distinct_system_type_and_samples_in_range():
    assert (
        HomogeneousXXZDMIImpurityFamily.system_type
        == "homogeneous_xxz_j1j2j3_dmi_impurity"
    )
    family = HomogeneousXXZDMIImpurityFamily(
        8, RANGES_MEV, impurities=(transverse(1), transverse(6))
    )
    chain = family.sample(np.random.default_rng(0))
    assert chain.exposes_dmi
    for value, (low, high) in zip(chain.as_array(), RANGES_MEV):
        assert low <= value <= high


def test_family_validates_the_impurity_configuration_up_front():
    # Catching this at construction matters: the alternative is discovering it
    # partway through a multi-hour generation run.
    with pytest.raises(ValueError, match="spin-1/2"):
        HomogeneousXXZDMIImpurityFamily(
            8, RANGES_MEV, impurities=(SiteImpurity(1, "S=1/2", transverse_mev=1.0),)
        )


@pytest.mark.integration
def test_simulator_hides_dmi_until_two_impurities_break_the_symmetry():
    """The measurement behind the whole design.

    A gauge pair shares ``sqrt(J1_xy^2 + D_z^2)`` and differs only in how it
    splits between exchange and DMI. With J2 = J3 = 0 the impurities are the
    only symmetry-breaking mechanism, so the pair must be *exactly* degenerate
    until two transverse impurities are present.
    """
    from hamlet.simulation import DmrgpySimulator, SpectroscopyProtocol

    j_eff, d_z = 5.0, 1.5
    j1_alt = float(np.sqrt(j_eff**2 - d_z**2))
    protocol = SpectroscopyProtocol.uniform(
        (0.0, 20.0), points=41, broadening_mev=0.25,
        observable="total_spin", observable_weights=(1.0, 1.0, 1.0),
        output_quantity="didv",
    )
    simulator = DmrgpySimulator(dynamics_mode="ED")

    def imprint(impurities):
        spectra = []
        for j1, dz in ((j_eff, 0.0), (j1_alt, d_z)):
            chain = HomogeneousXXZDMIImpurityChain(
                8, [j1, 0.0, 0.0, 5.5, dz], impurities=impurities
            )
            spectra.append(
                np.asarray(simulator.simulate(chain, protocol).spectral_map, dtype=float)
            )
        a, b = spectra
        return float(np.abs(a - b).max() / np.abs(a).max())

    # One transverse impurity: a global rotation about z undoes the gauge
    # rotation of its single in-plane axis, so the pair stays degenerate.
    assert imprint((transverse(3),)) < 1e-8
    # Two of them at different sites: their relative in-plane orientation
    # changes and no single global rotation restores both.
    assert imprint((transverse(1), transverse(6))) > 1e-3
