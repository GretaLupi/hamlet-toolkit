"""Chains of spin greater than one half, and what that means for impurities.

Two things have to hold. The spin must actually reach the simulator -- a
system whose `site_spins` is ignored would simulate the wrong Hilbert space
and report nothing wrong -- and it must be part of the recipe, because a
dataset generated at S=1 is not interchangeable with one generated at S=1/2.
"""

from __future__ import annotations


import numpy as np
import pytest

from hamlet.systems import (
    HomogeneousHeisenbergChain,
    HomogeneousHeisenbergFamily,
    HomogeneousXXZDMIImpurityChain,
    HomogeneousXXZDMIImpurityFamily,
    HomogeneousXXZLongRangeFamily,
    InhomogeneousHeisenbergFamily,
    SiteImpurity,
)
from hamlet.systems.heisenberg import validate_site_spin
from hamlet.simulation.dmrgpy import hilbert_dimension, recommended_dynamics_mode


ALL_SPINS = ["S=1/2", "S=1", "S=3/2", "S=2", "S=5/2"]


@pytest.mark.parametrize("spin", ALL_SPINS)
def test_every_family_carries_its_spin_into_the_sampled_chain(spin):
    """A family that dropped the spin would silently generate the wrong chain."""
    rng = np.random.default_rng(0)
    families = [
        InhomogeneousHeisenbergFamily(6, (30.0, 40.0), site_spin=spin),
        HomogeneousHeisenbergFamily(6, ((30.0, 40.0),), site_spin=spin),
        HomogeneousXXZLongRangeFamily(
            6, ((5, 15), (0, 1), (0, 1), (5, 15)), site_spin=spin
        ),
    ]
    for family in families:
        chain = family.sample(rng)
        assert chain.site_spins == (spin,) * 6, family.system_type


def test_the_spin_changes_the_hilbert_space_the_simulator_will_build():
    """This is the whole cost of the setting, and what picks ED over DMRG."""
    half = HomogeneousHeisenbergChain(8, [30.0])
    one = HomogeneousHeisenbergChain(8, [30.0], site_spin="S=1")

    assert hilbert_dimension(half) == 2**8
    assert hilbert_dimension(one) == 3**8
    # Eight spin-1 sites is 6561 states, past the ED limit, so the simulator
    # has to choose DMRG without being told.
    assert recommended_dynamics_mode(half) == "ED"
    assert recommended_dynamics_mode(one) == "DMRG"


@pytest.mark.parametrize("bad", ["S=0", "S=7", "1/2", "", "S=3"])
def test_an_unknown_spin_is_refused_with_the_list(bad):
    with pytest.raises(ValueError, match="site_spin must be one of"):
        validate_site_spin(bad)


# --- impurities have to differ from the chain -------------------------------

def test_an_impurity_may_not_carry_the_chains_own_spin():
    """A site matching the chain is a chain site, not a substitution."""
    with pytest.raises(ValueError, match="carry the chain's own spin"):
        HomogeneousXXZDMIImpurityChain(
            8, [5, 0, 0, 5, 1],
            impurities=(SiteImpurity(site=1, spin="S=1", transverse_mev=2.0),),
            site_spin="S=1",
        )


def test_the_same_impurity_is_fine_in_a_chain_it_differs_from():
    chain = HomogeneousXXZDMIImpurityChain(
        8, [5, 0, 0, 5, 1],
        impurities=(SiteImpurity(site=1, spin="S=1", transverse_mev=2.0),),
        site_spin="S=1/2",
    )
    assert chain.site_spins[1] == "S=1"
    assert chain.site_spins[0] == "S=1/2"


def test_a_higher_spin_chain_takes_a_higher_spin_impurity():
    """Raising the chain does not forbid impurities, it moves which ones work."""
    chain = HomogeneousXXZDMIImpurityChain(
        8, [5, 0, 0, 5, 1],
        impurities=(SiteImpurity(site=2, spin="S=3/2", transverse_mev=2.0),),
        site_spin="S=1",
    )
    assert chain.site_spins == ("S=1", "S=1", "S=3/2") + ("S=1",) * 5


def test_the_clash_is_caught_when_the_family_is_built_not_mid_run():
    """The family builds one chain up front; an hours-long run must not
    discover this after the first chunk."""
    with pytest.raises(ValueError, match="carry the chain's own spin"):
        HomogeneousXXZDMIImpurityFamily(
            8, ((5, 15), (0, 1), (0, 1), (5, 15), (0, 2)),
            impurities=(SiteImpurity(site=1, spin="S=1", transverse_mev=2.0),),
            site_spin="S=1",
        )


# --- the recipe -------------------------------------------------------------

def test_the_spin_is_part_of_the_recipe_fingerprint(tmp_path):
    """Unlike `workers`, this changes every number in the dataset.

    A cached S=1/2 dataset must not satisfy a request for S=1, which is
    exactly what leaving it out of the fingerprint would do.
    """
    from hamlet.project import DatasetGenerationConfig

    def make(spin):
        return DatasetGenerationConfig(
            system_type="homogeneous_heisenberg", output_path=tmp_path / "d.npz",
            n_sites=8, n_samples=10, coupling_ranges_mev=((5.0, 15.0),),
            site_spin=spin,
        )

    assert make("S=1").to_recipe()["site_spin"] == "S=1"
    assert make("S=1").to_recipe() != make("S=1/2").to_recipe()
    # And the execution detail still is not.
    assert "workers" not in make("S=1").to_recipe()


def test_a_bad_spin_is_refused_by_the_configuration(tmp_path):
    from hamlet.project import DatasetGenerationConfig

    with pytest.raises(ValueError, match="site_spin must be one of"):
        DatasetGenerationConfig(
            system_type="homogeneous_heisenberg", output_path=tmp_path / "d.npz",
            n_sites=8, n_samples=10, coupling_ranges_mev=((5.0, 15.0),),
            site_spin="S=9",
        )


def test_yaml_carries_the_spin_into_the_config(tmp_path):
    import yaml

    from hamlet.project import ProjectConfig

    config_path = tmp_path / "p.yaml"
    config_path.write_text(yaml.safe_dump({
        "config_schema_version": 1,
        "name": "spin one",
        "system_type": "homogeneous_heisenberg",
        "output_dir": str(tmp_path / "out"),
        "dataset": {"format": "generated", "generate": {
            "system": "homogeneous_heisenberg",
            "n_sites": 8, "n_samples": 4, "coupling_ranges_mev": [[5, 15]],
            "output_path": str(tmp_path / "d.npz"), "site_spin": "S=1",
        }},
        "training": {"cutoffs_mev": [40], "manual_cutoff_mev": 40,
                     "output_points": 20, "view": "global", "model": "ridge"},
    }), encoding="utf-8")
    config = ProjectConfig.from_file(config_path)
    assert config.generation.site_spin == "S=1"


# --- the interface ----------------------------------------------------------

def test_the_form_refuses_an_impurity_matching_the_chain(tmp_path):
    """Caught at the form, not after the generation stage has started."""
    from hamlet.gui import api

    form = {
        "name": "t", "system_type": "homogeneous_xxz_j1j2j3_dmi_impurity",
        "n_sites": 8, "n_samples": 4,
        "coupling_ranges_mev": [[2, 6], [-1, 1], [-1, 1], [2, 6], [0.3, 2.5]],
        "impurities": [{"site": 1, "spin": "S=1", "transverse_mev": 2.0}],
        "bias_range_mev": [0, 20], "bias_points": 21, "broadening_mev": 0.25,
        "observable": "total_spin", "observable_weights": [1, 1, 1],
        "cutoff_mev": 20.0, "output_points": 21, "model": "ridge",
        "site_spin": "S=1",
    }
    with pytest.raises(ValueError, match="same spin as the chain"):
        api.build_project_config(form, workspace=tmp_path)


def test_the_screening_page_offers_no_impurity_it_could_not_use():
    """It designs for a spin-1/2 chain and has no chain-spin control."""
    from hamlet.gui import api

    assert "S=1/2" not in api.describe_screening_options()["spins"]
    # The builder does have one, so there every spin is a possible impurity.
    builder = api.describe_builder_options()
    assert "S=1/2" in builder["spins"]
    assert builder["chain_spins"][0] == "S=1/2"


@pytest.mark.integration
def test_a_spin_one_chain_simulates_and_differs_from_spin_half():
    """The end-to-end check: the spin reaches DMRGPy and changes the spectrum.

    The total spectral weight of an on-site autocorrelator scales with
    S(S+1), so S=1 against S=1/2 should be near 2/0.75 = 2.67 times larger.
    That ratio is the evidence the Hilbert space really changed, rather than
    the label having been carried around and ignored.
    """
    pytest.importorskip("dmrgpy")
    from hamlet.simulation import DmrgpySimulator, SpectroscopyProtocol

    protocol = SpectroscopyProtocol.uniform(
        bias_range_mev=(0.0, 120.0), points=61, broadening_mev=2.0
    )
    weights = {}
    for spin in ("S=1/2", "S=1"):
        chain = HomogeneousHeisenbergChain(6, [30.0], site_spin=spin)
        simulator = DmrgpySimulator(dynamics_mode=recommended_dynamics_mode(chain))
        weights[spin] = simulator.simulate(chain, protocol).spectral_map.sum()

    ratio = weights["S=1"] / weights["S=1/2"]
    assert 2.0 < ratio < 3.3, f"spectral weight ratio {ratio:.2f} is not S(S+1)-like"
