"""Chains of spin greater than one half, and what that means for impurities.

Two things have to hold. The spin must actually reach the simulator -- a
system whose `site_spins` is ignored would simulate the wrong Hilbert space
and report nothing wrong -- and it must be part of the recipe, because a
dataset generated at S=1 is not interchangeable with one generated at S=1/2.
"""

from __future__ import annotations


from pathlib import Path

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
    """S=1/2 is never a usable impurity, whatever the host is.

    Not because it matches the chain -- the host is now choosable, so it may
    not -- but because single-ion anisotropy vanishes at S=1/2, and the
    transverse anisotropy is the whole mechanism that exposes D_z. An S=1/2
    impurity in a spin-1 chain is a legal substitution that cannot do the one
    thing this page is for.
    """
    from hamlet.gui import api

    screening = api.describe_screening_options()
    assert "S=1/2" not in screening["spins"]
    # The host, by contrast, is unrestricted.
    assert screening["chain_spins"][0] == "S=1/2"
    assert "S=5/2" in screening["chain_spins"]

    # The training builder offers S=1/2 impurities, where a spin-1 host makes
    # them a meaningful substitution rather than a symmetry-breaking one.
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


# --- DMI screening at higher spin -------------------------------------------
# Three things had to become flexible here: the host's spin, an impurity that
# differs from it, and impurities that differ from each other. The fourth
# thing, which none of those asked for, is that the answer stays correct once
# they do.

def _screening(tmp_path, chain_extra=None, **sweep):
    import yaml

    from hamlet.dmi_design import load_screening_config

    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump({
        "screening_schema_version": 1,
        "chain": {"n_sites": 8, "j_eff_mev": 5.0, "d_z_mev": 1.5, "jz_mev": 5.5,
                  **(chain_extra or {})},
        "protocol": {"bias_range_mev": [0, 20], "bias_points": 21,
                     "broadening_mev": 0.25},
        "sweep": {"sites": [[1, 6]], "transverse_mev": [2.0], **sweep},
    }), encoding="utf-8")
    return load_screening_config(path)


def test_a_screening_host_can_carry_more_than_spin_half(tmp_path):
    designs, _ = _screening(tmp_path, {"site_spin": "S=1"}, spin="S=3/2")
    design = designs[0]
    assert design.site_spin == "S=1"
    # And it has to reach the chains that are actually simulated, not stop at
    # the design: both members of the gauge pair, or the imprint is measuring
    # something other than the D_z split.
    for chain in design.gauge_pair():
        assert chain.site_spins == ("S=1", "S=3/2") + ("S=1",) * 4 + ("S=3/2", "S=1")


def test_one_arrangement_can_mix_impurity_species(tmp_path):
    designs, _ = _screening(tmp_path, {"site_spin": "S=1"}, spins=["S=3/2", "S=2"])
    spins = [impurity.spin for impurity in designs[0].impurities]
    assert spins == ["S=3/2", "S=2"]
    # Mixed species still break the symmetry: the gauge that hides D_z acts
    # with the same phase at every site whatever its spin, so the counting
    # rule is unchanged.
    assert designs[0].breaks_symmetry


def test_a_sweep_spin_list_must_match_the_arrangement(tmp_path):
    with pytest.raises(ValueError, match="one spin per site"):
        _screening(tmp_path, spins=["S=1", "S=3/2", "S=2"])


def test_a_sweep_cannot_give_both_spin_and_spins(tmp_path):
    with pytest.raises(ValueError, match="both spin and spins"):
        _screening(tmp_path, spin="S=1", spins=["S=1", "S=3/2"])


def test_an_old_screening_file_still_loads(tmp_path):
    """No site_spin, one scalar spin -- what every saved file and the shipped
    example contain."""
    designs, _ = _screening(tmp_path, spin="S=1")
    assert designs[0].site_spin == "S=1/2"
    assert [i.spin for i in designs[0].impurities] == ["S=1", "S=1"]


# --- the part that stops a wrong number being reported as a right one -------

def test_screening_does_not_truncate_a_design_it_can_resolve():
    """DMRG at the library default of 20 is not the same measurement.

    An eight-site spin-1 chain needs a bond dimension of 81 to be exact. At
    20 the imprint came out 41% low, which is enough to move a design across
    the thresholds in IMPRINT_CALIBRATION -- and those were measured on exact
    spectra, so the comparison is not merely noisy, it is against a different
    scale.
    """
    from hamlet.dmi_design import (
        MAX_SCREENING_BOND_DIMENSION,
        DmiDesign,
        exact_bond_dimension,
        screening_cost,
        simulator_for,
        transverse_impurities,
    )

    design = DmiDesign(
        8, 5.0, 1.5, 5.5,
        impurities=transverse_impurities([1, 6], 2.0, spin=["S=3/2", "S=2"]),
        site_spin="S=1",
    )
    chain = design.gauge_pair()[0]
    required = exact_bond_dimension(chain)
    assert required > 20, "this design would not have exposed the bug"
    assert required <= MAX_SCREENING_BOND_DIMENSION

    simulator = simulator_for(design)
    assert simulator.dynamics_mode == "DMRG"
    assert simulator.max_bond_dimension >= required
    assert simulator.kpm_max_bond_dimension >= required
    assert screening_cost(design)["exact"] is True


def test_a_design_too_large_to_resolve_says_so():
    """Truncation is allowed; reporting it as exact is not."""
    from hamlet.dmi_design import DmiDesign, screening_cost, transverse_impurities

    design = DmiDesign(
        8, 5.0, 1.5, 5.5,
        impurities=transverse_impurities([1, 6], 2.0, spin="S=5/2"),
        site_spin="S=2",
    )
    cost = screening_cost(design)
    assert cost["dynamics_mode"] == "DMRG"
    assert cost["exact"] is False
    assert cost["exact_bond_dimension"] > cost["bond_dimension"]


def test_a_cheap_design_is_still_solved_exactly():
    from hamlet.dmi_design import DmiDesign, screening_cost, transverse_impurities

    design = DmiDesign(
        8, 5.0, 1.5, 5.5, impurities=transverse_impurities([1, 6], 2.0, spin="S=1")
    )
    cost = screening_cost(design)
    assert cost["dynamics_mode"] == "ED"
    assert cost["exact"] is True


def test_the_table_marks_a_truncated_row():
    """An exact 'promising' and a truncated 'promising' must not read alike."""
    from hamlet.dmi_design import (
        DmiDesign, DmiImprint, format_screening_table, transverse_impurities,
    )

    def row(exact):
        return DmiImprint(
            design=DmiDesign(
                8, 5.0, 1.5, 5.5,
                impurities=transverse_impurities([1, 6], 2.0, spin="S=1"),
            ),
            imprint=0.15, verdict="promising", predicted_to_break_symmetry=True,
            detail={"exact": exact},
        )

    assert "*" not in format_screening_table([row(True)]).split("calibration")[0]
    marked = format_screening_table([row(False)])
    assert "truncated" in marked and "lower bound" in marked


def test_the_screening_form_offers_no_impurity_the_host_forbids(tmp_path):
    """Raising the host must not invalidate the page's own defaults."""
    from hamlet.gui import api

    defaults = api.describe_screening_options()["defaults"]
    form = {
        "name": "x",
        "chain": {**defaults["chain"], "site_spin": "S=1"},
        "protocol": defaults["protocol"],
        "candidates": defaults["candidates"],
    }
    # The server names which arrangement broke the rule, rather than letting a
    # bare message escape from the chain class.
    with pytest.raises(ValueError, match="arrangement 1"):
        api.build_screening_config(form, workspace=tmp_path)

    # And the page filters the menu so that state is not reachable by hand.
    script = (
        Path(__file__).resolve().parents[1]
        / "src" / "hamlet" / "gui" / "static" / "app.js"
    ).read_text(encoding="utf-8")
    assert "function impuritySpins" in script
    assert 'screening.spins.filter((s) => s !== host)' in script


def test_the_form_sends_one_species_per_site(tmp_path):
    from hamlet.gui import api

    built = api.build_screening_config({
        "name": "mixed",
        "chain": {"n_sites": 8, "j_eff_mev": 5.0, "d_z_mev": 1.5, "site_spin": "S=1"},
        "protocol": {"bias_range_mev": [0, 20], "bias_points": 21,
                     "broadening_mev": 0.25},
        "candidates": [{"label": "mixed", "sites": [1, 6],
                        "spins": ["S=3/2", "S=2"], "transverse_mev": 2.0}],
    }, workspace=tmp_path)
    impurities = built["candidates"][0]["impurities"]
    assert [(i["site"], i["spin"]) for i in impurities] == [(1, "S=3/2"), (6, "S=2")]

    # And the host reaches the file, rather than being validated and dropped.
    import yaml

    saved = yaml.safe_load(Path(built["config_path"]).read_text(encoding="utf-8"))
    assert saved["chain"]["site_spin"] == "S=1"
