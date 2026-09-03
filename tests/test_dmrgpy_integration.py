import numpy as np
import pytest

from hamlet.simulation import (
    DmrgpySimulator,
    SpectroscopyProtocol,
    dmrgpy_energy_to_mev,
    mev_to_dmrgpy_energy,
)
from hamlet.systems import (
    HomogeneousHeisenbergChain,
    HomogeneousXXZLongRangeChain,
    HomogeneousXXZLongRangeFamily,
    HomogeneousXXZDMILongRangeChain,
    HomogeneousXXZDMILongRangeFamily,
)


@pytest.mark.integration
def test_dmrgpy_generates_finite_j1_j2_didv_map(tmp_path, monkeypatch):
    pytest.importorskip("dmrgpy")
    monkeypatch.chdir(tmp_path)
    system = HomogeneousHeisenbergChain(3, [10.0, 2.0])
    protocol = SpectroscopyProtocol.uniform(
        (0.0, 20.0), points=8, broadening_mev=1.0, output_quantity="didv"
    )
    result = DmrgpySimulator(
        max_bond_dimension=10, kpm_max_bond_dimension=10
    ).simulate(system, protocol)
    assert result.spectral_map.shape == (3, 8)
    assert np.isfinite(result.spectral_map).all()
    np.testing.assert_allclose(result.spectral_map[:, 0], 0.0)


def test_dmrgpy_energy_conversion_round_trip():
    mev = np.array([0.0, 10.0, 30.0, 45.0])
    np.testing.assert_allclose(mev_to_dmrgpy_energy(mev), [0.0, 1.0, 3.0, 4.5])
    np.testing.assert_allclose(dmrgpy_energy_to_mev(mev_to_dmrgpy_energy(mev)), mev)


def test_total_spin_protocol_records_explicit_components_and_weights():
    protocol = SpectroscopyProtocol.uniform(
        (0.0, 20.0),
        points=8,
        observable="total_spin",
        observable_weights=(1.0, 0.5, 2.0),
    )
    assert protocol.resolved_observable_weights == (1.0, 0.5, 2.0)
    assert protocol.observable_contract() == {
        "kind": "total_spin",
        "components": ["Sxx", "Syy", "Szz"],
        "weights": [1.0, 0.5, 2.0],
    }
    with pytest.raises(ValueError, match="apply only"):
        SpectroscopyProtocol.uniform(
            (0.0, 20.0), points=8, observable="Sz", observable_weights=(1, 1, 1)
        )
    with pytest.raises(ValueError, match="Sz.*total_spin"):
        SpectroscopyProtocol.uniform((0.0, 20.0), points=8, observable="Sx")


@pytest.mark.integration
def test_dmrgpy_generates_total_spin_response(tmp_path, monkeypatch):
    pytest.importorskip("dmrgpy")
    monkeypatch.chdir(tmp_path)
    system = HomogeneousHeisenbergChain(2, [10.0])
    protocol = SpectroscopyProtocol.uniform(
        (0.0, 20.0),
        points=6,
        broadening_mev=1.0,
        observable="total_spin",
        output_quantity="didv",
    )
    result = DmrgpySimulator(
        max_bond_dimension=10,
        kpm_max_bond_dimension=10,
        max_relative_imaginary_residue=1e-3,
    ).simulate(system, protocol)
    assert result.spectral_map.shape == (2, 6)
    assert np.isfinite(result.spectral_map).all()
    np.testing.assert_allclose(result.spectral_map[:, 0], 0.0)
    sz_result = DmrgpySimulator(
        max_bond_dimension=10,
        kpm_max_bond_dimension=10,
        max_relative_imaginary_residue=1e-3,
    ).simulate(
        system,
        SpectroscopyProtocol.uniform(
            (0.0, 20.0),
            points=6,
            broadening_mev=1.0,
            observable="Sz",
            output_quantity="didv",
        ),
    )
    np.testing.assert_allclose(
        result.spectral_map,
        3.0 * sz_result.spectral_map,
        rtol=2e-3,
        atol=2e-4,
    )


def test_xxz_long_range_parameter_contract():
    family = HomogeneousXXZLongRangeFamily(
        8, ((25.0, 45.0), (-8.0, 8.0), (-5.0, 5.0), (25.0, 45.0))
    )
    system = family.sample(np.random.default_rng(4))
    assert system.n_sites == 8
    assert system.parameter_names == ("J1_xy", "J2", "J3", "Jz")
    assert system.as_array().shape == (4,)
    with pytest.raises(ValueError, match="J3 interactions"):
        HomogeneousXXZLongRangeChain(3, [35.0, 2.0, 1.0, 35.0])


def test_xxz_dmi_parameter_contract():
    family = HomogeneousXXZDMILongRangeFamily(
        8, ((25.0, 45.0), (-8.0, 8.0), (-5.0, 5.0), (25.0, 45.0), (0.0, 6.0))
    )
    system = family.sample(np.random.default_rng(7))
    assert system.parameter_names == (
        "J1_xy", "J2", "J3", "Jz", "D_z_magnitude"
    )
    assert system.as_array()[4] >= 0.0
    with pytest.raises(ValueError, match="non-negative"):
        HomogeneousXXZDMILongRangeChain(8, [35.0, 2.0, 1.0, 35.0, -1.0])


@pytest.mark.integration
def test_dmrgpy_generates_finite_xxz_dmi_total_spin_map(tmp_path, monkeypatch):
    pytest.importorskip("dmrgpy")
    monkeypatch.chdir(tmp_path)
    system = HomogeneousXXZDMILongRangeChain(4, [35.0, 2.0, 1.0, 34.0, 3.0])
    protocol = SpectroscopyProtocol.uniform(
        (0.0, 30.0),
        points=6,
        broadening_mev=1.0,
        observable="total_spin",
        output_quantity="didv",
    )
    result = DmrgpySimulator(
        max_bond_dimension=10,
        kpm_max_bond_dimension=10,
        max_relative_imaginary_residue=1e-3,
    ).simulate(system, protocol)
    assert result.spectral_map.shape == (4, 6)
    assert np.isfinite(result.spectral_map).all()
