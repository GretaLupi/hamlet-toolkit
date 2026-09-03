import numpy as np
import pytest

from hamlet.data import (
    SpectroscopyDataset,
    as_supervised,
    generate_dataset,
    generate_dataset_checkpointed,
)
from hamlet.simulation import SpectroscopyProtocol, SpectroscopyResult
from hamlet.systems import (
    HomogeneousHeisenbergFamily,
    InhomogeneousHeisenbergFamily,
)


class DeterministicSimulator:
    def simulate(self, system, protocol):
        target_sum = system.as_array().sum()
        spectra = np.stack(
            [target_sum + site + np.asarray(protocol.bias_mev) for site in range(system.n_sites)]
        )
        return SpectroscopyResult(protocol.bias_mev, spectra)


class FailAfterTwoSimulator(DeterministicSimulator):
    def __init__(self):
        self.calls = 0

    def simulate(self, system, protocol):
        self.calls += 1
        if self.calls > 2:
            raise RuntimeError("intentional interruption")
        return super().simulate(system, protocol)


def test_inhomogeneous_generation_and_local_view_are_reproducible(tmp_path):
    family = InhomogeneousHeisenbergFamily(6, (3.0, 4.0))
    protocol = SpectroscopyProtocol.uniform((0, 2), points=5)
    first = generate_dataset(family, DeterministicSimulator(), protocol, 4, seed=9)
    second = generate_dataset(family, DeterministicSimulator(), protocol, 4, seed=9)
    np.testing.assert_allclose(first.targets_mev, second.targets_mev)
    np.testing.assert_allclose(first.spectra, second.spectra)
    different = generate_dataset(family, DeterministicSimulator(), protocol, 4, seed=10)
    assert not np.array_equal(first.targets_mev, different.targets_mev)
    assert np.all((first.targets_mev >= 3.0) & (first.targets_mev <= 4.0))
    assert first.spectra.shape == (4, 6, 5)

    supervised = as_supervised(first, "local_bonds")
    assert supervised.inputs.shape == (16, 15)
    assert supervised.targets.shape == (16, 2)
    np.testing.assert_array_equal(supervised.group_ids, np.repeat(np.arange(4), 4))

    path = tmp_path / "dataset.npz"
    first.save(path)
    restored = SpectroscopyDataset.load(path)
    np.testing.assert_allclose(restored.spectra, first.spectra)
    assert restored.target_names == first.target_names


def test_homogeneous_j1_j2_generation_and_global_view():
    family = HomogeneousHeisenbergFamily(8, ((3.0, 4.0), (0.0, 1.0)))
    protocol = SpectroscopyProtocol.uniform(points=7)
    dataset = generate_dataset(family, DeterministicSimulator(), protocol, 3, seed=2)
    supervised = as_supervised(dataset, "global")
    assert dataset.target_names == ("J1", "J2")
    assert supervised.inputs.shape == (3, 8 * 7)
    assert supervised.targets.shape == (3, 2)


def test_protocol_records_requested_output_quantity():
    protocol = SpectroscopyProtocol.uniform(output_quantity="didv")
    assert protocol.output_quantity == "didv"


def test_checkpointed_generation_resumes_caches_and_rejects_mismatch(tmp_path):
    family = InhomogeneousHeisenbergFamily(5, (30.0, 40.0))
    protocol = SpectroscopyProtocol.uniform((0, 20), points=8)
    output = tmp_path / "generated.npz"
    recipe = {"system": "inhomogeneous_heisenberg", "n_samples": 5, "seed": 7}

    with pytest.raises(RuntimeError, match="intentional interruption"):
        generate_dataset_checkpointed(
            family,
            FailAfterTwoSimulator(),
            protocol,
            n_samples=5,
            output_path=output,
            recipe=recipe,
            seed=7,
            checkpoint_every=2,
        )
    assert not output.exists()
    assert len(list((tmp_path / ".generated.checkpoints").glob("chunk-*.npz"))) == 1

    resumed = generate_dataset_checkpointed(
        family,
        DeterministicSimulator(),
        protocol,
        n_samples=5,
        output_path=output,
        recipe=recipe,
        seed=7,
        checkpoint_every=2,
    )
    assert resumed.resumed_chunks == 1
    assert resumed.generated_chunks == 2
    assert resumed.dataset.n_samples == 5
    assert resumed.dataset.system_type == "inhomogeneous_heisenberg"
    assert not (tmp_path / ".generated.checkpoints").exists()

    uninterrupted = generate_dataset_checkpointed(
        family,
        DeterministicSimulator(),
        protocol,
        n_samples=5,
        output_path=tmp_path / "uninterrupted.npz",
        recipe=recipe,
        seed=7,
        checkpoint_every=2,
    )
    np.testing.assert_array_equal(resumed.dataset.targets_mev, uninterrupted.dataset.targets_mev)
    np.testing.assert_array_equal(resumed.dataset.spectra, uninterrupted.dataset.spectra)

    cached = generate_dataset_checkpointed(
        family,
        FailAfterTwoSimulator(),
        protocol,
        n_samples=5,
        output_path=output,
        recipe=recipe,
        seed=7,
        checkpoint_every=2,
    )
    assert cached.cache_hit
    with pytest.raises(FileExistsError, match="does not match"):
        generate_dataset_checkpointed(
            family,
            DeterministicSimulator(),
            protocol,
            n_samples=6,
            output_path=output,
            recipe={**recipe, "n_samples": 6},
            seed=7,
            checkpoint_every=2,
        )
