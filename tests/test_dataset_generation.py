from pathlib import Path

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


class ScratchWitnessSimulator(DeterministicSimulator):
    """Records the working directory each simulation ran in.

    DMRGPy derives its scratch paths from the working directory, so workers
    sharing one would overwrite each other's wavefunctions and silently
    produce a wrong dataset. Recording the directory is how a test can tell
    that they did not.
    """

    def simulate(self, system, protocol):
        import os

        result = super().simulate(system, protocol)
        witness = Path(os.getcwd()) / "cwd-witness.txt"
        witness.write_text(os.getcwd(), encoding="utf-8")
        return result


# --- parallel generation ----------------------------------------------------
# Generation is the long stage and every chain is independent, so it is the
# one place parallelism pays. What has to be true is that it changes nothing
# else: same numbers, same order, same cache.

def test_parallel_generation_gives_a_bit_identical_dataset(tmp_path):
    """Workers must be an execution detail and nothing more.

    Chunk seeds come from the seed sequence by index, so a chunk does not
    depend on when or where it ran. If that ever stops holding, a dataset
    becomes a function of the machine that made it.
    """
    family = HomogeneousHeisenbergFamily(6, ((5.0, 15.0),))
    protocol = SpectroscopyProtocol.uniform((0, 4), points=7)
    recipe = {"demo": True}

    serial = generate_dataset_checkpointed(
        family, DeterministicSimulator(), protocol,
        n_samples=12, output_path=tmp_path / "serial.npz", recipe=recipe,
        seed=5, checkpoint_every=3, workers=1,
    )
    parallel = generate_dataset_checkpointed(
        family, DeterministicSimulator(), protocol,
        n_samples=12, output_path=tmp_path / "parallel.npz", recipe=recipe,
        seed=5, checkpoint_every=3, workers=4,
    )

    assert parallel.workers_used > 1, "the parallel path did not run"
    assert np.array_equal(serial.dataset.spectra, parallel.dataset.spectra)
    assert np.array_equal(serial.dataset.targets_mev, parallel.dataset.targets_mev)
    # Order too, not just contents: chunks finish out of order and the sample
    # order has to stay the recipe's rather than the schedule's.
    assert serial.dataset.targets_mev.tolist() == parallel.dataset.targets_mev.tolist()


def test_each_worker_simulates_in_its_own_directory(tmp_path):
    """Two workers sharing a working directory would corrupt each other."""
    family = HomogeneousHeisenbergFamily(6, ((5.0, 15.0),))
    protocol = SpectroscopyProtocol.uniform((0, 4), points=7)
    origin = Path.cwd()

    result = generate_dataset_checkpointed(
        family, ScratchWitnessSimulator(), protocol,
        n_samples=8, output_path=tmp_path / "isolated.npz", recipe={"a": 1},
        seed=3, checkpoint_every=2, workers=4,
    )

    assert result.workers_used > 1
    # The parent's own directory must be untouched: it is process-global state
    # and everything else in the run resolves relative paths against it.
    assert Path.cwd() == origin
    assert not (origin / "cwd-witness.txt").exists()


def test_the_worker_pool_does_not_force_a_start_method():
    """Spawn would break the notebooks this package ships.

    Both spawn and forkserver re-import `__main__` in each worker, so they
    fail outright wherever `__main__` is not importable -- a notebook, or
    `python -c`. Pinning either one trades a low-probability fork/thread
    deadlock for a certain breakage of a documented workflow, so the platform
    default is used and the reasoning is recorded rather than rediscovered.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "hamlet" / "data" / "checkpointed.py"
    ).read_text(encoding="utf-8")

    assert "ProcessPoolExecutor(" in source
    assert "mp_context" not in source, (
        "a pinned start method breaks generation in notebooks; see the comment "
        "above the pool"
    )


def test_workers_are_not_part_of_the_recipe_fingerprint(tmp_path):
    """Otherwise a dataset made on four cores looks incompatible on eight.

    `workers` cannot change a single number in the result, so including it in
    the fingerprint would reject a valid cache and force an hours-long
    regeneration for nothing.
    """
    from hamlet.project import DatasetGenerationConfig

    four = DatasetGenerationConfig(
        system_type="homogeneous_heisenberg", output_path=tmp_path / "d.npz",
        n_sites=8, n_samples=10, coupling_ranges_mev=((5.0, 15.0),), workers=4,
    )
    one = DatasetGenerationConfig(
        system_type="homogeneous_heisenberg", output_path=tmp_path / "d.npz",
        n_sites=8, n_samples=10, coupling_ranges_mev=((5.0, 15.0),), workers=1,
    )
    assert "workers" not in four.to_recipe()
    assert four.to_recipe() == one.to_recipe()


def test_a_cached_dataset_is_reused_whatever_the_worker_count(tmp_path):
    family = HomogeneousHeisenbergFamily(6, ((5.0, 15.0),))
    protocol = SpectroscopyProtocol.uniform((0, 4), points=7)
    target = tmp_path / "cached.npz"
    first = generate_dataset_checkpointed(
        family, DeterministicSimulator(), protocol, n_samples=6,
        output_path=target, recipe={"a": 1}, seed=1, checkpoint_every=2, workers=1,
    )
    again = generate_dataset_checkpointed(
        family, DeterministicSimulator(), protocol, n_samples=6,
        output_path=target, recipe={"a": 1}, seed=1, checkpoint_every=2, workers=8,
    )
    assert not first.cache_hit and again.cache_hit
    assert np.array_equal(first.dataset.spectra, again.dataset.spectra)


def test_a_parallel_run_resumes_the_chunks_a_serial_one_wrote(tmp_path):
    """Stopping on one core and resuming on four has to be allowed."""
    family = HomogeneousHeisenbergFamily(6, ((5.0, 15.0),))
    protocol = SpectroscopyProtocol.uniform((0, 4), points=7)
    target = tmp_path / "resumed.npz"

    with pytest.raises(RuntimeError):
        generate_dataset_checkpointed(
            family, FailAfterTwoSimulator(), protocol, n_samples=9,
            output_path=target, recipe={"a": 1}, seed=2, checkpoint_every=1, workers=1,
        )
    finished = generate_dataset_checkpointed(
        family, DeterministicSimulator(), protocol, n_samples=9,
        output_path=target, recipe={"a": 1}, seed=2, checkpoint_every=1, workers=4,
    )
    assert finished.resumed_chunks == 2
    assert finished.generated_chunks == 7


@pytest.mark.parametrize("workers", [-1, -5])
def test_a_negative_worker_count_is_refused(tmp_path, workers):
    family = HomogeneousHeisenbergFamily(6, ((5.0, 15.0),))
    protocol = SpectroscopyProtocol.uniform((0, 4), points=7)
    with pytest.raises(ValueError, match="workers"):
        generate_dataset_checkpointed(
            family, DeterministicSimulator(), protocol, n_samples=4,
            output_path=tmp_path / "x.npz", recipe={}, seed=1, workers=workers,
        )


def test_asking_for_more_workers_than_chunks_does_not_start_idle_ones(tmp_path):
    family = HomogeneousHeisenbergFamily(6, ((5.0, 15.0),))
    protocol = SpectroscopyProtocol.uniform((0, 4), points=7)
    result = generate_dataset_checkpointed(
        family, DeterministicSimulator(), protocol, n_samples=4,
        output_path=tmp_path / "few.npz", recipe={}, seed=1,
        checkpoint_every=4, workers=32,
    )
    # One chunk of work: a pool of 32 processes would cost more to start than
    # the work is worth.
    assert result.workers_used == 1


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
