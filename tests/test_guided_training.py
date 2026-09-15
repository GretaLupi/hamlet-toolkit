import json

import numpy as np
import pytest

from hamlet.data import SpectroscopyDataset, as_supervised
from hamlet.training import (
    TrainingPreprocessingConfig,
    TrainingRun,
    prepare_training_dataset,
    train_supervised,
)


def make_training_dataset(n_samples=12, n_sites=5):
    bias = np.linspace(0.0, 80.0, 161)
    targets = np.stack(
        [np.linspace(30.0 + sample / 10, 38.0 + sample / 10, n_sites - 1)
         for sample in range(n_samples)]
    ).astype(np.float32)
    spectra = np.empty((n_samples, n_sites, bias.size), dtype=np.float32)
    for sample in range(n_samples):
        for site in range(n_sites):
            left = targets[sample, max(site - 1, 0)]
            right = targets[sample, min(site, n_sites - 2)]
            spectra[sample, site] = (
                0.2
                + 0.01 * site
                + np.exp(-0.5 * ((bias - left) / 3.0) ** 2)
                + 0.7 * np.exp(-0.5 * ((bias - right) / 4.0) ** 2)
            )
    return SpectroscopyDataset(
        spectra=spectra,
        targets_mev=targets,
        bias_mev=bias,
        target_names=tuple(f"J{i + 1}" for i in range(n_sites - 1)),
        system_type="inhomogeneous_heisenberg",
        metadata={"generator": "unit-test"},
    )


def test_guided_training_artifact_round_trip(tmp_path):
    prepared = prepare_training_dataset(
        make_training_dataset(),
        TrainingPreprocessingConfig(bias_cutoff_mev=50.0, output_points=40),
    )
    run = train_supervised(
        prepared,
        view="local_bonds",
        model="ridge",
        preset="quick",
        model_options={"alpha": 0.1},
    )
    assert run.metrics["test"]["unit"] == "meV"
    assert run.metrics["test"]["ensemble"]["mae"] >= 0
    assert 0 <= run.metrics["test"]["ensemble"]["correlation_fidelity"] <= 1
    assert [row["name"] for row in run.metrics["test"]["per_target"]] == [
        "J_left", "J_right"
    ]
    assert all("skill" in row for row in run.metrics["test"]["per_target"])

    artifact_path = run.save(tmp_path / "ridge-artifact")
    manifest = json.loads((artifact_path / "manifest.json").read_text())
    assert manifest["toolkit"]["name"] == "HamLeT"
    assert manifest["preprocessing"]["bias_cutoff_mev"] == 50.0
    assert manifest["energy_convention"]["dmrgpy_energy_unit_mev"] == 10.0
    assert manifest["target_scaler"]["minimum"]
    assert manifest["training_distribution"]["method"] == "diagonal_rms_z_score"
    assert manifest["ensemble_aggregation"]["method"] in {
        "mean", "median", "validation_weighted"
    }
    evaluation = json.loads((artifact_path / "held_out_evaluation.json").read_text())
    assert evaluation["split"]["test_groups"] > 0
    assert evaluation["test"]["ensemble"]["correlation_fidelity"] >= 0
    assert evaluation["test"]["per_target"][0]["name"] == "J_left"

    loaded = TrainingRun.load(artifact_path)
    assert loaded.distribution_profile is not None
    assert loaded.aggregation == run.aggregation
    inputs = as_supervised(prepared.dataset, "local_bonds").inputs[:3]
    np.testing.assert_allclose(loaded.predict(inputs), run.predict(inputs), atol=1e-6)

    analyzer = loaded.create_analyzer()
    result = analyzer.analyze(prepared.dataset.spectra[0], prepared.dataset.bias_mev)
    assert result.coupling_mean.shape == (prepared.dataset.n_sites - 1,)
    assert result.coupling_unit == "meV"
    with pytest.raises(FileExistsError, match="not empty"):
        run.save(artifact_path)


def test_global_run_creates_fixed_length_homogeneous_analyzer():
    prepared = prepare_training_dataset(
        make_training_dataset(), TrainingPreprocessingConfig(bias_cutoff_mev=50.0)
    )
    run = train_supervised(prepared, view="global", model="ridge", preset="quick")
    analyzer = run.create_analyzer()
    result = analyzer.analyze(prepared.dataset.spectra[0], prepared.dataset.bias_mev)
    assert result.n_sites == prepared.dataset.n_sites
    assert result.n_parameters == prepared.dataset.targets_mev.shape[1]
    assert result.parameter_names == prepared.dataset.target_names

    with pytest.raises(ValueError, match="requires exactly"):
        analyzer.analyze(
            prepared.dataset.spectra[0, :-1], prepared.dataset.bias_mev
        )


def test_a_newer_keras_artifact_is_diagnosed_rather_than_dumped():
    """Training on a cluster and applying the model on a laptop is ordinary.

    A .keras file is not portable backwards: a newer Keras writes layer
    configuration an older one rejects. What arrives is nested "could not be
    deserialized properly" blocks naming every layer and initializer, ending
    dozens of lines down in an unexpected keyword argument. Nothing in that
    text says "version", and the two versions are the entire story.
    """
    from hamlet.training.guided import _keras_load_error

    cause = TypeError("GlorotUniform.__init__() got an unexpected keyword 'input_axes'")
    error = _keras_load_error(
        {"file": "model_seed_42.keras", "keras_version": "3.15.1"}, "3.11.3", cause
    )
    message = str(error)
    assert "3.15.1" in message and "3.11.3" in message
    assert 'pip install "keras>=3.15.1"' in message
    assert "retrain the model here" in message
    # The original is kept: it is the evidence, just no longer the headline.
    assert "input_axes" in message


def test_the_direction_of_the_mismatch_changes_the_advice():
    """Upgrading cannot fix a file written by an older Keras."""
    from hamlet.training.guided import _keras_load_error

    older_file = _keras_load_error(
        {"file": "m.keras", "keras_version": "3.2.0"}, "3.11.3", ValueError("boom")
    )
    assert "pip install" not in str(older_file)
    assert "retrain the model in this environment" in str(older_file)


def test_the_writing_keras_version_is_read_from_the_file_when_absent(tmp_path):
    """Every artifact that already exists predates the manifest field."""
    import json
    import zipfile

    from hamlet.training.guided import _keras_load_error, _keras_version_of

    path = tmp_path / "model_seed_42.keras"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.json", json.dumps({"keras_version": "3.15.1"}))
    assert _keras_version_of(path) == "3.15.1"

    error = _keras_load_error({"file": path.name}, "3.11.3", ValueError("boom"), path)
    assert "3.15.1" in str(error), "the file itself carries the answer"

    # And a file that cannot be read at all still gets a usable sentence.
    unreadable = tmp_path / "broken.keras"
    unreadable.write_bytes(b"not a zip")
    assert _keras_version_of(unreadable) == ""
    fallback = _keras_load_error(
        {"file": "broken.keras"}, "3.11.3", ValueError("boom"), unreadable
    )
    assert "could not be read" in str(fallback)


def test_a_saved_keras_artifact_records_the_version_that_wrote_it(tmp_path):
    """So the next person does not have to open the zip to find out.

    Trained rather than mocked: what matters is that the real save path adds
    the field, and the smallest honest network is cheap.
    """
    keras = pytest.importorskip("keras")
    import json

    import numpy as np

    from hamlet.training.preprocessing import TrainingPreprocessingConfig

    rng = np.random.default_rng(0)
    bias = np.linspace(0.0, 20.0, 16)
    targets = rng.uniform(5.0, 15.0, size=(12, 2)).astype(np.float32)
    spectra = np.stack([
        np.stack([np.exp(-((bias - t) ** 2)) for t in row]) for row in targets
    ]).astype(np.float32)
    dataset = SpectroscopyDataset(
        spectra=spectra, targets_mev=targets, bias_mev=bias,
        target_names=("J1", "J2"), system_type="homogeneous_heisenberg",
        metadata={},
    )
    prepared = prepare_training_dataset(
        dataset, TrainingPreprocessingConfig(bias_cutoff_mev=20.0, output_points=16)
    )
    run = train_supervised(
        prepared, view="global", model="keras_mlp", preset="quick",
        model_options={"hidden_units": [4]}, verbose=0,
    )
    run.save(tmp_path / "artifact")
    manifest = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    keras_records = [r for r in manifest["models"] if r["format"] == "keras"]
    assert keras_records, "keras_mlp should store keras models"
    assert all(r.get("keras_version") == keras.__version__ for r in keras_records)


def _tiny_keras_artifact(tmp_path, model="keras_mlp", **options):
    """A trained, saved Keras artifact, small enough to be a test fixture."""
    import numpy as np

    from hamlet.training.preprocessing import TrainingPreprocessingConfig

    rng = np.random.default_rng(0)
    bias = np.linspace(0.0, 20.0, 12)
    targets = rng.uniform(5.0, 15.0, size=(12, 2)).astype(np.float32)
    spectra = np.stack([
        np.stack([np.exp(-((bias - t) ** 2)) for t in row]) for row in targets
    ]).astype(np.float32)
    dataset = SpectroscopyDataset(
        spectra=spectra, targets_mev=targets, bias_mev=bias,
        target_names=("J1", "J2"), system_type="homogeneous_heisenberg", metadata={},
    )
    prepared = prepare_training_dataset(
        dataset, TrainingPreprocessingConfig(bias_cutoff_mev=20.0, output_points=12)
    )
    run = train_supervised(
        prepared, view="global", model=model, preset="quick",
        model_options=options, verbose=0,
    )
    destination = tmp_path / "artifact"
    run.save(destination)
    return run, destination, prepared


@pytest.mark.parametrize(
    ("model", "options"),
    [("keras_mlp", {"hidden_units": [4]}), ("keras_cnn", {"filters": [4], "dense_units": [4]})],
)
def test_rebuilding_from_the_manifest_reproduces_the_saved_model_exactly(
    tmp_path, model, options
):
    """The rescue path has to give the same answers, or it is not a rescue.

    A model trained on a cluster with a newer Keras cannot be read by an older
    one, so HamLeT rebuilds the architecture from the manifest -- which it
    wrote in the first place -- and loads the weights, which are plain arrays.
    That is only legitimate if the reconstruction is the same network. Here
    both paths are available, so they can be compared directly.
    """
    pytest.importorskip("keras")
    import json

    import keras
    import numpy as np

    from hamlet.training.guided import _rebuild_keras_model

    _, destination, prepared = _tiny_keras_artifact(tmp_path, model=model, **options)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    record = next(r for r in manifest["models"] if r["format"] == "keras")
    stored = destination / record["file"]

    original = keras.saving.load_model(stored)
    rebuilt = _rebuild_keras_model(manifest, record, stored)

    rng = np.random.default_rng(1)
    probe = rng.normal(size=(5, original.input_shape[1])).astype("float32")
    np.testing.assert_array_equal(
        original.predict(probe, verbose=0), rebuilt.predict(probe, verbose=0)
    )


def test_a_model_written_by_another_keras_still_loads(tmp_path, monkeypatch):
    """End to end: the artifact loads even when Keras refuses to read it.

    Keras is made to fail the way a version mismatch makes it fail, which is
    the only part of the situation that cannot be reproduced on one machine.
    """
    pytest.importorskip("keras")
    import keras
    import numpy as np

    from hamlet.training.guided import TrainingRun

    run, destination, _ = _tiny_keras_artifact(tmp_path, hidden_units=[4])
    probe = np.random.default_rng(2).normal(
        size=(4, run.models[0].input_shape[1])
    ).astype("float32")
    expected = run.predict(probe)

    def refuse(*args, **kwargs):
        raise TypeError(
            "GlorotUniform.__init__() got an unexpected keyword argument 'input_axes'"
        )

    monkeypatch.setattr(keras.saving, "load_model", refuse)
    recovered = TrainingRun.load(destination)
    np.testing.assert_allclose(recovered.predict(probe), expected, rtol=0, atol=0)


def test_an_artifact_that_cannot_be_rebuilt_still_explains_the_versions(tmp_path):
    """When the rescue fails too, the message says both things that went wrong."""
    from hamlet.training.guided import _keras_load_error

    error = _keras_load_error(
        {"file": "m.keras", "keras_version": "3.15.1"},
        "3.11.3",
        TypeError("unexpected keyword argument 'input_axes'"),
        None,
        ValueError("no such option: filters"),
    )
    message = str(error)
    assert "3.15.1" in message and "3.11.3" in message
    assert "Rebuilding it from the manifest was tried first" in message
    assert "no such option: filters" in message
