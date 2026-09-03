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
