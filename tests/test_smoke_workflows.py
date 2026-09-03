"""Fast release-gate tests for the currently supported HamLeT workflows.

These tests intentionally use a tiny analytic spectroscopy simulator. They
validate HamLeT's orchestration and data contracts without requiring DMRGPy or
turning every local test run into a physics calculation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hamlet.data import SpectroscopyDataset, as_supervised, generate_dataset
from hamlet.simulation import SpectroscopyProtocol, SpectroscopyResult
from hamlet.systems import HomogeneousHeisenbergFamily, InhomogeneousHeisenbergFamily
from hamlet.training import (
    TrainingPreprocessingConfig,
    TrainingRun,
    prepare_training_dataset,
    train_supervised,
)


pytestmark = pytest.mark.smoke


class TinyAnalyticSimulator:
    """Encode every exchange coupling as a smooth, site-dependent peak."""

    def simulate(self, system, protocol):
        bias = np.asarray(protocol.bias_mev, dtype=float)
        couplings = system.as_array()
        spectra = []
        for site in range(system.n_sites):
            signal = 0.15 + 0.002 * bias + 0.01 * site
            for index, coupling in enumerate(couplings):
                width = 1.8 + 0.2 * index
                amplitude = 0.5 + 0.08 * (site + 1) + 0.05 * index
                signal = signal + amplitude * np.exp(
                    -0.5 * ((bias - coupling) / width) ** 2
                )
            spectra.append(signal)
        return SpectroscopyResult(bias, np.stack(spectra))


def _case(variant: str, *, n_sites: int = 6, n_samples: int = 15):
    if variant == "inhomogeneous":
        family = InhomogeneousHeisenbergFamily(n_sites, (15.0, 28.0))
        view = "local_bonds"
    else:
        family = HomogeneousHeisenbergFamily(
            n_sites, ((15.0, 28.0), (5.0, 12.0))
        )
        view = "global"
    protocol = SpectroscopyProtocol.uniform(
        (0.0, 60.0), points=61, output_quantity="didv"
    )
    dataset = generate_dataset(
        family, TinyAnalyticSimulator(), protocol, n_samples=n_samples, seed=123
    )
    return dataset, view


@pytest.mark.parametrize(
    ("variant", "n_sites", "expected_targets"),
    [
        ("inhomogeneous", 3, 2),
        ("inhomogeneous", 5, 4),
        ("inhomogeneous", 8, 7),
        ("homogeneous", 3, 2),
        ("homogeneous", 6, 2),
        ("homogeneous", 9, 2),
    ],
)
def test_small_dataset_generation_across_chain_lengths(
    variant, n_sites, expected_targets
):
    dataset, view = _case(variant, n_sites=n_sites, n_samples=4)

    assert dataset.spectra.shape == (4, n_sites, 61)
    assert dataset.targets_mev.shape == (4, expected_targets)
    assert dataset.metadata["seed"] == 123
    assert dataset.metadata["protocol"]["output_quantity"] == "didv"
    assert np.isfinite(dataset.spectra).all()

    supervised = as_supervised(dataset, view)
    expected_rows = 4 * (n_sites - 2) if view == "local_bonds" else 4
    assert supervised.inputs.shape[0] == expected_rows
    assert np.unique(supervised.group_ids).size == 4


@pytest.mark.parametrize("variant", ["inhomogeneous", "homogeneous"])
def test_generated_dataset_is_reproducible_and_portable(tmp_path, variant):
    first, _ = _case(variant, n_samples=5)
    second, _ = _case(variant, n_samples=5)
    np.testing.assert_array_equal(first.targets_mev, second.targets_mev)
    np.testing.assert_array_equal(first.spectra, second.spectra)

    path = tmp_path / f"{variant}.npz"
    first.save(path)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"]))
    assert manifest["toolkit"]["name"] == "HamLeT"

    restored = SpectroscopyDataset.load(path)
    np.testing.assert_array_equal(restored.targets_mev, first.targets_mev)
    np.testing.assert_array_equal(restored.spectra, first.spectra)
    assert restored.system_type == first.system_type


@pytest.mark.parametrize("variant", ["inhomogeneous", "homogeneous"])
@pytest.mark.parametrize("cutoff_mev", [35.0, 50.0])
def test_manual_cutoffs_produce_consistent_training_shapes(variant, cutoff_mev):
    dataset, view = _case(variant, n_samples=6)
    config = TrainingPreprocessingConfig(
        bias_cutoff_mev=cutoff_mev,
        output_points=24,
        baseline_range_mev=(0.0, 3.0),
    )
    prepared = prepare_training_dataset(dataset, config)
    supervised = as_supervised(prepared.dataset, view)

    assert prepared.dataset.spectra.shape == (6, dataset.n_sites, 24)
    assert prepared.dataset.bias_mev[0] == pytest.approx(0.0)
    assert prepared.dataset.bias_mev[-1] == pytest.approx(cutoff_mev)
    assert prepared.dataset.metadata["training_preprocessing"][
        "bias_cutoff_mev"
    ] == cutoff_mev
    expected_width = (3 if view == "local_bonds" else dataset.n_sites) * 24
    assert supervised.inputs.shape[1] == expected_width
    assert np.isfinite(supervised.inputs).all()


@pytest.mark.parametrize("variant", ["inhomogeneous", "homogeneous"])
@pytest.mark.parametrize("model_name", ["ridge", "random_forest"])
def test_generation_to_training_to_artifact_round_trip(
    tmp_path, variant, model_name
):
    pytest.importorskip("sklearn")
    dataset, view = _case(variant)
    prepared = prepare_training_dataset(
        dataset,
        TrainingPreprocessingConfig(bias_cutoff_mev=50.0, output_points=24),
    )
    options = {"alpha": 0.1} if model_name == "ridge" else {"n_estimators": 8, "n_jobs": 1}
    run = train_supervised(
        prepared,
        view=view,
        model=model_name,
        preset="quick",
        model_options=options,
    )

    assert run.system_type == dataset.system_type
    assert run.view == view
    assert np.isfinite(run.metrics["validation"]["ensemble"]["mae"])
    assert np.isfinite(run.metrics["test"]["ensemble"]["mae"])

    features = as_supervised(prepared.dataset, view).inputs[:3]
    before = run.predict(features)
    artifact = run.save(tmp_path / f"{variant}-{model_name}")
    restored = TrainingRun.load(artifact)
    np.testing.assert_allclose(restored.predict(features), before, atol=1e-6)

    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["toolkit"]["name"] == "HamLeT"
    assert manifest["preprocessing"]["bias_cutoff_mev"] == 50.0
    assert manifest["view"] == view


def test_inhomogeneous_artifact_can_analyze_an_unseen_chain():
    pytest.importorskip("sklearn")
    dataset, view = _case("inhomogeneous", n_samples=15)
    prepared = prepare_training_dataset(
        dataset,
        TrainingPreprocessingConfig(bias_cutoff_mev=50.0, output_points=24),
    )
    run = train_supervised(
        prepared,
        view=view,
        model="ridge",
        preset="quick",
        model_options={"alpha": 0.1},
    )

    unseen, _ = _case("inhomogeneous", n_samples=1)
    result = run.create_analyzer().analyze(unseen.spectra[0], unseen.bias_mev)
    assert result.n_bonds == unseen.n_sites - 1
    assert result.coupling_mean.shape == (unseen.n_sites - 1,)
    assert np.isfinite(result.coupling_mean).all()
    assert result.coupling_unit == "meV"


def test_neural_models_accept_the_same_prepared_data_contract():
    keras = pytest.importorskip("keras")
    from hamlet.models import CNNConfig, MLPConfig, create_supervised_model

    dataset, view = _case("inhomogeneous", n_samples=4)
    prepared = prepare_training_dataset(
        dataset,
        TrainingPreprocessingConfig(bias_cutoff_mev=50.0, output_points=24),
    )
    supervised = as_supervised(prepared.dataset, view)
    sample = supervised.inputs[:2]

    mlp = create_supervised_model(
        "keras_mlp",
        sample.shape[1],
        2,
        config=MLPConfig(hidden_units=(8,), dropout=0.0, batch_normalization=False),
    )
    cnn = create_supervised_model(
        "keras_cnn",
        sample.shape[1],
        2,
        spectrum_shape=(3, 24),
        config=CNNConfig(
            filters=(4,), dense_units=(8,), kernel_size=3, dropout=0.0,
            batch_normalization=False,
        ),
    )
    assert np.asarray(mlp(sample, training=False)).shape == (2, 2)
    assert np.asarray(cnn(sample, training=False)).shape == (2, 2)
    keras.backend.clear_session()
