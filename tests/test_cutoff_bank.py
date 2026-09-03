import json
from dataclasses import replace

import numpy as np

from hamlet.io import load_reference_heisenberg_datasets
from hamlet.training import (
    AugmentationConfig,
    recommend_artifact,
    train_cutoff_bank,
)

from test_guided_training import make_training_dataset


def test_reference_dataset_import_converts_dmrgpy_units(tmp_path):
    n_samples, n_sites, n_bias = 3, 4, 11
    bias_backend = np.linspace(0, 10, n_bias, dtype=np.float32)
    x = np.tile(bias_backend, (n_samples, n_sites, 1)).reshape(n_samples, -1)
    z = np.ones_like(x)
    couplings = np.full((n_samples, n_sites - 1), 3.5, dtype=np.float32)
    path = tmp_path / "reference.npz"
    np.savez(path, X=x, Z=z, J=couplings, n_sites=n_sites, n_bias=n_bias)

    dataset = load_reference_heisenberg_datasets([path])
    assert dataset.spectra.shape == (n_samples, n_sites, n_bias)
    assert dataset.bias_mev[-1] == 100.0
    np.testing.assert_allclose(dataset.targets_mev, 35.0)
    np.testing.assert_allclose(dataset.spectra[0, 0], dataset.bias_mev / 10.0)


def test_cutoff_bank_selects_winner_and_recommends_by_coverage(tmp_path):
    root = tmp_path / "bank"
    dataset = make_training_dataset()
    # Keep the deliberately small fixture normalizable even above its peaks;
    # an exactly flat scale band has no defined multiplicative normalization.
    dataset = replace(
        dataset,
        spectra=dataset.spectra
        + 1e-4 * dataset.bias_mev[np.newaxis, np.newaxis, :],
    )
    catalog = train_cutoff_bank(
        dataset,
        root,
        cutoffs_mev=(30.0, 50.0, 70.0),
        comparison_cutoff_mev=50.0,
        candidates=("ridge", "random_forest"),
        preset="quick",
        output_points=30,
        model_options={
            "ridge": {"alpha": 0.1},
            "random_forest": {"n_estimators": 5, "n_jobs": 1},
        },
        experimental_like=False,
    )
    assert catalog["selected_model"] in {"ridge", "random_forest"}
    assert set(catalog["cutoff_artifacts"]) == {"30.0", "50.0", "70.0"}
    assert (root / "catalog.json").exists()

    recommendation = recommend_artifact(
        root, np.linspace(0, 62, 100), strategy="largest_covered"
    )
    assert recommendation.cutoff_mev == 50.0
    manifest = json.loads((recommendation.artifact_path / "manifest.json").read_text())
    assert manifest["model_name"] == catalog["selected_model"]

    recommendation = recommend_artifact(
        root, np.linspace(0, 100, 100), strategy="largest_covered"
    )
    assert recommendation.cutoff_mev == 70.0

    expected_best = min(
        (
            json.loads((root / relative / "manifest.json").read_text())["metrics"]
            ["validation"]["ensemble"]["mae"],
            float(cutoff),
        )
        for cutoff, relative in catalog["cutoff_artifacts"].items()
    )[1]
    recommendation = recommend_artifact(root, np.linspace(0, 100, 100))
    assert recommendation.cutoff_mev == expected_best


def test_recommendation_rejects_insufficient_coverage(tmp_path):
    root = tmp_path / "artifacts"
    artifact = root / "model"
    artifact.mkdir(parents=True)
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "system_type": "inhomogeneous_heisenberg",
                "view": "local_bonds",
                "preprocessing": {"bias_min_mev": 0, "bias_cutoff_mev": 30},
            }
        )
    )
    try:
        recommend_artifact(root, np.linspace(0, 20, 20))
    except ValueError as exc:
        assert "smallest" in str(exc)
    else:
        raise AssertionError("insufficient coverage unexpectedly produced a recommendation")


def test_cutoff_bank_records_complete_augmentation_contract(tmp_path):
    config = AugmentationConfig(
        broadening_points=(1.0, 3.0),
        energy_shift_mev=(-1.0, 1.0),
        amplitude_range=(0.9, 1.1),
        noise=0.001,
        seed=8,
    )
    catalog = train_cutoff_bank(
        make_training_dataset(),
        tmp_path / "calibrated",
        cutoffs_mev=(50.0,),
        comparison_cutoff_mev=50.0,
        candidates=("ridge",),
        preset="quick",
        output_points=30,
        augmentation_config=config,
    )
    assert catalog["augmentation_config"]["energy_shift_mev"] == (-1.0, 1.0)
    saved = json.loads((tmp_path / "calibrated" / "catalog.json").read_text())
    assert saved["augmentation_config"]["seed"] == 8
    assert saved["augmentation_config"]["noise"] == 0.001
