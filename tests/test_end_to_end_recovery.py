"""Scientific plumbing gates using identifiable synthetic Hamiltonians.

These tests validate parameter/site alignment and the complete public workflow.
The surrogate spectra are intentionally invertible and are not a replacement
for physical DMRG-to-experiment validation.
"""

import json

import numpy as np
import pytest

from hamlet.data import SpectroscopyDataset
from hamlet.io import TextImportRecipe, import_text_measurement
from hamlet.measurements import Measurement
from hamlet.workflow import advise_experiment
from hamlet.training import (
    TrainingPreprocessingConfig,
    TrainingRun,
    prepare_training_dataset,
    train_supervised,
)


def _gaussian(bias, center, width):
    return np.exp(-0.5 * ((bias - center) / width) ** 2)


def _local_surrogate_spectra(couplings, bias):
    """Encode left/right bonds in distinct spectral components per site."""
    couplings = np.asarray(couplings)
    spectra = []
    for site in range(couplings.size + 1):
        values = 0.2 + 3.0 * _gaussian(bias, 46.0, 1.5)
        if site > 0:
            values = values + couplings[site - 1] / 40.0 * _gaussian(bias, 13.0, 2.0)
        if site < couplings.size:
            values = values + couplings[site] / 40.0 * _gaussian(bias, 29.0, 2.5)
        spectra.append(values)
    return np.asarray(spectra)


@pytest.fixture(scope="module")
def local_recovery_run(tmp_path_factory):
    rng = np.random.default_rng(20260819)
    bias = np.linspace(0.0, 50.0, 201)
    targets = rng.uniform(25.0, 42.0, size=(150, 5))
    dataset = SpectroscopyDataset(
        spectra=np.stack([_local_surrogate_spectra(values, bias) for values in targets]),
        targets_mev=targets,
        bias_mev=bias,
        target_names=tuple(f"J_bond_{index}" for index in range(5)),
        system_type="inhomogeneous_heisenberg",
        metadata={"generator": "identifiable-local-validation-surrogate", "seed": 20260819},
    )
    prepared = prepare_training_dataset(
        dataset,
        TrainingPreprocessingConfig(bias_cutoff_mev=50.0, output_points=101),
    )
    run = train_supervised(
        prepared,
        view="local_bonds",
        model="ridge",
        preset="quick",
        model_options={"alpha": 1e-8},
    )
    artifact = run.save(tmp_path_factory.mktemp("recovery-artifact") / "ridge")
    return TrainingRun.load(artifact), bias, artifact


@pytest.mark.parametrize(
    "couplings",
    [
        np.array([26.0, 29.0, 33.0]),
        np.array([26.0, 29.0, 33.0, 37.0, 40.0, 28.0, 35.0]),
        np.array([30.0, 30.0, 30.0, 30.0]),
    ],
)
def test_local_pipeline_recovers_known_chains_of_different_lengths(
    local_recovery_run, couplings
):
    run, bias, _ = local_recovery_run
    result = run.create_analyzer().analyze(_local_surrogate_spectra(couplings, bias), bias)
    assert result.n_sites == couplings.size + 1
    np.testing.assert_allclose(result.coupling_mean, couplings, atol=2e-4)
    assert result.diagnostics.mean_overlap_disagreement < 1e-3
    assert run.metrics["test"]["ensemble"]["mae"] < 1e-3


def test_raw_files_to_import_to_inference_recovers_known_chain(
    local_recovery_run, tmp_path
):
    run, bias, _ = local_recovery_run
    expected = np.array([27.0, 31.0, 36.0, 39.0, 29.0, 34.0])
    spectra = _local_surrogate_spectra(expected, bias)
    source = tmp_path / "raw spectra with spaces"
    source.mkdir()
    for site, values in enumerate(spectra, start=1):
        second_derivative = np.gradient(np.gradient(values, bias), bias)
        rows = ["bias_mV,didv_A,d2idv2_A"]
        rows.extend(
            f"{energy:.12g},{didv:.12g},{d2:.12g}"
            for energy, didv, d2 in zip(bias, values, second_derivative)
        )
        (source / f"site{site}.txt").write_text("\n".join(rows), encoding="utf-8")
    payload = {
        "input": {"directory": str(source), "pattern": "*.txt"},
        "format": {"delimiter": ","},
        "columns": {
            "energy": {"column": "bias_mV", "unit": "mV"},
            "primary": "didv",
            "signals": {
                "didv": {"column": "didv_A", "unit": "A"},
                "d2idv2": {"column": "d2idv2_A", "unit": "A"},
            },
        },
        "processing": {"energy_order": "ascending", "grid": "require_equal", "missing": "error"},
        "output": {"directory": str(tmp_path / "canonical")},
    }
    imported = import_text_measurement(TextImportRecipe.from_mapping(payload, base=tmp_path))
    result = run.create_analyzer().analyze_measurement(imported.measurement)
    np.testing.assert_allclose(result.coupling_mean, expected, atol=2e-4)
    assert imported.measurement.primary_channel == "didv"
    assert "d2idv2" in imported.measurement.channels


def test_negative_controls_are_detected_or_degrade_recovery(local_recovery_run):
    run, bias, _ = local_recovery_run
    expected = np.array([26.0, 29.0, 33.0, 37.0, 40.0, 28.0, 35.0])
    spectra = _local_surrogate_spectra(expected, bias)

    shuffled = run.create_analyzer().analyze(spectra[::-1], bias)
    shuffled_mae = float(np.mean(np.abs(shuffled.coupling_mean - expected)))
    assert shuffled_mae > 5.0
    assert any("training" in warning for warning in shuffled.diagnostics.warnings)

    outside = np.array([15.0, 50.0, 15.0, 50.0, 15.0])
    out_result = run.create_analyzer().analyze(_local_surrogate_spectra(outside, bias), bias)
    assert out_result.diagnostics.fraction_predictions_outside_training_range > 0.5


def test_artifact_manifest_records_recovery_contract(local_recovery_run):
    run, _, artifact = local_recovery_run
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["view"] == "local_bonds"
    assert manifest["preprocessing"]["bias_cutoff_mev"] == 50.0
    assert manifest["energy_convention"]["dmrgpy_energy_unit_mev"] == 10.0
    assert manifest["dataset_metadata"]["seed"] == 20260819
    assert run.system_type == "inhomogeneous_heisenberg"


def _global_surrogate_spectra(parameters, bias, n_sites=6):
    j1, j2 = parameters
    return np.stack(
        [
            0.15
            + 3.0 * _gaussian(bias, 46.0, 1.5)
            + (j1 / 40.0) * (1.0 + site / 20.0) * _gaussian(bias, 12.0, 2.0)
            + (j2 / 8.0) * (1.0 - site / 30.0) * _gaussian(bias, 28.0, 2.5)
            for site in range(n_sites)
        ]
    )


def test_global_j1_j2_pipeline_recovers_known_parameters(tmp_path):
    rng = np.random.default_rng(31415)
    bias = np.linspace(0.0, 50.0, 201)
    targets = np.column_stack(
        [rng.uniform(25.0, 42.0, 120), rng.uniform(0.5, 8.0, 120)]
    )
    dataset = SpectroscopyDataset(
        spectra=np.stack([_global_surrogate_spectra(values, bias) for values in targets]),
        targets_mev=targets,
        bias_mev=bias,
        target_names=("J1", "J2"),
        system_type="homogeneous_heisenberg",
        metadata={"generator": "identifiable-global-validation-surrogate"},
    )
    prepared = prepare_training_dataset(
        dataset, TrainingPreprocessingConfig(bias_cutoff_mev=50.0, output_points=101)
    )
    run = train_supervised(
        prepared,
        view="global",
        model="ridge",
        preset="quick",
        model_options={"alpha": 1e-8},
    )
    artifact = run.save(tmp_path / "homogeneous-j1-j2")
    loaded = TrainingRun.load(artifact)
    expected = np.array([34.0, 3.5])
    result = loaded.create_analyzer().analyze(
        _global_surrogate_spectra(expected, bias), bias
    )
    assert result.n_sites == 6
    assert result.parameter_names == ("J1", "J2")
    np.testing.assert_allclose(result.coupling_mean, expected, atol=2e-4)
    with pytest.raises(ValueError, match="exactly 6 sites"):
        loaded.create_analyzer().analyze(
            _global_surrogate_spectra(expected, bias, n_sites=5), bias
        )

    csv_path = tmp_path / "parameters.csv"
    report_path = tmp_path / "report.json"
    result.save_couplings_csv(csv_path)
    result.save_report_json(report_path)
    assert csv_path.read_text().startswith("parameter,coupling,uncertainty,unit")
    report = json.loads(report_path.read_text())
    assert report["view"] == "global"
    assert report["parameter_names"] == ["J1", "J2"]

    measurement = Measurement(
        axes={"site": np.arange(6), "bias": bias},
        channels={"didv": _global_surrogate_spectra(expected, bias)},
        axis_units={"site": "index", "bias": "meV"},
        channel_units={"didv": "a.u."},
        metadata={"system_type": "homogeneous_heisenberg", "view": "global"},
    )
    decision = advise_experiment(
        measurement,
        manual_cutoff_mev=50.0,
        artifact_roots=[artifact],
        allow_development_artifacts=True,
    )
    assert decision.action == "use_existing_model"
    assert decision.n_sites == 6
    assert decision.selected_artifact == artifact.resolve()
    assert run.metrics["test"]["ensemble"]["mae"] < 1e-3
