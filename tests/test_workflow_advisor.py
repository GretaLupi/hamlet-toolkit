import json

import numpy as np
import pandas as pd
import pytest

from hamlet.project_cli import main as project_cli_main
from hamlet.project import HamiltonianLearningProject, ProjectConfig
from hamlet.experimental.cli import main as analyze_cli_main
from hamlet.io import load_measurement_csv
from hamlet.measurements import Measurement
from hamlet.training import (
    TrainingPreprocessingConfig,
    TrainingPreset,
    prepare_training_dataset,
    recommend_artifact,
    train_supervised,
)
from hamlet.workflow import advise_experiment

from test_guided_training import make_training_dataset


def _write_experiment(path, bias, spectra):
    rows = []
    for site, values in enumerate(spectra, start=1):
        rows.extend(
            {"site": site, "bias_meV": energy, "didv_A": signal}
            for energy, signal in zip(bias, values)
        )
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture(scope="module")
def workflow_resources(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("workflow-resources")
    dataset = make_training_dataset(n_samples=40)
    dataset_path = tmp_path / "training_dataset.npz"
    dataset.save(dataset_path)
    prepared = prepare_training_dataset(
        dataset, TrainingPreprocessingConfig(bias_cutoff_mev=50.0, output_points=40)
    )
    standard = TrainingPreset(
        name="standard", seeds=(42,), epochs=1, batch_size=32, patience=1
    )
    run = train_supervised(
        prepared,
        view="local_bonds",
        model="ridge",
        preset=standard,
        model_options={"alpha": 0.1},
    )
    artifact = run.save(tmp_path / "artifacts" / "ridge-cut50")
    experiment = tmp_path / "experiment.csv"
    _write_experiment(experiment, dataset.bias_mev, dataset.spectra[20])
    return dataset, dataset_path, artifact, experiment


def test_advisor_selects_exact_standard_artifact_when_contract_matches(
    workflow_resources, tmp_path
):
    _, dataset_path, artifact, experiment = workflow_resources
    decision = advise_experiment(
        experiment,
        manual_cutoff_mev=50.0,
        artifact_roots=[artifact.parent],
        dataset_paths=[dataset_path],
    )
    assert decision.action == "use_existing_model"
    assert decision.can_use_existing_model
    assert decision.to_dict()["workflow_decision_schema_version"] == 1
    assert decision.to_dict()["toolkit"]["full_name"] == "Hamiltonian Learning Toolkit"
    assert decision.selected_artifact == artifact
    assessment = decision.artifact_assessments[0]
    assert assessment.compatible
    decision.save_json(tmp_path / "decision.json")
    decision.save_html(tmp_path / "decision.html")
    assert json.loads((tmp_path / "decision.json").read_text())["can_use_existing_model"]
    assert "use_existing_model" in (tmp_path / "decision.html").read_text()


def test_different_manual_cutoff_never_substitutes_existing_weights(workflow_resources):
    _, dataset_path, artifact, experiment = workflow_resources
    decision = advise_experiment(
        experiment,
        manual_cutoff_mev=40.0,
        artifact_roots=[artifact],
        dataset_paths=[dataset_path],
    )
    assert decision.action == "retrain_with_existing_dataset"
    assert decision.selected_dataset == dataset_path
    assert not decision.artifact_assessments[0].compatible
    assert "cutoff mismatch" in decision.artifact_assessments[0].reasons[0]


def test_advisor_requests_generation_when_no_compatible_resources(workflow_resources):
    _, _, _, experiment = workflow_resources
    decision = advise_experiment(experiment, manual_cutoff_mev=50.0)
    assert decision.action == "generate_dataset_and_retrain"
    assert decision.selected_artifact is None
    assert decision.selected_dataset is None


def test_missing_resource_paths_are_reported_not_silently_ignored(
    workflow_resources, tmp_path
):
    _, _, _, experiment = workflow_resources
    decision = advise_experiment(
        experiment,
        manual_cutoff_mev=50.0,
        artifact_roots=[tmp_path / "mistyped-model-bank"],
        dataset_paths=[tmp_path / "mistyped-dataset.npz"],
    )
    assert decision.action == "generate_dataset_and_retrain"
    assert "does not exist" in decision.artifact_assessments[0].reasons[0]
    assert "cannot load dataset" in decision.dataset_assessments[0].reasons[0]


def test_missing_experimental_coverage_cannot_be_fixed_by_retraining(
    workflow_resources, tmp_path
):
    dataset, dataset_path, artifact, _ = workflow_resources
    experiment = tmp_path / "short_coverage.csv"
    mask = dataset.bias_mev <= 30.0
    _write_experiment(experiment, dataset.bias_mev[mask], dataset.spectra[20, :, mask])
    decision = advise_experiment(
        experiment,
        manual_cutoff_mev=50.0,
        artifact_roots=[artifact],
        dataset_paths=[dataset_path],
    )
    assert decision.action == "fix_experiment_or_choose_lower_cutoff"
    assert any(item.startswith("FAIL") for item in decision.experiment_checks)


def test_spectral_shape_does_not_override_an_exact_model_contract(
    workflow_resources, tmp_path
):
    dataset, _, artifact, _ = workflow_resources
    experiment = tmp_path / "different-shape.csv"
    rng = np.random.default_rng(123)
    spectra = rng.normal(size=dataset.spectra[0].shape)
    _write_experiment(experiment, dataset.bias_mev, spectra)
    decision = advise_experiment(
        experiment,
        manual_cutoff_mev=50.0,
        artifact_roots=[artifact],
    )
    assert decision.action == "use_existing_model"
    assessment = decision.artifact_assessments[0]
    assert assessment.compatible
    assert not any("OOD" in item for item in (*assessment.reasons, *assessment.warnings))


def test_advisor_enforces_simulation_observable_declared_by_experiment_mode(
    workflow_resources,
):
    _, _, artifact, experiment = workflow_resources
    loaded = load_measurement_csv(experiment)
    measurement = Measurement(
        axes=loaded.axes,
        channels=loaded.channels,
        axis_units=loaded.axis_units,
        channel_units=loaded.channel_units,
        primary_channel=loaded.primary_channel,
        metadata={"simulation_observable": "total_spin"},
    )
    decision = advise_experiment(
        measurement,
        manual_cutoff_mev=50.0,
        artifact_roots=[artifact],
    )
    assert decision.action == "generate_dataset_and_retrain"
    assert any(
        "simulation-observable mismatch" in reason
        for reason in decision.artifact_assessments[0].reasons
    )


def test_global_artifact_reports_chain_length_mismatch(
    workflow_resources, tmp_path
):
    _, _, artifact, experiment = workflow_resources
    global_artifact = tmp_path / "global-artifact"
    global_artifact.mkdir()
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest.update(
        {
            "system_type": "homogeneous_heisenberg",
            "view": "global",
            "n_sites": 8,
        }
    )
    (global_artifact / "manifest.json").write_text(json.dumps(manifest))

    decision = advise_experiment(
        experiment,
        manual_cutoff_mev=50.0,
        artifact_roots=[global_artifact],
        system_type="homogeneous_heisenberg",
        view="global",
    )
    assert decision.action == "generate_dataset_and_retrain"
    reasons = decision.artifact_assessments[0].reasons
    assert any("chain-length mismatch" in item for item in reasons)
    assert any("artifact=8 sites, experiment=5 sites" in item for item in reasons)


def test_user_defined_metric_limit_can_force_retraining(workflow_resources):
    _, dataset_path, artifact, experiment = workflow_resources
    decision = advise_experiment(
        experiment,
        manual_cutoff_mev=50.0,
        artifact_roots=[artifact],
        dataset_paths=[dataset_path],
        max_test_mae_mev=1e-12,
    )
    assert decision.action == "retrain_with_existing_dataset"
    assert any(
        "exceeds the user limit" in reason
        for reason in decision.artifact_assessments[0].reasons
    )


def test_quick_artifact_is_development_only_unless_explicitly_allowed(
    workflow_resources, tmp_path
):
    dataset, dataset_path, _, experiment = workflow_resources
    prepared = prepare_training_dataset(
        dataset, TrainingPreprocessingConfig(bias_cutoff_mev=50.0, output_points=40)
    )
    quick = train_supervised(
        prepared,
        view="local_bonds",
        model="ridge",
        preset="quick",
        model_options={"alpha": 0.1},
    ).save(tmp_path / "quick")
    guarded = advise_experiment(
        experiment,
        manual_cutoff_mev=50.0,
        artifact_roots=[quick],
        dataset_paths=[dataset_path],
    )
    assert guarded.action == "retrain_with_existing_dataset"
    assert "development-only" in " ".join(guarded.artifact_assessments[0].reasons)
    allowed = advise_experiment(
        experiment,
        manual_cutoff_mev=50.0,
        artifact_roots=[quick],
        allow_development_artifacts=True,
    )
    assert allowed.action == "use_existing_model"


def test_advisor_cli_writes_human_and_machine_readable_decision(
    workflow_resources, tmp_path, capsys
):
    _, dataset_path, artifact, experiment = workflow_resources
    output = tmp_path / "workflow output"
    code = project_cli_main(
        [
            "advise",
            str(experiment),
            "--cutoff",
            "50",
            "--artifact-root",
            str(artifact.parent),
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    assert "use_existing_model" in capsys.readouterr().out
    assert (output / "workflow_decision.json").exists()
    assert (output / "workflow_decision.html").exists()
    with pytest.raises(FileExistsError, match="--overwrite"):
        project_cli_main(
            [
                "advise",
                str(experiment),
                "--cutoff",
                "50",
                "--output-dir",
                str(output),
            ]
        )


def test_legacy_bank_recommendation_can_require_exact_manual_cutoff(
    workflow_resources, tmp_path
):
    _, _, artifact, _ = workflow_resources
    # A simple catalog is not required: recursive manifest discovery is supported.
    recommendation = recommend_artifact(
        artifact.parent, np.linspace(0, 80, 100), required_cutoff_mev=50.0
    )
    assert recommendation.cutoff_mev == 50.0
    with pytest.raises(ValueError, match="no artifact was trained"):
        recommend_artifact(
            artifact.parent, np.linspace(0, 80, 100), required_cutoff_mev=40.0
        )


def test_project_inference_refuses_artifact_from_different_manual_cutoff(
    workflow_resources, tmp_path
):
    _, _, artifact, experiment = workflow_resources
    config = ProjectConfig(
        name="cutoff mismatch",
        experiment_csv=experiment,
        output_dir=tmp_path / "output",
        artifact_path=artifact,
        cutoffs_mev=(40.0,),
        manual_cutoff_mev=40.0,
        verbose=0,
    )
    with pytest.raises(ValueError, match="cutoff-specific weights"):
        HamiltonianLearningProject(config).infer()


def test_project_existing_artifact_runs_preflight_before_inference(
    workflow_resources, tmp_path
):
    _, _, artifact, experiment = workflow_resources
    config = ProjectConfig(
        name="preflight success",
        experiment_csv=experiment,
        output_dir=tmp_path / "output",
        artifact_path=artifact,
        cutoffs_mev=(50.0,),
        manual_cutoff_mev=50.0,
        output_points=40,
        verbose=0,
    )
    project = HamiltonianLearningProject(config)
    result = project.infer()
    assert result.n_bonds == 4
    assert project.workflow_decision.action == "use_existing_model"
    assert (config.output_dir / "preflight" / "workflow_decision.json").exists()
    with pytest.raises(FileExistsError, match="preserve the previous analysis"):
        HamiltonianLearningProject(config).infer()


def test_analysis_cli_runs_manual_cutoff_preflight(workflow_resources, tmp_path):
    _, _, artifact, experiment = workflow_resources
    measurement_path = tmp_path / "guided-measurement.npz"
    load_measurement_csv(experiment).save(measurement_path)
    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiment_project_schema_version": 1,
                "system_type": "inhomogeneous_heisenberg",
                "view": "local_bonds",
                "outputs": {"measurement": str(measurement_path)},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    with pytest.raises(SystemExit, match="explicit manual --cutoff"):
        analyze_cli_main(
            [
                str(manifest_path),
                "--artifact",
                str(artifact),
                "--output-dir",
                str(tmp_path / "unsafe-analysis"),
            ]
        )
    assert (
        analyze_cli_main(
            [
                str(manifest_path),
                "--artifact",
                str(artifact),
                "--cutoff",
                "50",
                "--output-dir",
                str(output),
                "--no-plot",
            ]
        )
        == 0
    )
    assert (output / "workflow_decision.json").exists()
    assert (output / "couplings.csv").exists()
    with pytest.raises(FileExistsError, match="--overwrite"):
        analyze_cli_main(
            [
                str(manifest_path),
                "--artifact",
                str(artifact),
                "--cutoff",
                "50",
                "--output-dir",
                str(output),
                "--no-plot",
            ]
        )
