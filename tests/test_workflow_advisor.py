import json
from pathlib import Path
import shutil

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


PUBLISHED_DMI_ARTIFACT = (
    "src/hamlet/resources/models/"
    "homogeneous_xxz_j1j2j3_dmi_impurity_l8_ridge_standard_v1"
)

# The conditions this published artifact was actually trained under.
DMI_TRAINING_CONDITIONS = {
    "impurities": [
        {"site": 1, "spin": "S=1", "transverse_mev": 2.0},
        {"site": 4, "spin": "S=1", "transverse_mev": 2.0},
        {"site": 6, "spin": "S=1", "transverse_mev": 2.0},
    ],
    "transverse_field_mev": 0.0,
}


@pytest.fixture
def dmi_experiment(tmp_path):
    """An L=8 experiment covering the published artifact's 0-20 meV window."""
    repo_root = Path(__file__).resolve().parents[1]
    artifact = repo_root / PUBLISHED_DMI_ARTIFACT
    if not (artifact / "manifest.json").exists():
        pytest.skip("published DMI artifact is not present")
    bias = np.linspace(0.0, 20.0, 81)
    spectra = np.cumsum(
        np.exp(-((bias[None, :] - np.linspace(4.0, 12.0, 8)[:, None]) ** 2) / 2.0),
        axis=1,
    )
    experiment = tmp_path / "dmi_experiment.csv"
    _write_experiment(experiment, bias, spectra)
    return artifact, experiment


def test_impurity_artifact_is_refused_when_conditions_are_not_declared(dmi_experiment):
    """The impurity configuration is a training condition, like the cutoff.

    ``system_type`` is identical for every impurity chain regardless of how
    many impurities there are, where they sit, or what their anisotropies are,
    so an undeclared configuration must block reuse rather than be assumed to
    match.
    """
    artifact, experiment = dmi_experiment
    decision = advise_experiment(
        experiment,
        manual_cutoff_mev=20.0,
        artifact_roots=(artifact,),
        system_type="homogeneous_xxz_j1j2j3_dmi_impurity",
        view="global",
    )
    assert not decision.can_use_existing_model
    assessment = next(item for item in decision.artifact_assessments if item.path == artifact)
    assert any("has not declared" in reason for reason in assessment.reasons)


def test_impurity_artifact_is_reusable_when_conditions_match(dmi_experiment):
    artifact, experiment = dmi_experiment
    decision = advise_experiment(
        experiment,
        manual_cutoff_mev=20.0,
        artifact_roots=(artifact,),
        system_type="homogeneous_xxz_j1j2j3_dmi_impurity",
        view="global",
        experiment_conditions=DMI_TRAINING_CONDITIONS,
    )
    assessment = next(item for item in decision.artifact_assessments if item.path == artifact)
    assert assessment.compatible, assessment.reasons
    assert decision.can_use_existing_model


@pytest.mark.parametrize(
    "changed, label",
    [
        (
            {
                "impurities": [
                    {"site": 1, "spin": "S=1", "transverse_mev": 2.0},
                    {"site": 4, "spin": "S=1", "transverse_mev": 2.0},
                ],
                "transverse_field_mev": 0.0,
            },
            "one fewer impurity",
        ),
        (
            {
                "impurities": [
                    {"site": 2, "spin": "S=1", "transverse_mev": 2.0},
                    {"site": 4, "spin": "S=1", "transverse_mev": 2.0},
                    {"site": 6, "spin": "S=1", "transverse_mev": 2.0},
                ],
                "transverse_field_mev": 0.0,
            },
            "one impurity moved",
        ),
        (
            {
                "impurities": [
                    {"site": 1, "spin": "S=1", "transverse_mev": 2.0},
                    {"site": 4, "spin": "S=1", "transverse_mev": 2.0},
                    {"site": 6, "spin": "S=3/2", "transverse_mev": 2.0},
                ],
                "transverse_field_mev": 0.0,
            },
            "different species",
        ),
        (
            {
                "impurities": [
                    {"site": 1, "spin": "S=1", "transverse_mev": 1.5},
                    {"site": 4, "spin": "S=1", "transverse_mev": 2.0},
                    {"site": 6, "spin": "S=1", "transverse_mev": 2.0},
                ],
                "transverse_field_mev": 0.0,
            },
            "different anisotropy",
        ),
        (
            {**DMI_TRAINING_CONDITIONS, "transverse_field_mev": 1.0},
            "a field the model never saw",
        ),
    ],
)
def test_impurity_artifact_is_refused_when_any_condition_differs(
    dmi_experiment, changed, label
):
    artifact, experiment = dmi_experiment
    decision = advise_experiment(
        experiment,
        manual_cutoff_mev=20.0,
        artifact_roots=(artifact,),
        system_type="homogeneous_xxz_j1j2j3_dmi_impurity",
        view="global",
        experiment_conditions=changed,
    )
    assessment = next(item for item in decision.artifact_assessments if item.path == artifact)
    assert not assessment.compatible, f"{label} should block reuse"
    assert any("fixed-condition mismatch" in reason for reason in assessment.reasons)


def test_impurity_order_does_not_affect_the_condition_contract():
    """Listing the same impurities in another order is the same physical chain."""
    from hamlet.workflow import _canonical_fixed_conditions

    reversed_conditions = {
        "impurities": list(reversed(DMI_TRAINING_CONDITIONS["impurities"])),
        "transverse_field_mev": 0.0,
    }
    assert _canonical_fixed_conditions(
        DMI_TRAINING_CONDITIONS
    ) == _canonical_fixed_conditions(reversed_conditions)


def test_systems_without_fixed_conditions_are_unaffected():
    """Systems whose physics system_type fully determines declare nothing."""
    from hamlet.workflow import _canonical_fixed_conditions

    assert _canonical_fixed_conditions({"system_type": "homogeneous_heisenberg"}) is None
    assert _canonical_fixed_conditions({"impurities": [], "transverse_field_mev": 0.0}) is None
    assert _canonical_fixed_conditions(None) is None


def test_an_undeclared_system_does_not_exclude_another_family(
    workflow_resources, tmp_path
):
    """A measurement does not say which family of Hamiltonian produced it.

    The advisor used to default to ``inhomogeneous_heisenberg`` and then reject
    every homogeneous model for a "system mismatch" the data had never
    asserted -- while inference itself ran those same models happily. Whether a
    chain is treated as homogeneous or bond-inhomogeneous is a modelling
    choice: homogeneous is the special case where every bond is equal, and both
    are legitimate for one set of spectra.
    """
    _, _, artifact, experiment = workflow_resources
    homogeneous = tmp_path / "homogeneous-global"
    shutil.copytree(artifact, homogeneous)
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest.update(
        {"system_type": "homogeneous_heisenberg", "view": "global", "n_sites": 5}
    )
    (homogeneous / "manifest.json").write_text(json.dumps(manifest))

    decision = advise_experiment(
        experiment, manual_cutoff_mev=50.0, artifact_roots=[homogeneous]
    )
    assert decision.system_type is None, "a measurement does not fix the family"
    assert decision.view is None
    assert decision.artifact_assessments[0].compatible, (
        decision.artifact_assessments[0].reasons
    )
    assert decision.action == "use_existing_model"

    # The reports must survive an unconstrained family rather than crash on it.
    decision.save_json(tmp_path / "decision.json")
    decision.save_html(tmp_path / "decision.html")

    # Naming a family still narrows the search, for a caller that knows one.
    constrained = advise_experiment(
        experiment,
        manual_cutoff_mev=50.0,
        artifact_roots=[homogeneous],
        system_type="inhomogeneous_heisenberg",
    )
    assert not constrained.artifact_assessments[0].compatible
    assert any(
        "system mismatch" in reason
        for reason in constrained.artifact_assessments[0].reasons
    )


def test_a_model_that_fixes_conditions_is_still_refused_without_them(
    workflow_resources, tmp_path
):
    """Dropping the family assumption must not drop the physical requirement.

    An impurity model asserts impurities at named sites. That is a fact about
    the sample, not a modelling choice, so it stays refused until the sample is
    declared to have them -- which is what separates it from the homogeneous
    model that is now offered.
    """
    _, _, artifact, experiment = workflow_resources
    impurity = tmp_path / "impurity-global"
    shutil.copytree(artifact, impurity)
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest.update(
        {
            "system_type": "homogeneous_xxz_j1j2j3_dmi_impurity",
            "view": "global",
            "n_sites": 5,
        }
    )
    manifest.setdefault("dataset_metadata", {})["generation_recipe"] = {
        "impurities": [{"site": 1, "site_spin": 1.0, "axial_d_mev": 0.0,
                        "transverse_e_mev": 2.0, "in_plane_angle_deg": 0.0}],
        "transverse_field_mev": 0.0,
    }
    (impurity / "manifest.json").write_text(json.dumps(manifest))

    decision = advise_experiment(
        experiment, manual_cutoff_mev=50.0, artifact_roots=[impurity]
    )
    assert not decision.artifact_assessments[0].compatible
    assert any(
        "fixes physical conditions" in reason
        for reason in decision.artifact_assessments[0].reasons
    )


def test_a_clean_chain_is_an_answer_not_a_silence(workflow_resources, tmp_path):
    """"Declared clean" and "nobody said" reach the same value, not the same meaning.

    Both canonicalise to no impurities and no field, so an impurity model is
    rightly refused either way. But telling someone who has just described
    their sample as clean that they "have not declared" reads as the interface
    ignoring them, and leaves them with no idea what to do next.
    """
    _, _, artifact, experiment = workflow_resources
    impurity = tmp_path / "impurity-model"
    shutil.copytree(artifact, impurity)
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest.setdefault("dataset_metadata", {})["generation_recipe"] = {
        "impurities": [{"site": 1, "spin": "S=1", "axial_mev": 0.0,
                        "transverse_mev": 2.0, "transverse_angle_rad": 0.0}],
        "transverse_field_mev": 0.0,
    }
    (impurity / "manifest.json").write_text(json.dumps(manifest))

    def reasons(conditions):
        decision = advise_experiment(
            experiment,
            manual_cutoff_mev=50.0,
            artifact_roots=[impurity],
            experiment_conditions=conditions,
        )
        assert not decision.artifact_assessments[0].compatible
        return " ".join(decision.artifact_assessments[0].reasons)

    silent = reasons(None)
    assert "has not declared" in silent

    clean = reasons({"impurities": [], "transverse_field_mev": 0.0})
    assert "has not declared" not in clean, "they did declare; they said none"
    assert "no impurities and no field" in clean
    assert "trained on a chain with" in clean, "say what the model needs"
