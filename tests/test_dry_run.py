"""Tests for the side-effect-free project plan behind ``--dry-run``."""

import json

import numpy as np
import pytest
import yaml

from hamlet import HamiltonianLearningProject, ProjectConfig
from hamlet.data import SpectroscopyDataset
from hamlet.project_cli import main as project_cli_main


def write_generating_config(tmp_path, *, observable="Sz", n_samples=40, output_dir="out"):
    payload = {
        "config_schema_version": 1,
        "name": "dry run subject",
        "system_type": "homogeneous_heisenberg",
        "output_dir": output_dir,
        "dataset": {
            "generate": {
                "system": "homogeneous_heisenberg",
                "output": "data/generated.npz",
                "n_sites": 8,
                "n_samples": n_samples,
                "coupling_ranges_mev": [[30, 45], [0, 10]],
                "bias_range_mev": [0, 60],
                "bias_points": 61,
                "observable": observable,
                **(
                    {"observable_weights": [1, 1, 1]}
                    if observable == "total_spin"
                    else {}
                ),
            }
        },
        "training": {"cutoffs_mev": [60], "view": "global", "model": "ridge"},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_plan_lists_stages_and_outputs_without_touching_the_disk(tmp_path):
    config_path = write_generating_config(tmp_path)
    project = HamiltonianLearningProject.from_config(config_path)

    plan = project.plan()

    assert "generate simulations" in plan.stages
    assert any("train ridge" in stage for stage in plan.stages)
    assert plan.dataset_source == "generate"
    assert plan.generation_chains == 40
    assert not plan.would_refuse

    # The whole point is that planning is free: nothing may be created.
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "data").exists()
    assert sorted(item.name for item in tmp_path.iterdir()) == ["project.yaml"]


def test_budget_scales_with_the_number_of_evaluated_correlators(tmp_path):
    sz = HamiltonianLearningProject.from_config(
        write_generating_config(tmp_path / "a", observable="Sz")
    ).plan()
    total = HamiltonianLearningProject.from_config(
        write_generating_config(tmp_path / "b", observable="total_spin")
    ).plan()

    # total_spin evaluates Sxx, Syy and Szz, so it must cost about three times
    # a single Sz correlator rather than the same.
    assert sz.dataset_detail["evaluated_correlators"] == 1
    assert total.dataset_detail["evaluated_correlators"] == 3
    assert total.estimated_generation_seconds == pytest.approx(
        3 * sz.estimated_generation_seconds
    )


def test_measured_rate_overrides_the_reference_anchor(tmp_path):
    project = HamiltonianLearningProject.from_config(
        write_generating_config(tmp_path, n_samples=100)
    )

    plan = project.plan(seconds_per_chain=2.0)

    assert plan.seconds_per_chain == 2.0
    assert plan.estimated_generation_seconds == pytest.approx(200.0)


def test_existing_analysis_outputs_are_reported_as_refusing(tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    SpectroscopyDataset(
        spectra=np.zeros((4, 8, 61), dtype=np.float32),
        targets_mev=np.zeros((4, 2), dtype=np.float32),
        bias_mev=np.linspace(0.0, 60.0, 61),
        target_names=("J1", "J2"),
        system_type="homogeneous_heisenberg",
    ).save(dataset_path)

    experiment = tmp_path / "experiment.csv"
    experiment.write_text("site,bias_meV,didv_A\n1,0,1\n", encoding="utf-8")

    output_dir = tmp_path / "out"
    analysis = output_dir / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "report.json").write_text("{}", encoding="utf-8")

    config = ProjectConfig(
        name="already analysed",
        experiment_csv=experiment,
        output_dir=output_dir,
        system_type="homogeneous_heisenberg",
        dataset_path=dataset_path,
        cutoffs_mev=(60.0,),
        manual_cutoff_mev=60.0,
        view="global",
        model="ridge",
    )

    plan = HamiltonianLearningProject(config).plan()

    refusing = [item for item in plan.outputs if item.blocks_run]
    assert [item.path.name for item in refusing] == ["report.json"]
    assert plan.would_refuse


def test_plan_flags_a_missing_dataset(tmp_path):
    config = ProjectConfig(
        name="absent dataset",
        experiment_csv=None,
        output_dir=tmp_path / "out",
        system_type="homogeneous_heisenberg",
        dataset_path=tmp_path / "nope.npz",
        cutoffs_mev=(60.0,),
        view="global",
        model="ridge",
    )

    plan = HamiltonianLearningProject(config).plan()

    assert any("does not exist" in issue for issue in plan.blocking_issues)
    assert plan.would_refuse


def test_cli_dry_run_reports_and_writes_plan_json(tmp_path, capsys):
    config_path = write_generating_config(tmp_path)
    plan_json = tmp_path / "nested" / "plan.json"

    exit_code = project_cli_main(
        ["run", str(config_path), "--dry-run", "--plan-json", str(plan_json)]
    )

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "DRY RUN" in printed
    assert "Compute budget" in printed
    assert "Ready to run." in printed

    payload = json.loads(plan_json.read_text(encoding="utf-8"))
    assert payload["plan_schema_version"] == 1
    assert payload["generation_chains"] == 40
    assert payload["would_refuse"] is False
    # Still no real work performed.
    assert not (tmp_path / "out").exists()


def test_cli_dry_run_exit_code_signals_a_run_that_would_be_refused(tmp_path):
    config_path = write_generating_config(tmp_path, output_dir="out")
    analysis = tmp_path / "out" / "analysis"
    analysis.mkdir(parents=True)

    # Without an experiment there is no analysis stage, so this config plans
    # cleanly; adding an experiment plus an existing report is what refuses.
    assert project_cli_main(["run", str(config_path), "--dry-run"]) == 0

    experiment = tmp_path / "experiment.csv"
    experiment.write_text("site,bias_meV,didv_A\n1,0,1\n", encoding="utf-8")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["experiment"] = {"csv": str(experiment)}
    payload["training"]["manual_cutoff_mev"] = 60
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    (analysis / "couplings.csv").write_text("", encoding="utf-8")

    assert project_cli_main(["run", str(config_path), "--dry-run"]) == 1
