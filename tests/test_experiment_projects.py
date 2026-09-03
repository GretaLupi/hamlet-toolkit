import json

import numpy as np
import pytest
import yaml

from hamlet import (
    HamiltonianLearningProject,
    ProjectConfig,
    advise_experiment,
    available_experiment_modes,
    load_experiment_project,
    resolve_experiment_mode,
)
from hamlet.measurements import Measurement
from hamlet.project_cli import main as project_cli_main


def _write_raw_experiment(tmp_path, *, n_sites=4):
    source = tmp_path / "raw site files"
    source.mkdir()
    bias = np.arange(-10.0, 61.0, 10.0)
    for site in range(1, n_sites + 1):
        rows = ["Bias (mV),LI Demod 1 X (A),LI Demod 2 X (A)"]
        for energy in bias:
            didv = 2.0 + 0.1 * site + np.sin((energy + site) / 15.0)
            second = np.cos((energy + site) / 12.0)
            rows.append(f"{energy},{didv},{second}")
        (source / f"site{site}.txt").write_text("\n".join(rows), encoding="utf-8")
    recipe = {
        "input": {"directory": str(source), "pattern": "*.txt"},
        "format": {"delimiter": ","},
        "columns": {
            "energy": {"column": "Bias (mV)", "unit": "mV"},
            "primary": "didv",
            "signals": {
                "didv": {"column": "LI Demod 1 X (A)", "unit": "A"},
                "d2idv2": {"column": "LI Demod 2 X (A)", "unit": "A"},
            },
        },
        "site": {"mode": "filename_number"},
        "processing": {
            "energy_order": "ascending",
            "grid": "require_equal",
            "missing": "error",
        },
        "output": {"directory": str(tmp_path / "recipe output")},
    }
    recipe_path = tmp_path / "raw import.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    return recipe_path


def test_registered_heisenberg_modes_have_distinct_experiment_contracts():
    assert available_experiment_modes() == {
        "heisenberg": (
            "homogeneous", "inhomogeneous", "xxz_dmi", "xxz_long_range"
        )
    }
    local = resolve_experiment_mode("heisenberg", "inhomo")
    assert local.system_type == "inhomogeneous_heisenberg"
    assert local.view == "local_bonds"
    assert local.experimental_inference_supported
    global_profile = resolve_experiment_mode("homogeneous-heisenberg")
    assert global_profile.view == "global"
    assert global_profile.experimental_inference_supported
    xxz = resolve_experiment_mode("heisenberg", "xxz")
    assert xxz.system_type == "homogeneous_xxz_j1j2j3"
    assert xxz.view == "global"
    dmi = resolve_experiment_mode("heisenberg", "dmi")
    assert dmi.system_type == "homogeneous_xxz_j1j2j3_dmi"
    assert dmi.recommended_observable == "total_spin"
    with pytest.raises(ValueError, match="available"):
        resolve_experiment_mode("qpi", "diffusion")


def test_modes_command_exposes_support_boundary(capsys):
    assert project_cli_main(["modes"]) == 0
    output = capsys.readouterr().out
    assert "heisenberg/inhomogeneous" in output
    assert "experimental workflow available" in output
    assert "heisenberg/homogeneous" in output
    assert "heisenberg/xxz_long_range" in output
    assert "heisenberg/xxz_dmi" in output
    assert output.count("experimental workflow available") == 4


def test_guided_cli_imports_inspects_and_creates_advisor_ready_manifest(
    tmp_path, capsys
):
    recipe = _write_raw_experiment(tmp_path)
    output = tmp_path / "guided experiment"
    assert (
        project_cli_main(
            [
                "inspect-experiment",
                str(recipe),
                "--mode",
                "heisenberg",
                "--variant",
                "inhomogeneous",
                "--output-dir",
                str(output),
                "--candidate-cutoffs",
                "40",
                "50",
                "70",
            ]
        )
        == 0
    )
    console = capsys.readouterr().out
    assert "ready_for_cutoff_selection" in console
    assert "40 meV, 50 meV" in console

    manifest_path = output / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["experiment_project_schema_version"] == 1
    assert manifest["toolkit"]["name"] == "HamLeT"
    assert manifest["mode"] == "heisenberg"
    assert manifest["variant"] == "inhomogeneous"
    assert manifest["system_type"] == "inhomogeneous_heisenberg"
    assert manifest["view"] == "local_bonds"
    assert manifest["selected_cutoff_mev"] is None
    assert manifest["available_candidate_cutoffs_mev"] == [40.0, 50.0]
    assert manifest["measurement"]["channel_roles"] == {
        "didv": "inference",
        "d2idv2": "plot_qc_only",
    }
    assert "before_model_preprocessing" in manifest["inspection_stage"]
    inspection = (output / "experiment_inspection.html").read_text()
    assert "data:image/png;base64," in inspection
    assert "Coverage is not a recommendation" in inspection

    measurement, loaded_manifest = load_experiment_project(manifest_path)
    assert loaded_manifest["system_type"] == "inhomogeneous_heisenberg"
    assert measurement.metadata["experiment_variant"] == "inhomogeneous"
    assert measurement.shape == (4, 8)

    decision = advise_experiment(manifest_path, manual_cutoff_mev=50.0)
    assert decision.system_type == "inhomogeneous_heisenberg"
    assert decision.view == "local_bonds"
    assert decision.action == "generate_dataset_and_retrain"
    too_high = advise_experiment(manifest_path, manual_cutoff_mev=70.0)
    assert too_high.action == "fix_experiment_or_choose_lower_cutoff"
    with pytest.raises(ValueError, match="conflicts with experiment manifest"):
        advise_experiment(
            manifest_path,
            manual_cutoff_mev=50.0,
            system_type="homogeneous_heisenberg",
        )

    with pytest.raises(FileExistsError, match="--overwrite"):
        project_cli_main(
            [
                "inspect-experiment",
                str(recipe),
                "--mode",
                "heisenberg",
                "--variant",
                "inhomogeneous",
                "--output-dir",
                str(output),
            ]
        )


def test_guided_output_override_preserves_explicit_input_files(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    files = []
    for site in (1, 2, 3):
        path = source / f"site{site}.dat"
        path.write_text(
            "Bias (mV),LI Demod 1 X (A)\n0,1\n10,2\n20,3\n",
            encoding="utf-8",
        )
        files.append(str(path))
    payload = {
        "input": {"files": files},
        "format": {"delimiter": ","},
        "columns": {
            "energy": {"column": "Bias (mV)", "unit": "mV"},
            "signals": {"didv": {"column": "LI Demod 1 X (A)", "unit": "A"}},
        },
        "site": {"mode": "sequential", "start": 1},
        "output": {"directory": str(tmp_path / "unused")},
    }
    recipe = tmp_path / "explicit-files.yaml"
    recipe.write_text(yaml.safe_dump(payload), encoding="utf-8")

    output = tmp_path / "redirected"
    assert (
        project_cli_main(
            [
                "inspect-experiment",
                str(recipe),
                "--mode",
                "heisenberg",
                "--variant",
                "inhomogeneous",
                "--output-dir",
                str(output),
                "--candidate-cutoffs",
                "10",
            ]
        )
        == 0
    )
    measurement, _ = load_experiment_project(output / "experiment_manifest.json")
    np.testing.assert_array_equal(measurement.axes["site"], [1, 2, 3])


def test_mode_can_be_selected_after_inspection_without_reimport(tmp_path, capsys):
    recipe = _write_raw_experiment(tmp_path, n_sites=6)
    inspected = tmp_path / "inspected-local"
    project_cli_main(
        [
            "inspect-experiment",
            str(recipe),
            "--mode",
            "heisenberg",
            "--variant",
            "inhomogeneous",
            "--output-dir",
            str(inspected),
            "--candidate-cutoffs",
            "50",
        ]
    )
    capsys.readouterr()

    homogeneous = tmp_path / "selected-homogeneous"
    assert project_cli_main(
        [
            "select-experiment-mode",
            str(inspected / "experiment_manifest.json"),
            "--mode",
            "heisenberg",
            "--variant",
            "homogeneous",
            "--output-dir",
            str(homogeneous),
        ]
    ) == 0
    console = capsys.readouterr().out
    assert "global" in console
    assert "without re-importing" in console
    measurement, manifest = load_experiment_project(
        homogeneous / "experiment_manifest.json"
    )
    assert measurement.shape[0] == 6
    assert measurement.metadata["experiment_variant"] == "homogeneous"
    assert manifest["system_type"] == "homogeneous_heisenberg"
    assert manifest["view"] == "global"
    assert manifest["mode_selection_provenance"]["data_reimported"] is False


def test_project_inspection_accepts_guided_manifest_without_using_csv_directly(tmp_path):
    recipe = _write_raw_experiment(tmp_path)
    guided_output = tmp_path / "guided"
    project_cli_main(
        [
            "inspect-experiment",
            str(recipe),
            "--mode",
            "heisenberg",
            "--variant",
            "inhomogeneous",
            "--output-dir",
            str(guided_output),
            "--candidate-cutoffs",
            "50",
        ]
    )
    config = ProjectConfig.from_mapping(
        {
            "config_schema_version": 1,
            "name": "manifest experiment",
            "system_type": "inhomogeneous_heisenberg",
            "artifact": "not-loaded-during-inspection",
            "experiment": {"manifest": str(guided_output / "experiment_manifest.json")},
            "output_dir": str(tmp_path / "project inspection"),
            "training": {"cutoffs_mev": [50], "manual_cutoff_mev": 50},
        },
        base_dir=tmp_path,
    )
    summary = HamiltonianLearningProject(config).inspect_experiment()
    assert summary["n_sites"] == 4
    assert summary["bias_max_mev"] == 60.0


def test_homogeneous_mode_is_inspectable_and_inference_supported(tmp_path):
    recipe = _write_raw_experiment(tmp_path)
    output = tmp_path / "homogeneous"
    project_cli_main(
        [
            "inspect-experiment",
            str(recipe),
            "--mode",
            "heisenberg",
            "--variant",
            "homogeneous",
            "--output-dir",
            str(output),
            "--candidate-cutoffs",
            "50",
        ]
    )
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert manifest["status"] == "ready_for_cutoff_selection"
    assert manifest["system_type"] == "homogeneous_heisenberg"
    assert manifest["view"] == "global"
    assert manifest["profile"]["experimental_inference_supported"]
    assert not any("analyzer/report is not implemented" in item for item in manifest["warnings"])


def test_incomplete_canonical_experiment_returns_fix_decision_instead_of_crashing():
    measurement = Measurement(
        axes={"site": [1, 2, 3], "bias": [0.0, 25.0, 50.0]},
        channels={"didv": [[1.0, 2.0, 3.0], [1.0, np.nan, 3.0], [1.0, 2.0, 3.0]]},
        axis_units={"bias": "meV"},
        metadata={
            "system_type": "inhomogeneous_heisenberg",
            "view": "local_bonds",
        },
    )
    decision = advise_experiment(measurement, manual_cutoff_mev=50.0)
    assert decision.action == "fix_experiment_or_choose_lower_cutoff"
    assert any("unresolved missing" in item for item in decision.experiment_checks)
