import json

import numpy as np
import pandas as pd
import pytest

from hamlet.project import HamiltonianLearningProject, ProjectConfig
from hamlet.simulation import SpectroscopyResult
from hamlet.training import AugmentationConfig, augment_experimental_like
from test_guided_training import make_training_dataset


def _write_long_csv(path, bias, spectra):
    rows = []
    for site, spectrum in enumerate(spectra, start=1):
        for energy, value in zip(bias, spectrum):
            rows.append({"site": site, "bias_meV": energy, "didv_A": value})
    pd.DataFrame(rows).to_csv(path, index=False)


class LightweightSimulator:
    def __init__(self):
        self.calls = 0

    def simulate(self, system, protocol):
        self.calls += 1
        scale = float(np.sum(system.as_array()))
        spectra = np.stack(
            [scale + site + np.asarray(protocol.bias_mev) for site in range(system.n_sites)]
        )
        return SpectroscopyResult(protocol.bias_mev, spectra)


def test_project_config_resolves_relative_yaml_paths(tmp_path):
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        """
name: demo
artifact: artifact
experiment:
  csv: experiment.csv
output_dir: output
training:
  cutoffs_mev: [40, 50]
  manual_cutoff_mev: 50
"""
    )
    config = ProjectConfig.from_file(config_path)
    assert config.artifact_path == (tmp_path / "artifact").resolve()
    assert config.experiment_csv == (tmp_path / "experiment.csv").resolve()
    assert config.output_dir == (tmp_path / "output").resolve()
    assert config.cutoffs_mev == (40.0, 50.0)
    assert config.manual_cutoff_mev == 50.0
    assert config.config_schema_version == 1


def test_project_config_rejects_unknown_schema_version(tmp_path):
    with pytest.raises(ValueError, match="unsupported config_schema_version"):
        ProjectConfig.from_mapping(
            {
                "config_schema_version": 99,
                "name": "future config",
                "artifact": "artifact",
            },
            base_dir=tmp_path,
        )


def test_manual_cutoff_must_be_one_of_configured_candidates(tmp_path):
    with pytest.raises(ValueError, match="included"):
        ProjectConfig.from_mapping(
            {
                "name": "bad manual cutoff",
                "artifact": "artifact",
                "training": {"cutoffs_mev": [40, 50], "manual_cutoff_mev": 45},
            },
            base_dir=tmp_path,
        )


def test_experiment_project_requires_explicit_manual_cutoff(tmp_path):
    with pytest.raises(ValueError, match="does not choose"):
        ProjectConfig.from_mapping(
            {
                "name": "missing manual cutoff",
                "artifact": "artifact",
                "experiment": {"csv": "experiment.csv"},
            },
            base_dir=tmp_path,
        )


def test_project_generates_dataset_from_yaml_and_reuses_cache(tmp_path):
    config_path = tmp_path / "generate.yaml"
    config_path.write_text(
        """
name: generated demo
system_type: inhomogeneous_heisenberg
output_dir: output
dataset:
  generate:
    system: inhomogeneous_heisenberg
    output: generated/train.npz
    n_sites: 6
    n_samples: 5
    coupling_range_mev: [30, 45]
    bias_range_mev: [0, 60]
    bias_points: 13
    broadening_mev: 0.5
    output_quantity: didv
    backend: dmrgpy
    seed: 4
    checkpoint_every: 2
"""
    )
    config = ProjectConfig.from_file(config_path)
    assert config.experiment_csv is None
    assert config.generation is not None
    assert config.generation.output_path == (tmp_path / "generated/train.npz").resolve()

    simulator = LightweightSimulator()
    project = HamiltonianLearningProject(config)
    generated = project.generate_training_dataset(simulator=simulator)
    assert generated.dataset.spectra.shape == (5, 6, 13)
    assert generated.dataset.system_type == "inhomogeneous_heisenberg"
    assert simulator.calls == 5

    unused = LightweightSimulator()
    cached = HamiltonianLearningProject(config).generate_training_dataset(simulator=unused)
    assert cached.cache_hit
    assert unused.calls == 0


def test_homogeneous_generation_recipe_defaults_to_global_view(tmp_path):
    config = ProjectConfig.from_mapping(
        {
            "name": "J1-J2 generation",
            "output_dir": "output",
            "dataset": {
                "generate": {
                    "system": "homogeneous_heisenberg",
                    "n_sites": 8,
                    "n_samples": 10,
                    "coupling_ranges_mev": [[30, 45], [0, 10]],
                }
            },
        },
        base_dir=tmp_path,
    )
    assert config.system_type == "homogeneous_heisenberg"
    assert config.view == "global"
    assert config.generation is not None
    assert config.generation.coupling_ranges_mev == ((30.0, 45.0), (0.0, 10.0))


def test_xxz_long_range_recipe_has_named_four_parameter_contract(tmp_path):
    config = ProjectConfig.from_mapping(
        {
            "name": "XXZ pilot",
            "output_dir": "output",
            "dataset": {
                "generate": {
                    "system": "homogeneous_xxz_j1j2j3",
                    "n_sites": 8,
                    "n_samples": 10,
                    "coupling_ranges_mev": [
                        [25, 45], [-8, 8], [-5, 5], [25, 45]
                    ],
                    "observable": "total_spin",
                    "observable_weights": [1, 1, 1],
                }
            },
        },
        base_dir=tmp_path,
    )
    assert config.system_type == "homogeneous_xxz_j1j2j3"
    assert config.view == "global"
    assert config.generation is not None
    assert len(config.generation.coupling_ranges_mev) == 4
    assert config.generation.observable == "total_spin"
    assert config.generation.observable_weights == (1.0, 1.0, 1.0)


def test_xxz_dmi_recipe_requires_positive_fifth_range(tmp_path):
    payload = {
        "name": "XXZ DMI pilot",
        "output_dir": "output",
        "dataset": {
            "generate": {
                "system": "homogeneous_xxz_j1j2j3_dmi",
                "n_sites": 8,
                "n_samples": 10,
                "coupling_ranges_mev": [
                    [25, 45], [-8, 8], [-5, 5], [25, 45], [0, 6]
                ],
                "observable": "total_spin",
            }
        },
    }
    config = ProjectConfig.from_mapping(payload, base_dir=tmp_path)
    assert config.view == "global"
    assert config.generation is not None
    assert len(config.generation.coupling_ranges_mev) == 5
    payload["dataset"]["generate"]["coupling_ranges_mev"][-1] = [-1, 6]
    with pytest.raises(ValueError, match="D_z magnitude"):
        ProjectConfig.from_mapping(payload, base_dir=tmp_path)


def test_project_calibrates_trains_infers_and_builds_html(tmp_path):
    raw = make_training_dataset(n_samples=40)
    dataset_path = tmp_path / "dataset.npz"
    raw.save(dataset_path)
    augmentation = AugmentationConfig(
        broadening_points=(1.0, 1.0), noise=0.0, seed=12
    )
    experimental_like = augment_experimental_like(
        raw,
        broadening_points=augmentation.broadening_points,
        noise=augmentation.noise,
        seed=augmentation.seed,
    )
    experiment_path = tmp_path / "experiment.csv"
    _write_long_csv(
        experiment_path, experimental_like.bias_mev, experimental_like.spectra[7]
    )
    config = ProjectConfig(
        name="Unit-test chain",
        experiment_csv=experiment_path,
        output_dir=tmp_path / "output",
        dataset_path=dataset_path,
        cutoffs_mev=(50.0, 70.0),
        manual_cutoff_mev=50.0,
        output_points=30,
        model="ridge",
        preset="quick",
        model_options={"alpha": 0.1},
        verbose=0,
    )
    project = HamiltonianLearningProject(config)
    inspection = project.inspect_experiment()
    assert inspection["n_sites"] == raw.n_sites
    resolved = json.loads(
        (config.output_dir / "resolved_project_config.json").read_text()
    )
    assert resolved["config_schema_version"] == 1
    assert resolved["manual_cutoff_mev"] == 50.0
    project.calibrate_preprocessing(candidates={"matched": augmentation})
    assert project.selected_cutoff_mev == 50.0
    assert tuple(project.calibrations) == (50.0,)
    project.prepare_training_data()
    run = project.train()
    assert run.model_name == "ridge"
    result = project.infer()
    assert result.n_bonds == raw.n_sites - 1

    html_path = config.output_dir / "analysis" / "report.html"
    assert html_path.exists()
    html = html_path.read_text()
    assert "Unit-test chain" in html
    assert "Inferred exchange couplings" in html
    assert "Generated by HamLeT" in html
    report = json.loads((config.output_dir / "analysis" / "report.json").read_text())
    assert report["diagnostics"]["aggregation_method"] == run.aggregation.method
