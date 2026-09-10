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
        self.systems = []

    def simulate(self, system, protocol):
        self.calls += 1
        self.systems.append(system)
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


def test_impurity_recipe_accepts_arbitrary_measured_sites_and_properties(tmp_path):
    impurities = [
        {"site": 0, "spin": "S=1", "transverse_mev": 2.0},
        {"site": 2, "spin": "S=3/2", "axial_mev": -0.4},
        {
            "site": 5,
            "spin": "S=2",
            "transverse_mev": 1.3,
            "transverse_angle_rad": 0.25,
        },
        {"site": 7, "spin": "S=5/2"},
    ]
    config = ProjectConfig.from_mapping(
        {
            "name": "measured impurity configuration",
            "output_dir": "output",
            "dataset": {
                "generate": {
                    "system": "homogeneous_xxz_j1j2j3_dmi_impurity",
                    "n_sites": 8,
                    "n_samples": 10,
                    "coupling_ranges_mev": [
                        [2, 6], [-1.5, 1.5], [-1, 1], [2, 6], [0.3, 2.5]
                    ],
                    "impurities": impurities,
                }
            },
        },
        base_dir=tmp_path,
    )

    assert config.system_type == "homogeneous_xxz_j1j2j3_dmi_impurity"
    assert config.view == "global"
    assert config.generation is not None
    assert [item.site for item in config.generation.impurities] == [0, 2, 5, 7]
    assert config.generation.impurities[2].spin == "S=2"
    assert config.generation.impurities[2].transverse_angle_rad == 0.25
    assert len(config.generation.impurities) == 4

    simulator = LightweightSimulator()
    result = HamiltonianLearningProject(config).generate_training_dataset(
        simulator=simulator
    )
    assert result.dataset.system_type == "homogeneous_xxz_j1j2j3_dmi_impurity"
    assert simulator.systems[0].site_spins == (
        "S=1",
        "S=1/2",
        "S=3/2",
        "S=1/2",
        "S=1/2",
        "S=2",
        "S=1/2",
        "S=5/2",
    )


@pytest.mark.parametrize(
    ("impurities", "message"),
    [
        ([{"spin": "S=1"}], "requires site"),
        ([{"site": 1, "mystery": 2}], "unknown fields"),
        ([{"site": 1.5, "spin": "S=1"}], "site must be an integer"),
        ([{"site": 1}, {"site": 1}], "sites must be distinct"),
        ([{"site": 8}], "must index a site"),
    ],
)
def test_impurity_recipe_rejects_invalid_configurations(tmp_path, impurities, message):
    with pytest.raises(ValueError, match=message):
        ProjectConfig.from_mapping(
            {
                "name": "invalid impurities",
                "dataset": {
                    "generate": {
                        "system": "homogeneous_xxz_j1j2j3_dmi_impurity",
                        "n_sites": 8,
                        "n_samples": 1,
                        "coupling_ranges_mev": [
                            [2, 6], [-1.5, 1.5], [-1, 1], [2, 6], [0.3, 2.5]
                        ],
                        "impurities": impurities,
                    }
                },
            },
            base_dir=tmp_path,
        )


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


def test_run_without_an_experiment_trains_and_stops(tmp_path):
    """A model built before the measurement exists is an ordinary thing to want.

    `run()` used to require an experiment, which meant the configuration the
    guided form writes could not be re-run by the command line it printed --
    the interface had to reimplement the stages instead, and the two could
    drift.
    """
    raw = make_training_dataset(n_samples=40)
    dataset_path = tmp_path / "dataset.npz"
    raw.save(dataset_path)
    config = ProjectConfig(
        name="model first",
        experiment_csv=None,
        output_dir=tmp_path / "output",
        dataset_path=dataset_path,
        cutoffs_mev=(50.0,),
        manual_cutoff_mev=50.0,
        output_points=30,
        model="ridge",
        preset="quick",
        verbose=0,
    )
    outcome = HamiltonianLearningProject(config).run()
    assert outcome.report_path is None, "nothing was inferred, so nothing to report"
    assert outcome.analysis_dir is None
    assert "no experiment" in outcome.status
    assert outcome.selected_cutoff_mev == 50.0
    assert (outcome.artifact_path / "manifest.json").exists()
    summary = json.loads((config.output_dir / "project_summary.json").read_text())
    assert summary["validation_mae_mev"] >= 0
    assert summary["test_mae_mev"] >= 0


def test_run_refuses_a_project_with_neither_an_experiment_nor_training(tmp_path):
    """Reusing an artifact with nothing to analyse has nothing to do."""
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text("{}")
    config = ProjectConfig(
        name="nothing to do",
        experiment_csv=None,
        output_dir=tmp_path / "output",
        artifact_path=artifact,
        manual_cutoff_mev=50.0,
    )
    with pytest.raises(RuntimeError, match="nothing for it to do"):
        HamiltonianLearningProject(config).run()


def test_a_search_runs_before_training_and_records_every_trial(tmp_path):
    """The winning settings must be the ones the artifact was trained with."""
    raw = make_training_dataset(n_samples=40)
    dataset_path = tmp_path / "dataset.npz"
    raw.save(dataset_path)
    from hamlet.project import TuningConfig

    config = ProjectConfig(
        name="tuned",
        experiment_csv=None,
        output_dir=tmp_path / "output",
        dataset_path=dataset_path,
        cutoffs_mev=(50.0,),
        manual_cutoff_mev=50.0,
        output_points=30,
        model="ridge",
        preset="quick",
        tuning=TuningConfig(n_trials=4, preset="quick", seed=1),
        verbose=0,
    )
    project = HamiltonianLearningProject(config)
    project.run()
    report = json.loads((config.output_dir / "tuning.json").read_text())
    assert report["n_trials_evaluated"] == 5, "the defaults plus four trials"
    assert report["selection_split"] == "validation"
    manifest = json.loads((config.output_dir / "artifact" / "manifest.json").read_text())
    assert manifest["model_options"] == report["best_options"]


# --- reading the search settings from a configuration -------------------------

def test_tuning_can_be_asked_for_with_a_bare_true(tmp_path):
    """The common case is "search it", with no opinion about how."""
    from hamlet.project import TuningConfig

    config = ProjectConfig.from_mapping(
        {"name": "x", "output_dir": str(tmp_path), "artifact": str(tmp_path / "a"),
         "training": {"manual_cutoff_mev": 50.0, "cutoffs_mev": [50.0],
                      "tuning": True}},
        base_dir=tmp_path,
    )
    assert config.tuning == TuningConfig()


@pytest.mark.parametrize("value", [None, False])
def test_no_tuning_is_the_default(tmp_path, value):
    config = ProjectConfig.from_mapping(
        {"name": "x", "output_dir": str(tmp_path), "artifact": str(tmp_path / "a"),
         "training": {"manual_cutoff_mev": 50.0, "cutoffs_mev": [50.0],
                      "tuning": value}},
        base_dir=tmp_path,
    )
    assert config.tuning is None


@pytest.mark.parametrize(
    "tuning, expected",
    [
        ({"n_trials": 0}, "at least 1"),
        ({"trials": 5}, "unknown fields"),
        ({"n_trials": 5, "timeout_seconds": 0}, "must be positive"),
        ("twenty", "must be a mapping"),
    ],
)
def test_unusable_tuning_settings_are_refused_at_load(tmp_path, tuning, expected):
    """A typo in a search budget must not be discovered hours into a run."""
    with pytest.raises(ValueError, match=expected):
        ProjectConfig.from_mapping(
            {"name": "x", "output_dir": str(tmp_path), "artifact": str(tmp_path / "a"),
             "training": {"manual_cutoff_mev": 50.0, "cutoffs_mev": [50.0],
                          "tuning": tuning}},
            base_dir=tmp_path,
        )


def test_tuning_settings_survive_into_the_resolved_configuration(tmp_path):
    """The resolved record is the reproducibility contract for a run."""
    from hamlet.project import TuningConfig

    raw = make_training_dataset(n_samples=20)
    dataset = tmp_path / "dataset.npz"
    raw.save(dataset)
    config = ProjectConfig(
        name="recorded",
        experiment_csv=None,
        output_dir=tmp_path / "output",
        dataset_path=dataset,
        cutoffs_mev=(50.0,),
        manual_cutoff_mev=50.0,
        output_points=30,
        model="ridge",
        preset="quick",
        tuning=TuningConfig(n_trials=2, preset="quick", seed=5),
        verbose=0,
    )
    project = HamiltonianLearningProject(config)
    project._record_resolved_config()
    resolved = json.loads(
        (config.output_dir / "resolved_project_config.json").read_text()
    )
    assert resolved["tuning"]["n_trials"] == 2
    assert resolved["tuning"]["seed"] == 5


def test_tuning_without_a_project_that_configures_it_is_refused(tmp_path):
    raw = make_training_dataset(n_samples=20)
    dataset = tmp_path / "dataset.npz"
    raw.save(dataset)
    config = ProjectConfig(
        name="untuned",
        experiment_csv=None,
        output_dir=tmp_path / "output",
        dataset_path=dataset,
        cutoffs_mev=(50.0,),
        manual_cutoff_mev=50.0,
        output_points=30,
        model="ridge",
        preset="quick",
        verbose=0,
    )
    with pytest.raises(RuntimeError, match="does not configure training.tuning"):
        HamiltonianLearningProject(config).tune()


def test_a_search_can_be_run_and_inspected_without_training(tmp_path):
    """Separate from train() so a search can be repeated without committing to
    the full budget it feeds."""
    from hamlet.project import TuningConfig

    raw = make_training_dataset(n_samples=40)
    dataset = tmp_path / "dataset.npz"
    raw.save(dataset)
    config = ProjectConfig(
        name="search only",
        experiment_csv=None,
        output_dir=tmp_path / "output",
        dataset_path=dataset,
        cutoffs_mev=(50.0,),
        manual_cutoff_mev=50.0,
        output_points=30,
        model="ridge",
        preset="quick",
        tuning=TuningConfig(n_trials=3, seed=2),
        verbose=0,
    )
    project = HamiltonianLearningProject(config)
    report = project.tune()
    assert report.model == "ridge"
    assert len(report.trials) == 4
    assert project.tuning_report is report
    assert (config.output_dir / "tuning.json").exists()
    # Nothing was trained: the artifact directory is still absent.
    assert not (config.output_dir / "artifact").exists()


def test_training_directly_works_without_an_experiment(tmp_path):
    """`train()` used to demand a calibration step that cannot exist without a
    measurement, so calling it on a model-only project failed on the path the
    interface's own runs take."""
    raw = make_training_dataset(n_samples=40)
    dataset = tmp_path / "dataset.npz"
    raw.save(dataset)
    config = ProjectConfig(
        name="train directly",
        experiment_csv=None,
        output_dir=tmp_path / "output",
        dataset_path=dataset,
        cutoffs_mev=(50.0,),
        manual_cutoff_mev=50.0,
        output_points=30,
        model="ridge",
        preset="quick",
        verbose=0,
    )
    project = HamiltonianLearningProject(config)
    run = project.train()
    assert run.model_name == "ridge"
    assert project.selected_cutoff_mev == 50.0
    # Uncalibrated deliberately: there was no measurement to calibrate against.
    assert project.selected_augmentation is None


def test_preparation_is_not_repeated_once_it_has_happened(tmp_path):
    raw = make_training_dataset(n_samples=20)
    dataset = tmp_path / "dataset.npz"
    raw.save(dataset)
    config = ProjectConfig(
        name="prepared once",
        experiment_csv=None,
        output_dir=tmp_path / "output",
        dataset_path=dataset,
        cutoffs_mev=(50.0,),
        manual_cutoff_mev=50.0,
        output_points=30,
        model="ridge",
        preset="quick",
        verbose=0,
    )
    project = HamiltonianLearningProject(config)
    first = project.prepare_training_data_without_experiment()
    assert project._ensure_prepared() is first
