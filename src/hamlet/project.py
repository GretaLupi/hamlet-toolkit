"""Configuration-driven, experimentalist-facing project workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .branding import brand_manifest
from .data import (
    CheckpointedGenerationResult,
    SpectroscopyDataset,
    generate_dataset_checkpointed,
)
from .experimental import ExperimentalChainResult, ExperimentalGlobalResult
from .experiments import load_canonical_experiment
from .io import load_reference_heisenberg_datasets
from .simulation import DmrgpySimulator, SpectroscopyProtocol, SpectroscopySimulator
from .systems import (
    HomogeneousHeisenbergFamily,
    HomogeneousXXZLongRangeFamily,
    HomogeneousXXZDMILongRangeFamily,
    InhomogeneousHeisenbergFamily,
)
from .training import (
    AugmentationCalibrationResult,
    AugmentationConfig,
    TrainingPreprocessingConfig,
    TrainingRun,
    augment_experimental_like,
    calibrate_augmentation,
    prepare_training_dataset,
    train_supervised,
)

PROJECT_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetGenerationConfig:
    """Resolved recipe for generating a portable Heisenberg dataset."""

    system_type: str
    output_path: Path
    n_sites: int
    n_samples: int
    coupling_range_mev: tuple[float, float] | None = None
    coupling_ranges_mev: tuple[tuple[float, float], ...] = ()
    bias_range_mev: tuple[float, float] = (0.0, 100.0)
    bias_points: int = 200
    broadening_mev: float = 0.5
    observable: str = "Sz"
    observable_weights: tuple[float, float, float] | None = None
    output_quantity: str = "didv"
    backend: str = "dmrgpy"
    max_bond_dimension: int = 20
    kpm_max_bond_dimension: int = 20
    seed: int = 42
    checkpoint_every: int = 25

    def __post_init__(self) -> None:
        if self.system_type not in {
            "inhomogeneous_heisenberg",
            "homogeneous_heisenberg",
            "homogeneous_xxz_j1j2j3",
            "homogeneous_xxz_j1j2j3_dmi",
        }:
            raise ValueError(
                "generated system must be inhomogeneous_heisenberg, "
                "homogeneous_heisenberg, homogeneous_xxz_j1j2j3, or "
                "homogeneous_xxz_j1j2j3_dmi"
            )
        if self.n_sites < 2 or self.n_samples < 1:
            raise ValueError("n_sites must be at least 2 and n_samples positive")
        if self.bias_points < 2 or self.checkpoint_every < 1:
            raise ValueError("bias_points must be at least 2 and checkpoint_every positive")
        if (
            len(self.bias_range_mev) != 2
            or not self.bias_range_mev[0] < self.bias_range_mev[1]
        ):
            raise ValueError("bias_range_mev must contain increasing low and high values")
        if self.broadening_mev <= 0:
            raise ValueError("broadening_mev must be positive")
        if self.max_bond_dimension < 1 or self.kpm_max_bond_dimension < 1:
            raise ValueError("DMRGPy bond dimensions must be positive")
        if self.backend != "dmrgpy":
            raise ValueError("the configuration interface currently supports backend: dmrgpy")
        # Reuse the simulator-independent protocol validation here so invalid
        # observable contracts fail before a long generation job starts.
        SpectroscopyProtocol.uniform(
            self.bias_range_mev,
            points=self.bias_points,
            broadening_mev=self.broadening_mev,
            observable=self.observable,
            observable_weights=self.observable_weights,
            output_quantity=self.output_quantity,
        )
        if self.system_type == "inhomogeneous_heisenberg":
            if self.coupling_range_mev is None or self.coupling_ranges_mev:
                raise ValueError("inhomogeneous generation requires only coupling_range_mev")
            if (
                len(self.coupling_range_mev) != 2
                or not self.coupling_range_mev[0] < self.coupling_range_mev[1]
            ):
                raise ValueError("coupling_range_mev must contain increasing low and high values")
        elif not self.coupling_ranges_mev or self.coupling_range_mev is not None:
            raise ValueError("homogeneous generation requires only coupling_ranges_mev")
        elif any(
            len(interval) != 2 or not interval[0] < interval[1]
            for interval in self.coupling_ranges_mev
        ):
            raise ValueError("every coupling_ranges_mev interval must be increasing")
        if (
            self.system_type == "homogeneous_xxz_j1j2j3"
            and len(self.coupling_ranges_mev) != 4
        ):
            raise ValueError(
                "homogeneous_xxz_j1j2j3 requires four ranges ordered "
                "J1_xy, J2, J3, Jz"
            )
        if (
            self.system_type == "homogeneous_xxz_j1j2j3_dmi"
            and len(self.coupling_ranges_mev) != 5
        ):
            raise ValueError(
                "homogeneous_xxz_j1j2j3_dmi requires five ranges ordered "
                "J1_xy, J2, J3, Jz, D_z"
            )
        if (
            self.system_type == "homogeneous_xxz_j1j2j3_dmi"
            and self.coupling_ranges_mev[4][0] < 0.0
        ):
            raise ValueError("D_z magnitude range cannot include negative values")

    def to_recipe(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("output_path")
        return values


@dataclass(frozen=True)
class ProjectConfig:
    """Portable configuration for one training and analysis project."""

    name: str
    experiment_csv: Path | None
    output_dir: Path
    config_schema_version: int = PROJECT_CONFIG_SCHEMA_VERSION
    system_type: str = "inhomogeneous_heisenberg"
    dataset_path: Path | None = None
    reference_npz: tuple[Path, ...] = ()
    generation: DatasetGenerationConfig | None = None
    artifact_path: Path | None = None
    cutoffs_mev: tuple[float, ...] = (50.0,)
    manual_cutoff_mev: float | None = None
    cutoff_policy: str = "largest_passing"
    output_points: int = 200
    view: str = "local_bonds"
    model: str = "keras_mlp"
    preset: str = "standard"
    model_options: dict[str, Any] = field(default_factory=dict)
    allow_development_artifacts: bool = False
    max_validation_mae_mev: float | None = None
    max_test_mae_mev: float | None = None
    verbose: int = 1

    def __post_init__(self) -> None:
        if self.config_schema_version != PROJECT_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported config_schema_version {self.config_schema_version}; "
                f"this package supports {PROJECT_CONFIG_SCHEMA_VERSION}"
            )
        if not self.name:
            raise ValueError("project name cannot be empty")
        sources = (
            int(self.dataset_path is not None)
            + int(bool(self.reference_npz))
            + int(self.generation is not None)
        )
        if self.artifact_path is None and sources != 1:
            raise ValueError("training requires exactly one dataset source")
        if self.artifact_path is not None and sources > 1:
            raise ValueError("configure at most one dataset source")
        if not self.cutoffs_mev or any(value <= 0 for value in self.cutoffs_mev):
            raise ValueError("cutoffs_mev must contain positive values")
        if self.manual_cutoff_mev is not None:
            if self.manual_cutoff_mev <= 0:
                raise ValueError("manual_cutoff_mev must be positive")
            if not any(
                np.isclose(self.manual_cutoff_mev, item, rtol=0.0, atol=1e-8)
                for item in self.cutoffs_mev
            ):
                raise ValueError("manual_cutoff_mev must be included in cutoffs_mev")
        if self.experiment_csv is not None and self.manual_cutoff_mev is None:
            raise ValueError(
                "projects with an experiment require training.manual_cutoff_mev; "
                "the package does not choose the usable experimental cutoff"
            )
        if self.cutoff_policy not in {"largest_passing", "smallest_passing"}:
            raise ValueError("cutoff_policy must be largest_passing or smallest_passing")
        if self.output_points < 2:
            raise ValueError("output_points must be at least 2")
        for name, value in (
            ("max_validation_mae_mev", self.max_validation_mae_mev),
            ("max_test_mae_mev", self.max_test_mae_mev),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_file(cls, path: str | Path) -> "ProjectConfig":
        config_path = Path(path).resolve()
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ImportError("YAML configuration requires PyYAML") from exc
            payload = yaml.safe_load(text)
        if not isinstance(payload, Mapping):
            raise ValueError("project configuration must contain a mapping")
        return cls.from_mapping(payload, base_dir=config_path.parent)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, base_dir: str | Path = "."
    ) -> "ProjectConfig":
        base = Path(base_dir).resolve()
        dataset = payload.get("dataset") or {}
        experiment = payload.get("experiment") or {}
        training = payload.get("training") or {}
        if not isinstance(dataset, Mapping) or not isinstance(experiment, Mapping):
            raise ValueError("dataset and experiment sections must be mappings")
        if not isinstance(training, Mapping):
            raise ValueError("training section must be a mapping")

        output_dir = _resolve_path(base, payload.get("output_dir", "hamlet-output"))
        dataset_format = dataset.get(
            "format", "generated" if "generate" in dataset else "portable"
        )
        dataset_path = None
        reference_npz: tuple[Path, ...] = ()
        generation = None
        if dataset:
            if dataset_format == "portable":
                if "path" not in dataset:
                    raise ValueError("portable dataset requires dataset.path")
                dataset_path = _resolve_path(base, dataset["path"])
            elif dataset_format == "reference_heisenberg":
                paths = dataset.get("paths") or []
                if not paths:
                    raise ValueError("reference_heisenberg dataset requires dataset.paths")
                reference_npz = tuple(_resolve_path(base, item) for item in paths)
            elif dataset_format == "generated":
                values = dataset.get("generate")
                if not isinstance(values, Mapping):
                    raise ValueError("generated dataset requires a dataset.generate mapping")
                generation = _parse_generation_config(values, base, output_dir)
            else:
                raise ValueError(
                    "dataset.format must be portable, reference_heisenberg, or generated"
                )

        experiment_sources = [
            experiment[key]
            for key in ("manifest", "measurement", "csv")
            if experiment.get(key) is not None
        ]
        if len(experiment_sources) > 1:
            raise ValueError("experiment must define only one of manifest, measurement, or csv")
        experiment_value = (
            experiment_sources[0]
            if experiment_sources
            else payload.get("experiment_csv")
        )
        artifact_value = payload.get("artifact")
        inferred_system = (
            generation.system_type if generation is not None else "inhomogeneous_heisenberg"
        )
        system_type = str(payload.get("system_type", inferred_system))
        if generation is not None and generation.system_type != system_type:
            raise ValueError("dataset.generate.system must match project system_type")
        return cls(
            name=str(payload.get("name", "HamLeT project")),
            config_schema_version=int(
                payload.get("config_schema_version", PROJECT_CONFIG_SCHEMA_VERSION)
            ),
            system_type=system_type,
            experiment_csv=(
                _resolve_path(base, experiment_value) if experiment_value is not None else None
            ),
            output_dir=output_dir,
            dataset_path=dataset_path,
            reference_npz=reference_npz,
            generation=generation,
            artifact_path=_resolve_path(base, artifact_value) if artifact_value else None,
            cutoffs_mev=tuple(float(item) for item in training.get("cutoffs_mev", [50.0])),
            manual_cutoff_mev=(
                float(training["manual_cutoff_mev"])
                if training.get("manual_cutoff_mev") is not None
                else None
            ),
            cutoff_policy=str(training.get("cutoff_policy", "largest_passing")),
            output_points=int(training.get("output_points", 200)),
            view=str(
                training.get(
                    "view",
                    "global" if system_type.startswith("homogeneous_") else "local_bonds",
                )
            ),
            model=str(training.get("model", "keras_mlp")),
            preset=str(training.get("preset", "standard")),
            model_options=dict(training.get("model_options") or {}),
            allow_development_artifacts=bool(
                training.get("allow_development_artifacts", False)
            ),
            max_validation_mae_mev=(
                float(training["max_validation_mae_mev"])
                if training.get("max_validation_mae_mev") is not None
                else None
            ),
            max_test_mae_mev=(
                float(training["max_test_mae_mev"])
                if training.get("max_test_mae_mev") is not None
                else None
            ),
            verbose=int(training.get("verbose", 1)),
        )


@dataclass(frozen=True)
class ProjectOutcome:
    selected_cutoff_mev: float
    artifact_path: Path
    analysis_dir: Path
    report_path: Path
    status: str


# Order-of-magnitude anchor for the generation budget, measured on this
# project's own reference runs: one L=8 chain on a 61-point 0-60 meV grid with
# DMRGPy `dynamics_mode="ED"` costs roughly this per evaluated correlator on
# one core. `total_spin` evaluates Sxx, Syy and Szz and so costs about three
# times a single `Sz` run. It is a scale anchor, not a promise: a different
# chain length, bias resolution, bond dimension, or DMRG rather than ED moves
# it substantially. Override with `seconds_per_chain` when a local measurement
# is available.
REFERENCE_SECONDS_PER_CORRELATOR_L8 = 25.0


@dataclass(frozen=True)
class PlannedOutput:
    """One path the run would write, and whether it already exists."""

    path: Path
    description: str
    exists: bool
    blocks_run: bool


@dataclass(frozen=True)
class ProjectPlan:
    """What a project would do, resolved without executing or writing anything.

    Answers the questions worth asking before starting a job that may run for
    hours: which stages execute, how much simulation is implied, what gets
    written, and which existing files would make the run refuse partway
    through.
    """

    name: str
    system_type: str
    view: str
    stages: tuple[str, ...]
    dataset_source: str
    dataset_detail: dict[str, Any]
    generation_chains: int | None
    estimated_generation_seconds: float | None
    seconds_per_chain: float | None
    outputs: tuple[PlannedOutput, ...]
    notes: tuple[str, ...]
    blocking_issues: tuple[str, ...]

    @property
    def would_refuse(self) -> bool:
        return bool(self.blocking_issues) or any(item.blocks_run for item in self.outputs)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "plan_schema_version": 1,
            "toolkit": brand_manifest(),
            "name": self.name,
            "system_type": self.system_type,
            "view": self.view,
            "stages": list(self.stages),
            "dataset_source": self.dataset_source,
            "dataset_detail": _json_safe(self.dataset_detail),
            "generation_chains": self.generation_chains,
            "estimated_generation_seconds": self.estimated_generation_seconds,
            "seconds_per_chain": self.seconds_per_chain,
            "outputs": [
                {
                    "path": str(item.path),
                    "description": item.description,
                    "exists": item.exists,
                    "blocks_run": item.blocks_run,
                }
                for item in self.outputs
            ],
            "notes": list(self.notes),
            "blocking_issues": list(self.blocking_issues),
            "would_refuse": self.would_refuse,
        }
        return payload


class HamiltonianLearningProject:
    """Stateful façade over calibration, training, inference, and reporting."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.dataset: SpectroscopyDataset | None = None
        self.calibrations: dict[float, AugmentationCalibrationResult] = {}
        self.selected_cutoff_mev: float | None = None
        self.selected_augmentation: AugmentationConfig | None = None
        self.prepared: Any | None = None
        self.training_run: TrainingRun | None = None
        self.experimental_result: ExperimentalChainResult | ExperimentalGlobalResult | None = None
        self.generation_result: CheckpointedGenerationResult | None = None
        self.workflow_decision: Any | None = None

    @classmethod
    def from_config(cls, path: str | Path) -> "HamiltonianLearningProject":
        return cls(ProjectConfig.from_file(path))

    def load_dataset(self) -> SpectroscopyDataset:
        if self.dataset is None:
            if self.config.dataset_path is not None:
                self.dataset = SpectroscopyDataset.load(self.config.dataset_path)
            elif self.config.reference_npz:
                self.dataset = load_reference_heisenberg_datasets(self.config.reference_npz)
            elif self.config.generation is not None:
                self.dataset = self.generate_training_dataset().dataset
            else:
                raise RuntimeError("this inference-only project has no dataset")
            if self.dataset.system_type != self.config.system_type:
                raise ValueError(
                    f"dataset system {self.dataset.system_type!r} does not match "
                    f"project system {self.config.system_type!r}"
                )
        return self.dataset

    def _load_experiment(self) -> tuple[Any, str]:
        if self.config.experiment_csv is None:
            raise RuntimeError("an experiment input is required")
        measurement, source, declared_system, declared_view = load_canonical_experiment(
            self.config.experiment_csv
        )
        if declared_system is not None and declared_system != self.config.system_type:
            raise ValueError(
                f"experiment manifest system {declared_system!r} does not match "
                f"project system {self.config.system_type!r}"
            )
        if declared_view is not None and declared_view != self.config.view:
            raise ValueError(
                f"experiment manifest view {declared_view!r} does not match "
                f"project view {self.config.view!r}"
            )
        return measurement, source

    def generate_training_dataset(
        self,
        *,
        simulator: SpectroscopySimulator | None = None,
        progress: Any | None = None,
    ) -> CheckpointedGenerationResult:
        """Generate or safely reuse the configured portable training dataset."""
        recipe = self.config.generation
        if recipe is None:
            raise RuntimeError("project configuration has no dataset.generate recipe")
        self._record_resolved_config()
        if recipe.system_type == "inhomogeneous_heisenberg":
            assert recipe.coupling_range_mev is not None
            family = InhomogeneousHeisenbergFamily(
                recipe.n_sites, recipe.coupling_range_mev
            )
        elif recipe.system_type == "homogeneous_heisenberg":
            family = HomogeneousHeisenbergFamily(
                recipe.n_sites, recipe.coupling_ranges_mev
            )
        elif recipe.system_type == "homogeneous_xxz_j1j2j3":
            family = HomogeneousXXZLongRangeFamily(
                recipe.n_sites, recipe.coupling_ranges_mev
            )
        else:
            family = HomogeneousXXZDMILongRangeFamily(
                recipe.n_sites, recipe.coupling_ranges_mev
            )
        protocol = SpectroscopyProtocol.uniform(
            bias_range_mev=recipe.bias_range_mev,
            points=recipe.bias_points,
            broadening_mev=recipe.broadening_mev,
            observable=recipe.observable,
            observable_weights=recipe.observable_weights,
            output_quantity=recipe.output_quantity,
        )
        resolved_simulator = simulator or DmrgpySimulator(
            max_bond_dimension=recipe.max_bond_dimension,
            kpm_max_bond_dimension=recipe.kpm_max_bond_dimension,
        )
        resolved_progress = progress
        if resolved_progress is None and self.config.verbose:
            resolved_progress = _console_progress
        self.generation_result = generate_dataset_checkpointed(
            family,
            resolved_simulator,
            protocol,
            n_samples=recipe.n_samples,
            output_path=recipe.output_path,
            recipe=recipe.to_recipe(),
            seed=recipe.seed,
            checkpoint_every=recipe.checkpoint_every,
            progress=resolved_progress,
        )
        self.dataset = self.generation_result.dataset
        return self.generation_result

    def plan(self, *, seconds_per_chain: float | None = None) -> ProjectPlan:
        """Resolve what ``run`` would do, without executing or writing anything.

        Deliberately side-effect free: it does not create the output directory,
        touch the experiment, load a dataset, or import a simulation backend.
        Everything reported comes from the configuration plus what already
        exists on disk.
        """
        config = self.config
        output_dir = config.output_dir
        reuses_artifact = config.artifact_path is not None

        stages: list[str] = []
        notes: list[str] = []
        blocking: list[str] = []
        outputs: list[PlannedOutput] = []

        def add_output(path: Path, description: str, *, blocks: bool) -> None:
            exists = path.exists()
            outputs.append(
                PlannedOutput(
                    path=path,
                    description=description,
                    exists=exists,
                    blocks_run=exists and blocks,
                )
            )

        # --- dataset stage -------------------------------------------------
        generation_chains: int | None = None
        estimated_seconds: float | None = None
        rate = seconds_per_chain
        dataset_detail: dict[str, Any] = {}

        if config.generation is not None:
            recipe = config.generation
            dataset_source = "generate"
            generation_chains = recipe.n_samples
            correlators = 3 if recipe.observable == "total_spin" else 1
            if rate is None:
                rate = REFERENCE_SECONDS_PER_CORRELATOR_L8 * correlators
            estimated_seconds = generation_chains * rate
            dataset_detail = {
                "system": recipe.system_type,
                "n_sites": recipe.n_sites,
                "n_samples": recipe.n_samples,
                "bias_range_mev": list(recipe.bias_range_mev),
                "bias_points": recipe.bias_points,
                "broadening_mev": recipe.broadening_mev,
                "observable": recipe.observable,
                "evaluated_correlators": correlators,
                "backend": recipe.backend,
                "seed": recipe.seed,
                "output_path": recipe.output_path,
            }
            already_generated = recipe.output_path.exists()
            outputs.append(
                PlannedOutput(
                    path=recipe.output_path,
                    description=(
                        "generated dataset (exists: reused when the recipe fingerprint "
                        "matches, refused when it differs)"
                        if already_generated
                        else "generated dataset"
                    ),
                    exists=already_generated,
                    blocks_run=False,
                )
            )
            add_output(
                recipe.output_path.with_suffix(".generation.json"),
                "recipe fingerprint sidecar",
                blocks=False,
            )
            if already_generated:
                notes.append(
                    f"{recipe.output_path} already exists; an identical recipe is a cache "
                    "hit and costs nothing, while a changed recipe is refused rather than "
                    "overwritten. The generation estimate below assumes a full run."
                )
            stages.append("generate simulations")
            if recipe.n_sites < 3 and config.view == "local_bonds":
                blocking.append(
                    f"local_bonds needs at least three sites, recipe generates {recipe.n_sites}"
                )
        elif config.dataset_path is not None:
            dataset_source = "portable"
            dataset_detail = {"path": config.dataset_path}
            if not config.dataset_path.exists():
                blocking.append(f"dataset does not exist: {config.dataset_path}")
        elif config.reference_npz:
            dataset_source = "reference_heisenberg"
            dataset_detail = {"paths": list(config.reference_npz)}
            for item in config.reference_npz:
                if not item.exists():
                    blocking.append(f"reference dataset does not exist: {item}")
        else:
            dataset_source = "none (inference from a saved artifact)"

        # --- experiment stage ----------------------------------------------
        if config.experiment_csv is not None:
            stages.append("inspect experiment")
            add_output(
                output_dir / "experiment_inspection.json", "experiment inspection", blocks=False
            )
            if not config.experiment_csv.exists():
                blocking.append(f"experiment input does not exist: {config.experiment_csv}")
        else:
            notes.append(
                "no experiment configured: this project can only generate or train, not infer"
            )

        add_output(
            output_dir / "resolved_project_config.json",
            "resolved configuration (refuses a directory holding a different one)",
            blocks=False,
        )

        # --- training or reuse ---------------------------------------------
        if reuses_artifact:
            stages.append("preflight the saved artifact")
            artifact_dir = config.artifact_path
            dataset_detail.setdefault("artifact", artifact_dir)
            if not (artifact_dir / "manifest.json").exists():
                blocking.append(f"artifact has no manifest.json: {artifact_dir}")
            for name in ("workflow_decision.json", "workflow_decision.html"):
                add_output(output_dir / "preflight" / name, "preflight decision", blocks=True)
        else:
            if config.experiment_csv is not None:
                stages.append("calibrate the manual cutoff")
                cutoffs = (
                    (config.manual_cutoff_mev,)
                    if config.manual_cutoff_mev is not None
                    else config.cutoffs_mev
                )
                for cutoff in cutoffs:
                    add_output(
                        output_dir / "calibration" / f"cutoff-{_cutoff_tag(cutoff)}.json",
                        f"calibration at {cutoff:g} meV",
                        blocks=False,
                    )
                add_output(
                    output_dir / "calibration" / "summary.json",
                    "calibration summary",
                    blocks=False,
                )
            stages.append(f"train {config.model} ({config.preset} preset, view={config.view})")
            artifact_dir = output_dir / "artifact"
            artifact_populated = artifact_dir.exists() and any(artifact_dir.iterdir())
            outputs.append(
                PlannedOutput(
                    path=artifact_dir,
                    description="trained artifact (refuses a non-empty directory)",
                    exists=artifact_populated,
                    blocks_run=artifact_populated,
                )
            )

        # --- inference ------------------------------------------------------
        if config.experiment_csv is not None:
            stages.append("infer and report")
            analysis = output_dir / "analysis"
            for name, description in (
                ("couplings.csv", "coupling table"),
                ("report.json", "machine-readable report"),
                ("report.html", "self-contained HTML report"),
                ("summary.png", "quality-control figure"),
            ):
                add_output(analysis / name, description, blocks=True)
            add_output(output_dir / "project_summary.json", "project summary", blocks=False)

        # --- static contract checks ----------------------------------------
        # A missing manual cutoff alongside an experiment is not checked here:
        # ProjectConfig already refuses to construct in that case, so the
        # configuration cannot reach this method.
        if reuses_artifact and config.manual_cutoff_mev is not None:
            manifest_path = config.artifact_path / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    stored = float(manifest["preprocessing"]["bias_cutoff_mev"])
                except (KeyError, TypeError, ValueError, OSError):
                    notes.append(f"could not read the stored cutoff from {manifest_path}")
                else:
                    if not np.isclose(stored, config.manual_cutoff_mev, rtol=0.0, atol=1e-8):
                        blocking.append(
                            f"artifact was trained at {stored:g} meV but manual_cutoff_mev is "
                            f"{config.manual_cutoff_mev:g} meV; cutoff-specific weights are "
                            "never substituted"
                        )
                    stored_system = manifest.get("system_type")
                    if stored_system is not None and stored_system != config.system_type:
                        blocking.append(
                            f"artifact system {stored_system!r} does not match project "
                            f"system {config.system_type!r}"
                        )

        return ProjectPlan(
            name=config.name,
            system_type=config.system_type,
            view=config.view,
            stages=tuple(stages),
            dataset_source=dataset_source,
            dataset_detail=dataset_detail,
            generation_chains=generation_chains,
            estimated_generation_seconds=estimated_seconds,
            seconds_per_chain=rate,
            outputs=tuple(outputs),
            notes=tuple(notes),
            blocking_issues=tuple(blocking),
        )

    def inspect_experiment(self) -> dict[str, Any]:
        if self.config.experiment_csv is None:
            raise RuntimeError("experiment.csv is required for experiment inspection")
        measurement, source = self._load_experiment()
        bias, spectra = measurement.site_spectra()
        inspection = {
            "source": source,
            "n_sites": int(spectra.shape[0]),
            "n_bias_points": int(bias.size),
            "bias_min_mev": float(bias.min()),
            "bias_max_mev": float(bias.max()),
            "median_bias_step_mev": float(np.median(np.diff(bias))),
            "finite": bool(np.all(np.isfinite(spectra))),
            "configured_cutoffs_mev": list(self.config.cutoffs_mev),
            "manual_cutoff_mev": self.config.manual_cutoff_mev,
        }
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._record_resolved_config()
        _write_json(self.config.output_dir / "experiment_inspection.json", inspection)
        return inspection

    def _record_resolved_config(self) -> Path:
        """Persist the resolved contract, refusing mixed output directories."""
        destination = self.config.output_dir / "resolved_project_config.json"
        resolved = {"toolkit": brand_manifest(), **_json_safe(asdict(self.config))}
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != resolved:
                raise FileExistsError(
                    f"{destination} belongs to a different resolved configuration; "
                    "choose a new output_dir"
                )
            return destination
        destination.write_text(
            json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8"
        )
        return destination

    def calibrate_preprocessing(
        self,
        *,
        candidates: Mapping[str, AugmentationConfig] | None = None,
    ) -> dict[float, AugmentationCalibrationResult]:
        if self.config.experiment_csv is None:
            raise RuntimeError("experiment.csv is required for calibration")
        dataset = self.load_dataset()
        measurement, _ = self._load_experiment()
        bias, spectra = measurement.site_spectra()
        evaluated_cutoffs = (
            (self.config.manual_cutoff_mev,)
            if self.config.manual_cutoff_mev is not None
            else self.config.cutoffs_mev
        )
        available_max = min(float(dataset.bias_mev[-1]), float(bias[-1]))
        invalid = [item for item in evaluated_cutoffs if item > available_max + 1e-8]
        if invalid:
            raise ValueError(
                f"cutoffs {invalid} exceed shared simulation/experiment coverage "
                f"through {available_max:g} meV"
            )

        calibration_dir = self.config.output_dir / "calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        for cutoff in evaluated_cutoffs:
            result = calibrate_augmentation(
                dataset,
                spectra,
                bias,
                cutoffs_mev=(cutoff,),
                candidates=candidates,
                output_points=self.config.output_points,
            )
            self.calibrations[cutoff] = result
            result.save(calibration_dir / f"cutoff-{_cutoff_tag(cutoff)}.json")

        passing = [cutoff for cutoff, result in self.calibrations.items() if result.acceptance_passed]
        if not passing:
            prefix = (
                "the manually selected cutoff did not pass experimental compatibility"
                if self.config.manual_cutoff_mev is not None
                else "no configured cutoff passed experimental compatibility"
            )
            raise RuntimeError(
                f"{prefix}; inspect {calibration_dir} before training"
            )
        if self.config.manual_cutoff_mev is not None:
            self.selected_cutoff_mev = float(self.config.manual_cutoff_mev)
        else:
            selector = max if self.config.cutoff_policy == "largest_passing" else min
            self.selected_cutoff_mev = float(selector(passing))
        self.selected_augmentation = self.calibrations[
            self.selected_cutoff_mev
        ].selected_config
        summary = {
            "policy": self.config.cutoff_policy,
            "manual_cutoff_mev": self.config.manual_cutoff_mev,
            "selected_cutoff_mev": self.selected_cutoff_mev,
            "passing_cutoffs_mev": passing,
            "failed_cutoffs_mev": [
                cutoff for cutoff in evaluated_cutoffs if cutoff not in passing
            ],
            "evaluated_cutoffs_mev": list(evaluated_cutoffs),
            "selected_augmentation": asdict(self.selected_augmentation),
        }
        _write_json(calibration_dir / "summary.json", summary)
        return dict(self.calibrations)

    def prepare_training_data(self):
        if self.selected_cutoff_mev is None or self.selected_augmentation is None:
            raise RuntimeError("calibrate_preprocessing must run before preparation")
        config = self.selected_augmentation
        augmented = augment_experimental_like(
            self.load_dataset(),
            seed=config.seed,
            noise=config.noise,
            broadening_points=config.broadening_points,
            energy_shift_mev=config.energy_shift_mev,
            energy_stretch=config.energy_stretch,
            background_quadratic=config.background_quadratic,
            amplitude_range=config.amplitude_range,
        )
        self.prepared = prepare_training_dataset(
            augmented,
            TrainingPreprocessingConfig(
                bias_cutoff_mev=self.selected_cutoff_mev,
                output_points=self.config.output_points,
            ),
        )
        return self.prepared

    def train(self) -> TrainingRun:
        if self.prepared is None:
            self.prepare_training_data()
        artifact = self.config.output_dir / "artifact"
        if artifact.exists() and any(artifact.iterdir()):
            raise FileExistsError(
                f"artifact directory is not empty: {artifact}; choose a new output_dir"
            )
        self.training_run = train_supervised(
            self.prepared,
            view=self.config.view,
            model=self.config.model,
            preset=self.config.preset,
            model_options=self.config.model_options,
            verbose=self.config.verbose,
        )
        self.training_run.save(artifact)
        return self.training_run

    def infer(self) -> ExperimentalChainResult | ExperimentalGlobalResult:
        if self.config.experiment_csv is None:
            raise RuntimeError("experiment.csv is required for inference")
        loaded_external_artifact = self.training_run is None and self.config.artifact_path is not None
        analysis_dir = self.config.output_dir / "analysis"
        expected_outputs = [
            analysis_dir / "couplings.csv",
            analysis_dir / "report.json",
            analysis_dir / "report.html",
            analysis_dir / "summary.png",
        ]
        if loaded_external_artifact:
            expected_outputs.extend(
                [
                    self.config.output_dir / "preflight" / "workflow_decision.json",
                    self.config.output_dir / "preflight" / "workflow_decision.html",
                ]
            )
        existing = [path for path in expected_outputs if path.exists()]
        if existing:
            raise FileExistsError(
                f"inference outputs already exist: {[str(path) for path in existing]}; "
                "choose a new output_dir to preserve the previous analysis"
            )
        if self.training_run is None:
            artifact = self.config.artifact_path or self.config.output_dir / "artifact"
            self.training_run = TrainingRun.load(artifact)
        if self.training_run.system_type != self.config.system_type:
            raise ValueError("artifact system does not match project system")
        configured_cutoff = self.config.manual_cutoff_mev
        assert configured_cutoff is not None  # enforced for experiment projects
        artifact_cutoff = self.training_run.preprocessing_config.bias_cutoff_mev
        if configured_cutoff is not None and not np.isclose(
            artifact_cutoff, configured_cutoff, rtol=0.0, atol=1e-8
        ):
            raise ValueError(
                f"artifact was trained at {artifact_cutoff:g} meV but the manually "
                f"selected cutoff is {configured_cutoff:g} meV; cutoff-specific weights "
                "cannot be substituted"
            )
        if loaded_external_artifact:
            from .workflow import advise_experiment

            dataset_paths = (
                [self.config.dataset_path] if self.config.dataset_path is not None else []
            )
            self.workflow_decision = advise_experiment(
                self.config.experiment_csv,
                manual_cutoff_mev=float(configured_cutoff),
                artifact_roots=[self.config.artifact_path],
                dataset_paths=dataset_paths,
                system_type=self.config.system_type,
                view=self.config.view,
                allow_development_artifacts=self.config.allow_development_artifacts,
                max_validation_mae_mev=self.config.max_validation_mae_mev,
                max_test_mae_mev=self.config.max_test_mae_mev,
            )
            decision_dir = self.config.output_dir / "preflight"
            self.workflow_decision.save_json(decision_dir / "workflow_decision.json")
            self.workflow_decision.save_html(decision_dir / "workflow_decision.html")
            if not self.workflow_decision.can_use_existing_model:
                raise RuntimeError(
                    f"existing artifact preflight decision: {self.workflow_decision.action}; "
                    f"{self.workflow_decision.summary}; inspect {decision_dir}"
                )
        self.selected_cutoff_mev = self.training_run.preprocessing_config.bias_cutoff_mev
        measurement, source = self._load_experiment()
        self.experimental_result = self.training_run.create_analyzer().analyze_measurement(
            measurement, source=source
        )
        analysis_dir.mkdir(parents=True, exist_ok=True)
        self.experimental_result.save_couplings_csv(analysis_dir / "couplings.csv")
        self.experimental_result.save_report_json(analysis_dir / "report.json")
        figure = self.experimental_result.plot_summary()
        figure.savefig(analysis_dir / "summary.png", dpi=160)
        try:
            import matplotlib.pyplot as plt

            plt.close(figure)
        except ImportError:  # pragma: no cover
            pass
        self.experimental_result.save_html_report(
            analysis_dir / "report.html",
            title=self.config.name,
            artifact_manifest=(
                self.config.artifact_path or self.config.output_dir / "artifact"
            ) / "manifest.json",
        )
        return self.experimental_result

    def run(self) -> ProjectOutcome:
        self.inspect_experiment()
        if self.config.artifact_path is None:
            self.calibrate_preprocessing()
            self.prepare_training_data()
            self.train()
            artifact = self.config.output_dir / "artifact"
        else:
            artifact = self.config.artifact_path
        result = self.infer()
        outcome = ProjectOutcome(
            selected_cutoff_mev=float(self.selected_cutoff_mev),
            artifact_path=artifact,
            analysis_dir=self.config.output_dir / "analysis",
            report_path=self.config.output_dir / "analysis" / "report.html",
            status=result.diagnostics.status,
        )
        _write_json(self.config.output_dir / "project_summary.json", asdict(outcome))
        return outcome


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _parse_generation_config(
    values: Mapping[str, Any], base: Path, output_dir: Path
) -> DatasetGenerationConfig:
    system_type = str(values.get("system", "inhomogeneous_heisenberg"))
    output_value = values.get("output", output_dir / "data" / "generated_dataset.npz")
    output_path = _resolve_path(base, output_value)
    coupling_range = values.get("coupling_range_mev")
    coupling_ranges = values.get("coupling_ranges_mev") or []
    return DatasetGenerationConfig(
        system_type=system_type,
        output_path=output_path,
        n_sites=int(values.get("n_sites", 12)),
        n_samples=int(values.get("n_samples", 1000)),
        coupling_range_mev=(
            tuple(float(item) for item in coupling_range)
            if coupling_range is not None
            else None
        ),
        coupling_ranges_mev=tuple(
            tuple(float(item) for item in interval) for interval in coupling_ranges
        ),
        bias_range_mev=tuple(
            float(item) for item in values.get("bias_range_mev", [0.0, 100.0])
        ),
        bias_points=int(values.get("bias_points", 200)),
        broadening_mev=float(values.get("broadening_mev", 0.5)),
        observable=str(values.get("observable", "Sz")),
        observable_weights=(
            tuple(float(item) for item in values["observable_weights"])
            if values.get("observable_weights") is not None
            else None
        ),
        output_quantity=str(values.get("output_quantity", "didv")),
        backend=str(values.get("backend", "dmrgpy")),
        max_bond_dimension=int(values.get("max_bond_dimension", 20)),
        kpm_max_bond_dimension=int(values.get("kpm_max_bond_dimension", 20)),
        seed=int(values.get("seed", 42)),
        checkpoint_every=int(values.get("checkpoint_every", 25)),
    )


def _console_progress(done: int, total: int) -> None:
    end = "\n" if done == total else "\r"
    print(f"generating simulations: {done}/{total}", end=end, flush=True)


def _cutoff_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    branded = {"toolkit": brand_manifest(), **dict(payload)}
    path.write_text(
        json.dumps(_json_safe(branded), indent=2, sort_keys=True), encoding="utf-8"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
