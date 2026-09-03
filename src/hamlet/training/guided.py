"""Guided, reproducible supervised training and portable model artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..benchmarks import correlation_fidelity, mae, rmse
from ..branding import brand_manifest
from ..data import as_supervised
from ..models import create_supervised_model
from .preprocessing import PreparedTrainingDataset, TrainingPreprocessingConfig
from .scaling import MinMaxTargetScaler
from .splitting import grouped_split
from .distribution import TrainingDistributionProfile
from .ensemble import EnsembleAggregation, select_ensemble_aggregation


ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrainingPreset:
    """Compute budget and leakage-safe split policy for one training run."""

    name: str
    seeds: tuple[int, ...]
    epochs: int
    batch_size: int
    patience: int
    validation_fraction: float = 0.2
    test_fraction: float = 0.1
    split_seed: int = 42


TRAINING_PRESETS: dict[str, TrainingPreset] = {
    "quick": TrainingPreset("quick", (42,), 20, 32, 4),
    "standard": TrainingPreset("standard", (42, 43, 44), 100, 32, 12),
    "research": TrainingPreset("research", (41, 42, 43, 44, 45), 250, 32, 25),
}


def get_training_preset(preset: str | TrainingPreset) -> TrainingPreset:
    if isinstance(preset, TrainingPreset):
        return preset
    try:
        return TRAINING_PRESETS[preset]
    except KeyError as exc:
        choices = ", ".join(TRAINING_PRESETS)
        raise ValueError(f"unknown training preset {preset!r}; choose {choices}") from exc


@dataclass
class TrainingRun:
    """A trained ensemble plus everything required to reproduce inference."""

    models: tuple[Any, ...]
    target_scaler: MinMaxTargetScaler
    preprocessing_config: TrainingPreprocessingConfig
    model_name: str
    view: str
    preset: TrainingPreset
    target_names: tuple[str, ...]
    coupling_unit: str
    metrics: dict[str, Any]
    model_options: dict[str, Any]
    dataset_metadata: dict[str, Any]
    system_type: str
    n_sites: int
    distribution_profile: TrainingDistributionProfile | None = None
    aggregation: EnsembleAggregation = EnsembleAggregation()
    histories: tuple[dict[str, list[float]], ...] = ()

    def predict(self, inputs: ArrayLike) -> NDArray[np.float32]:
        """Return ensemble-mean predictions in physical coupling units."""
        features = np.asarray(inputs, dtype=np.float32)
        physical = np.stack(
            [self.target_scaler.inverse_transform(_predict(model, features)) for model in self.models]
        )
        return self.aggregation.aggregate(physical)

    def create_analyzer(self):
        """Create the local or global analyzer stored by the training contract."""
        model_names = [
            f"{self.model_name}_seed_{seed}" for seed in self.preset.seeds
        ]
        if self.view == "local_bonds":
            from ..experimental import ExperimentalChainAnalyzer

            return ExperimentalChainAnalyzer(
                self.models,
                self.preprocessing_config.build_preprocessor(),
                target_scaler=self.target_scaler,
                coupling_unit=self.coupling_unit,
                model_names=model_names,
                flatten_inputs=True,
                aggregation=self.aggregation,
            )
        if self.view == "global":
            from ..experimental import ExperimentalGlobalAnalyzer

            return ExperimentalGlobalAnalyzer(
                self.models,
                self.preprocessing_config.build_preprocessor(),
                n_sites=self.n_sites,
                parameter_names=self.target_names,
                target_scaler=self.target_scaler,
                coupling_unit=self.coupling_unit,
                model_names=model_names,
                aggregation=self.aggregation,
            )
        raise ValueError(f"unsupported trained view: {self.view!r}")

    def save(self, path: str | Path) -> Path:
        """Save models and a human-readable inference/training manifest."""
        destination = Path(path)
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"artifact directory is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)

        model_records = []
        for index, model in enumerate(self.models):
            if self.model_name.startswith("keras_"):
                filename = f"model_seed_{self.preset.seeds[index]}.keras"
                model.save(destination / filename)
                model_format = "keras"
            else:
                try:
                    import joblib
                except ImportError as exc:  # pragma: no cover
                    raise ImportError("saving sklearn artifacts requires scikit-learn") from exc
                filename = f"model_seed_{self.preset.seeds[index]}.joblib"
                # Compressed because tree ensembles dominate artifact size and
                # pickle them very redundantly: a 600-tree forest trained on
                # 3000 chains measured 55 MB per seed uncompressed and 21 MB at
                # this level, for the same predictions. joblib.load detects
                # compression from the file, so artifacts written either way
                # keep loading.
                joblib.dump(model, destination / filename, compress=3)
                model_format = "joblib"
            model_records.append(
                {"seed": self.preset.seeds[index], "file": filename, "format": model_format}
            )

        manifest = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "toolkit": brand_manifest(),
            "package_version": "0.1.0",
            "model_name": self.model_name,
            "view": self.view,
            "system_type": self.system_type,
            "n_sites": self.n_sites,
            "target_names": list(self.target_names),
            "coupling_unit": self.coupling_unit,
            "energy_convention": {"dmrgpy_energy_unit_mev": 10.0},
            "preprocessing": self.preprocessing_config.to_metadata(),
            "target_scaler": self.target_scaler.to_metadata(),
            "training_preset": asdict(self.preset),
            "model_options": _jsonable(self.model_options),
            "metrics": _jsonable(self.metrics),
            "dataset_metadata": _jsonable(self.dataset_metadata),
            "models": model_records,
            "ensemble_aggregation": _jsonable(self.aggregation),
        }
        if self.distribution_profile is not None:
            self.distribution_profile.save(destination / "training_distribution.npz")
            manifest["training_distribution"] = {
                "file": "training_distribution.npz",
                **self.distribution_profile.to_metadata(),
            }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "TrainingRun":
        source = Path(path)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported model artifact schema version")
        models = []
        for record in manifest["models"]:
            if record["format"] == "keras":
                try:
                    import keras
                except ImportError as exc:  # pragma: no cover
                    raise ImportError("loading this artifact requires the ml dependency") from exc
                models.append(keras.saving.load_model(source / record["file"]))
            elif record["format"] == "joblib":
                import joblib

                models.append(joblib.load(source / record["file"]))
            else:
                raise ValueError(f"unknown stored model format: {record['format']}")

        prep_values = dict(manifest["preprocessing"])
        prep_values.pop("resolved_scale_range_mev", None)
        for key in ("baseline_range_mev", "scale_range_mev", "clip"):
            if prep_values.get(key) is not None:
                prep_values[key] = tuple(prep_values[key])
        scaler_values = manifest["target_scaler"]
        preset_values = manifest["training_preset"]
        preset_values["seeds"] = tuple(preset_values["seeds"])
        distribution_record = manifest.get("training_distribution")
        distribution_profile = (
            TrainingDistributionProfile.load(source / distribution_record["file"])
            if distribution_record
            else None
        )
        aggregation_record = manifest.get("ensemble_aggregation", {"method": "mean"})
        if aggregation_record.get("weights") is not None:
            aggregation_record["weights"] = tuple(aggregation_record["weights"])
        return cls(
            models=tuple(models),
            target_scaler=MinMaxTargetScaler(
                np.asarray(scaler_values["minimum"], dtype=np.float32),
                np.asarray(scaler_values["maximum"], dtype=np.float32),
            ),
            preprocessing_config=TrainingPreprocessingConfig(**prep_values),
            model_name=manifest["model_name"],
            view=manifest["view"],
            preset=TrainingPreset(**preset_values),
            target_names=tuple(manifest["target_names"]),
            coupling_unit=manifest["coupling_unit"],
            metrics=manifest["metrics"],
            model_options=manifest["model_options"],
            dataset_metadata=manifest["dataset_metadata"],
            system_type=manifest["system_type"],
            n_sites=int(manifest["n_sites"]),
            distribution_profile=distribution_profile,
            aggregation=EnsembleAggregation(**aggregation_record),
        )


def train_supervised(
    prepared: PreparedTrainingDataset,
    *,
    view: str,
    model: str = "keras_mlp",
    preset: str | TrainingPreset = "standard",
    model_options: Mapping[str, Any] | None = None,
    coupling_unit: str = "meV",
    verbose: int = 0,
) -> TrainingRun:
    """Train and evaluate a supervised ensemble with safe defaults.

    The target scaler is fit on training chains only. Validation and test
    metrics are always reported after conversion back to physical units.
    """
    policy = get_training_preset(preset)
    supervised = as_supervised(prepared.dataset, view)  # type: ignore[arg-type]
    split = grouped_split(
        supervised,
        validation_fraction=policy.validation_fraction,
        test_fraction=policy.test_fraction,
        seed=policy.split_seed,
    )
    scaler = MinMaxTargetScaler.fit(split.train.targets)
    distribution_profile = TrainingDistributionProfile.fit(
        split.train.inputs,
        split.train.targets,
        seed=policy.split_seed,
    )
    options = dict(model_options or {})
    models: list[Any] = []
    histories: list[dict[str, list[float]]] = []

    for seed in policy.seeds:
        current_options = dict(options)
        if model == "random_forest":
            current_options.setdefault("random_state", seed)
        if model == "keras_cnn":
            window_sites = 3 if view == "local_bonds" else prepared.dataset.n_sites
            current_options.setdefault(
                "spectrum_shape", (window_sites, prepared.config.output_points)
            )
        if model.startswith("keras_"):
            try:
                import keras
            except ImportError as exc:  # pragma: no cover
                raise ImportError("neural training requires the ml optional dependency") from exc
            keras.utils.set_random_seed(seed)

        estimator = create_supervised_model(
            model,
            input_dim=split.train.inputs.shape[1],
            output_dim=split.train.targets.shape[1],
            **current_options,
        )
        train_y = scaler.transform(split.train.targets)
        validation_y = scaler.transform(split.validation.targets)
        if model.startswith("keras_"):
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=policy.patience, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", patience=max(2, policy.patience // 2), factor=0.5
                ),
            ]
            history = estimator.fit(
                split.train.inputs,
                train_y,
                validation_data=(split.validation.inputs, validation_y),
                epochs=policy.epochs,
                batch_size=policy.batch_size,
                callbacks=callbacks,
                verbose=verbose,
            )
            histories.append(
                {key: [float(value) for value in values] for key, values in history.history.items()}
            )
        else:
            estimator.fit(split.train.inputs, train_y)
            histories.append({})
        models.append(estimator)

    validation_predictions = _physical_predictions(models, scaler, split.validation.inputs)
    aggregation, aggregation_candidates = select_ensemble_aggregation(
        validation_predictions, split.validation.targets
    )
    metrics = {
        "validation": _evaluate(
            models, scaler, split.validation.inputs, split.validation.targets, aggregation
        ),
        "test": _evaluate(models, scaler, split.test.inputs, split.test.targets, aggregation),
        "aggregation_selection": {
            "selected": aggregation.method,
            "selection_split": "validation",
            "selection_metric": "mae",
            "candidates": aggregation_candidates,
        },
        "split": {
            "train_groups": int(np.unique(split.train.group_ids).size),
            "validation_groups": int(np.unique(split.validation.group_ids).size),
            "test_groups": int(np.unique(split.test.group_ids).size),
            "split_seed": policy.split_seed,
        },
    }
    return TrainingRun(
        models=tuple(models),
        target_scaler=scaler,
        preprocessing_config=prepared.config,
        model_name=model,
        view=view,
        preset=policy,
        target_names=supervised.target_names,
        coupling_unit=coupling_unit,
        metrics=metrics,
        model_options=options,
        dataset_metadata=prepared.dataset.metadata,
        system_type=prepared.dataset.system_type,
        n_sites=prepared.dataset.n_sites,
        distribution_profile=distribution_profile,
        aggregation=aggregation,
        histories=tuple(histories),
    )


def _predict(model: Any, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
    try:
        values = model.predict(inputs, verbose=0)
    except TypeError:
        values = model.predict(inputs)
    return np.asarray(values, dtype=np.float32)


def _evaluate(
    models: list[Any],
    scaler: MinMaxTargetScaler,
    inputs: NDArray[np.float32],
    expected: NDArray[np.float32],
    aggregation: EnsembleAggregation | None = None,
) -> dict[str, Any]:
    per_model = _physical_predictions(models, scaler, inputs)
    rule = aggregation or EnsembleAggregation()
    ensemble = rule.aggregate(per_model)

    def scores(predicted: NDArray[np.float32]) -> dict[str, float]:
        return {
            "mae": mae(predicted, expected),
            "rmse": rmse(predicted, expected),
            "correlation_fidelity": correlation_fidelity(predicted, expected),
        }

    return {
        "unit": "meV",
        "ensemble": scores(ensemble),
        "ensemble_aggregation": rule.method,
        "per_model": [scores(item) for item in per_model],
        "n_examples": int(expected.shape[0]),
    }


def _physical_predictions(
    models: Sequence[Any],
    scaler: MinMaxTargetScaler,
    inputs: NDArray[np.float32],
) -> NDArray[np.float32]:
    return np.stack([scaler.inverse_transform(_predict(item, inputs)) for item in models])


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
