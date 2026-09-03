"""Controlled architecture comparison and bias-cutoff artifact banks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from ..branding import brand_manifest
from .guided import TrainingPreset, TrainingRun, train_supervised
from .preprocessing import TrainingPreprocessingConfig, prepare_training_dataset
from ..data import SpectroscopyDataset
from .augmentation import augment_experimental_like
from .calibration import AugmentationConfig


@dataclass(frozen=True)
class ArtifactRecommendation:
    artifact_path: Path
    cutoff_mev: float
    available_bias_min_mev: float
    available_bias_max_mev: float
    reason: str


def train_cutoff_bank(
    dataset: SpectroscopyDataset,
    output_dir: str | Path,
    *,
    cutoffs_mev: Sequence[float] = (30.0, 40.0, 50.0, 70.0, 100.0),
    comparison_cutoff_mev: float = 50.0,
    candidates: Sequence[str] = ("keras_mlp", "keras_cnn"),
    preset: str | TrainingPreset = "research",
    output_points: int = 200,
    model_options: Mapping[str, Mapping[str, Any]] | None = None,
    experimental_like: bool = True,
    augmentation_seed: int = 42,
    augmentation_noise: float = 0.002,
    augmentation_config: AugmentationConfig | None = None,
    verbose: int = 0,
) -> dict[str, Any]:
    """Select an architecture at one cutoff, then train a controlled bank.

    Every call uses the same grouped split seed through the selected training
    preset. The candidate with the lowest ensemble validation MAE is selected;
    its already-trained comparison artifact is reused in the cutoff bank.
    """
    cutoffs = tuple(float(value) for value in cutoffs_mev)
    if not cutoffs or any(value <= 0 for value in cutoffs):
        raise ValueError("cutoffs_mev must contain positive values")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("cutoffs_mev cannot contain duplicates")
    if comparison_cutoff_mev not in cutoffs:
        raise ValueError("comparison_cutoff_mev must be present in cutoffs_mev")
    if not candidates:
        raise ValueError("at least one candidate model is required")
    if max(cutoffs) > float(np.max(dataset.bias_mev)) + 1e-8:
        raise ValueError("the raw dataset does not cover every requested cutoff")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    resolved_augmentation = augmentation_config or AugmentationConfig(
        seed=augmentation_seed, noise=augmentation_noise
    )
    training_dataset = (
        augment_experimental_like(
            dataset,
            seed=resolved_augmentation.seed,
            noise=resolved_augmentation.noise,
            broadening_points=resolved_augmentation.broadening_points,
            energy_shift_mev=resolved_augmentation.energy_shift_mev,
            energy_stretch=resolved_augmentation.energy_stretch,
            background_quadratic=resolved_augmentation.background_quadratic,
            amplitude_range=resolved_augmentation.amplitude_range,
        )
        if experimental_like
        else dataset
    )
    options_by_model = dict(model_options or {})
    comparison_prepared = prepare_training_dataset(
        training_dataset,
        TrainingPreprocessingConfig(
            bias_cutoff_mev=float(comparison_cutoff_mev), output_points=output_points
        ),
    )
    comparison: dict[str, dict[str, Any]] = {}
    candidate_runs: dict[str, TrainingRun] = {}
    candidate_paths: dict[str, Path] = {}
    for candidate in candidates:
        run = train_supervised(
            comparison_prepared,
            view="local_bonds",
            model=candidate,
            preset=preset,
            model_options=options_by_model.get(candidate),
            verbose=verbose,
        )
        artifact = root / "architecture_comparison" / f"{candidate}-cut{_cutoff_tag(comparison_cutoff_mev)}"
        run.save(artifact)
        candidate_runs[candidate] = run
        candidate_paths[candidate] = artifact
        comparison[candidate] = {
            "artifact": str(artifact.relative_to(root)),
            "validation": run.metrics["validation"]["ensemble"],
            "test": run.metrics["test"]["ensemble"],
        }

    winner = min(
        candidates,
        key=lambda name: comparison[name]["validation"]["mae"],
    )
    bank: dict[str, str] = {}
    for cutoff in cutoffs:
        if cutoff == comparison_cutoff_mev:
            artifact = candidate_paths[winner]
        else:
            prepared = prepare_training_dataset(
                training_dataset,
                TrainingPreprocessingConfig(
                    bias_cutoff_mev=cutoff, output_points=output_points
                ),
            )
            run = train_supervised(
                prepared,
                view="local_bonds",
                model=winner,
                preset=preset,
                model_options=options_by_model.get(winner),
                verbose=verbose,
            )
            artifact = root / "cutoff_bank" / f"{winner}-cut{_cutoff_tag(cutoff)}"
            run.save(artifact)
        bank[str(cutoff)] = str(artifact.relative_to(root))

    catalog = {
        "toolkit": brand_manifest(),
        "catalog_schema_version": 1,
        "selection_rule": "lowest ensemble validation MAE in meV",
        "comparison_cutoff_mev": float(comparison_cutoff_mev),
        "selected_model": winner,
        "candidates": comparison,
        "cutoff_artifacts": bank,
        "cutoffs_mev": list(cutoffs),
        "preset": preset.name if isinstance(preset, TrainingPreset) else preset,
        "experimental_like": experimental_like,
        "augmentation_config": (
            asdict(resolved_augmentation) if experimental_like else None
        ),
    }
    (root / "catalog.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True), encoding="utf-8"
    )
    return catalog


def recommend_artifact(
    artifact_root: str | Path,
    experiment_bias_mev: ArrayLike,
    *,
    system_type: str = "inhomogeneous_heisenberg",
    view: str = "local_bonds",
    strategy: str = "best_validation_mae",
    required_cutoff_mev: float | None = None,
) -> ArtifactRecommendation:
    """Choose a validated, physically compatible artifact from measured coverage.

    ``best_validation_mae`` compares only artifacts fully covered by the
    experiment. ``largest_covered`` ignores benchmark performance and uses the
    widest compatible window. When ``required_cutoff_mev`` is provided, only
    independently trained weights at exactly that manual cutoff are eligible.
    """
    if strategy not in {"best_validation_mae", "largest_covered"}:
        raise ValueError("strategy must be best_validation_mae or largest_covered")
    bias = np.asarray(experiment_bias_mev, dtype=float)
    if bias.ndim != 1 or bias.size < 2 or not np.all(np.isfinite(bias)):
        raise ValueError("experiment_bias_mev must be a finite one-dimensional grid")
    available_min = float(np.min(bias))
    available_max = float(np.max(bias))
    eligible: list[tuple[float, float, Path]] = []
    discovered: list[float] = []
    root = Path(artifact_root)
    catalog_path = root / "catalog.json"
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        artifact_dirs = []
        for value in catalog.get("cutoff_artifacts", {}).values():
            stored = Path(value)
            artifact_dirs.append(stored if stored.is_absolute() else root / stored)
        manifest_paths = [path / "manifest.json" for path in artifact_dirs]
    else:
        manifest_paths = list(root.rglob("manifest.json"))
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("system_type") != system_type or manifest.get("view") != view:
            continue
        preprocessing = manifest.get("preprocessing", {})
        cutoff = float(preprocessing["bias_cutoff_mev"])
        bias_min = float(preprocessing.get("bias_min_mev", 0.0))
        discovered.append(cutoff)
        if required_cutoff_mev is not None and not np.isclose(
            cutoff, required_cutoff_mev, rtol=0.0, atol=1e-8
        ):
            continue
        if available_min <= bias_min + 1e-8 and available_max >= cutoff - 1e-8:
            validation_mae = float(
                manifest.get("metrics", {})
                .get("validation", {})
                .get("ensemble", {})
                .get("mae", float("inf"))
            )
            eligible.append((cutoff, validation_mae, manifest_path.parent))
    if not discovered:
        raise FileNotFoundError("no compatible artifacts were found")
    if not eligible:
        if required_cutoff_mev is not None and any(
            np.isclose(item, required_cutoff_mev, rtol=0.0, atol=1e-8)
            for item in discovered
        ):
            raise ValueError(
                f"an artifact exists at {required_cutoff_mev:g} meV, but the experiment "
                f"does not cover its complete preprocessing window"
            )
        if required_cutoff_mev is not None:
            raise ValueError(
                f"no artifact was trained at the manually selected cutoff "
                f"{required_cutoff_mev:g} meV; available cutoffs: {sorted(set(discovered))}"
            )
        smallest = min(discovered)
        raise ValueError(
            f"experiment covers {available_min:g}–{available_max:g} meV, but the "
            f"smallest compatible artifact requires coverage through {smallest:g} meV"
        )
    if strategy == "best_validation_mae" and any(np.isfinite(item[1]) for item in eligible):
        cutoff, validation_mae, path = min(eligible, key=lambda item: (item[1], -item[0]))
        reason = (
            f"selected the lowest validation MAE ({validation_mae:.3g} meV) among "
            f"cutoffs fully covered by the experiment; selected cutoff: {cutoff:g} meV"
        )
    else:
        cutoff, _, path = max(eligible, key=lambda item: item[0])
        reason = (
            f"selected the largest trained cutoff ({cutoff:g} meV) fully covered "
            f"by the experiment ({available_min:g}–{available_max:g} meV)"
        )
    return ArtifactRecommendation(
        artifact_path=path,
        cutoff_mev=cutoff,
        available_bias_min_mev=available_min,
        available_bias_max_mev=available_max,
        reason=reason,
    )


def _cutoff_tag(cutoff: float) -> str:
    return f"{cutoff:g}".replace(".", "p")
