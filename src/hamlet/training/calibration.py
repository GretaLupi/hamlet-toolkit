"""Label-free calibration of simulation nuisance augmentation to experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from ..branding import brand_manifest
from ..data import SpectroscopyDataset, as_supervised
from ..preprocessing import make_local_windows
from .augmentation import augment_experimental_like
from .distribution import TrainingDistributionProfile
from .preprocessing import TrainingPreprocessingConfig, prepare_training_dataset
from .splitting import grouped_split


@dataclass(frozen=True)
class AugmentationConfig:
    broadening_points: tuple[float, float] = (0.5, 2.0)
    energy_shift_mev: tuple[float, float] = (0.0, 0.0)
    energy_stretch: tuple[float, float] = (1.0, 1.0)
    background_quadratic: tuple[float, float] = (0.0, 0.0)
    amplitude_range: tuple[float, float] = (1.0, 1.0)
    noise: float = 0.002
    seed: int = 42


DEFAULT_AUGMENTATION_CANDIDATES: dict[str, AugmentationConfig] = {
    "reference": AugmentationConfig(),
    "smoother": AugmentationConfig(broadening_points=(1.0, 4.0)),
    "smooth_calibrated": AugmentationConfig(
        broadening_points=(1.0, 5.0),
        energy_shift_mev=(-2.0, 2.0),
        energy_stretch=(0.97, 1.03),
        background_quadratic=(-0.02, 0.02),
        amplitude_range=(0.85, 1.15),
    ),
    "moderate_full": AugmentationConfig(
        broadening_points=(1.0, 7.0),
        energy_shift_mev=(-4.0, 4.0),
        energy_stretch=(0.94, 1.06),
        background_quadratic=(-0.04, 0.04),
        amplitude_range=(0.7, 1.3),
        noise=0.003,
    ),
}


@dataclass(frozen=True)
class AugmentationCalibrationResult:
    selected_name: str
    selected_config: AugmentationConfig
    candidate_metrics: dict[str, dict[str, object]]
    cutoffs_mev: tuple[float, ...]
    acceptance_passed: bool

    def save(self, path: str | Path) -> None:
        payload = {
            "toolkit": brand_manifest(),
            "selected_name": self.selected_name,
            "selected_config": asdict(self.selected_config),
            "candidate_metrics": self.candidate_metrics,
            "cutoffs_mev": list(self.cutoffs_mev),
            "acceptance_passed": self.acceptance_passed,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def calibrate_augmentation(
    dataset: SpectroscopyDataset,
    experimental_spectra: ArrayLike,
    experimental_bias_mev: ArrayLike,
    *,
    cutoffs_mev: Sequence[float] = (50.0, 70.0),
    candidates: Mapping[str, AugmentationConfig] | None = None,
    output_points: int = 200,
    split_seed: int = 42,
) -> AugmentationCalibrationResult:
    """Compare nuisance settings without using model predictions or labels.

    Candidate selection minimizes the worst normalized experimental OOD score
    across cutoffs, then the fraction above the training 95% threshold.
    """
    spectra = np.asarray(experimental_spectra, dtype=np.float32)
    bias = np.asarray(experimental_bias_mev, dtype=float)
    if spectra.ndim != 2 or spectra.shape[1] != bias.size:
        raise ValueError("experimental_spectra must have shape (sites, len(bias))")
    settings = dict(candidates or DEFAULT_AUGMENTATION_CANDIDATES)
    if not settings:
        raise ValueError("at least one augmentation candidate is required")
    cutoffs = tuple(float(value) for value in cutoffs_mev)
    metrics: dict[str, dict[str, object]] = {}

    for name, config in settings.items():
        enhanced = augment_experimental_like(
            dataset,
            seed=config.seed,
            noise=config.noise,
            broadening_points=config.broadening_points,
            energy_shift_mev=config.energy_shift_mev,
            energy_stretch=config.energy_stretch,
            background_quadratic=config.background_quadratic,
            amplitude_range=config.amplitude_range,
        )
        cutoff_metrics: dict[str, object] = {}
        worst_normalized_max = 0.0
        worst_fraction_95 = 0.0
        all_below_99 = True
        for cutoff in cutoffs:
            prepared = prepare_training_dataset(
                enhanced,
                TrainingPreprocessingConfig(
                    bias_cutoff_mev=cutoff, output_points=output_points
                ),
            )
            supervised = as_supervised(prepared.dataset, "local_bonds")
            split = grouped_split(supervised, seed=split_seed)
            profile = TrainingDistributionProfile.fit(
                split.train.inputs, split.train.targets, max_references=64, seed=split_seed
            )
            processed_experiment = prepared.preprocessor.transform_map(spectra, bias)
            windows = make_local_windows(processed_experiment).reshape(
                spectra.shape[0] - 2, -1
            )
            scores = profile.score(windows)
            threshold_95 = profile.threshold_95
            threshold_99 = float(profile.score_quantiles[-1])
            fraction_95 = float(np.mean(scores > threshold_95))
            normalized_max = float(np.max(scores) / threshold_99)
            worst_fraction_95 = max(worst_fraction_95, fraction_95)
            worst_normalized_max = max(worst_normalized_max, normalized_max)
            all_below_99 = all_below_99 and bool(np.all(scores <= threshold_99))
            cutoff_metrics[str(cutoff)] = {
                "experimental_scores": scores.astype(float).tolist(),
                "training_threshold_95": threshold_95,
                "training_threshold_99": threshold_99,
                "fraction_above_95": fraction_95,
                "all_below_99": bool(np.all(scores <= threshold_99)),
            }
        objective = worst_normalized_max + worst_fraction_95
        metrics[name] = {
            "config": asdict(config),
            "by_cutoff": cutoff_metrics,
            "worst_normalized_max": worst_normalized_max,
            "worst_fraction_above_95": worst_fraction_95,
            "all_below_99": all_below_99,
            "objective": objective,
        }

    selected = min(
        settings,
        key=lambda name: (
            float(metrics[name]["objective"]),
            list(settings).index(name),
        ),
    )
    selected_metrics = metrics[selected]
    acceptance = bool(selected_metrics["all_below_99"]) and float(
        selected_metrics["worst_fraction_above_95"]
    ) <= 0.25
    return AugmentationCalibrationResult(
        selected,
        settings[selected],
        metrics,
        cutoffs,
        acceptance,
    )
