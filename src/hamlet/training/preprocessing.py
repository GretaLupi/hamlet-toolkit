"""Experiment-selected preprocessing configuration for supervised training."""

from dataclasses import asdict, dataclass, replace

from ..data import SpectroscopyDataset
from ..preprocessing import SpectralPreprocessor


@dataclass(frozen=True)
class TrainingPreprocessingConfig:
    """Define the spectral window used for both training and inference.

    ``bias_cutoff_mev`` is deliberately user-selected from the usable
    experimental range. If ``scale_range_mev`` is omitted, normalization uses
    the final ``scale_bandwidth_mev`` below that cutoff.
    """

    bias_cutoff_mev: float
    bias_min_mev: float = 0.0
    output_points: int = 200
    baseline_range_mev: tuple[float, float] = (0.0, 3.0)
    scale_range_mev: tuple[float, float] | None = None
    scale_bandwidth_mev: float = 10.0
    clip: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.bias_min_mev < self.bias_cutoff_mev:
            raise ValueError("bias_min_mev must be below bias_cutoff_mev")
        if self.output_points < 2:
            raise ValueError("output_points must be at least 2")
        if self.scale_bandwidth_mev <= 0:
            raise ValueError("scale_bandwidth_mev must be positive")
        baseline_start, baseline_stop = self.baseline_range_mev
        if baseline_start < self.bias_min_mev or baseline_stop > self.bias_cutoff_mev:
            raise ValueError("baseline_range_mev must lie inside the selected bias window")
        # SpectralPreprocessor performs the remaining interval validation.
        self.build_preprocessor()

    @property
    def resolved_scale_range_mev(self) -> tuple[float, float]:
        if self.scale_range_mev is not None:
            return self.scale_range_mev
        return (
            max(self.bias_min_mev, self.bias_cutoff_mev - self.scale_bandwidth_mev),
            self.bias_cutoff_mev,
        )

    def build_preprocessor(self) -> SpectralPreprocessor:
        return SpectralPreprocessor(
            output_points=self.output_points,
            bias_range_mev=(self.bias_min_mev, self.bias_cutoff_mev),
            baseline_range_mev=self.baseline_range_mev,
            scale_range_mev=self.resolved_scale_range_mev,
            clip=self.clip,
        )

    def to_metadata(self) -> dict[str, object]:
        metadata = asdict(self)
        metadata["resolved_scale_range_mev"] = self.resolved_scale_range_mev
        return metadata


@dataclass(frozen=True)
class PreparedTrainingDataset:
    dataset: SpectroscopyDataset
    preprocessor: SpectralPreprocessor
    config: TrainingPreprocessingConfig


def prepare_training_dataset(
    dataset: SpectroscopyDataset,
    config: TrainingPreprocessingConfig,
) -> PreparedTrainingDataset:
    """Crop and normalize simulations using the experiment-selected contract."""
    preprocessor = config.build_preprocessor()
    processed_spectra = preprocessor.transform_map(dataset.spectra, dataset.bias_mev)
    metadata = dict(dataset.metadata)
    metadata["training_preprocessing"] = config.to_metadata()
    prepared = replace(
        dataset,
        spectra=processed_spectra,
        bias_mev=preprocessor.output_bias_mev,
        metadata=metadata,
    )
    return PreparedTrainingDataset(prepared, preprocessor, config)

