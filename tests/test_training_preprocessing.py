import numpy as np
import pytest

from hamlet.data import SpectroscopyDataset
from hamlet.training import (
    TrainingPreprocessingConfig,
    prepare_training_dataset,
)


def raw_dataset():
    bias = np.linspace(0, 100, 401)
    spectra = np.stack(
        [
            np.stack([1 + sample + site + np.sin(bias / 10) for site in range(5)])
            for sample in range(3)
        ]
    )
    return SpectroscopyDataset(
        spectra=spectra,
        targets_mev=np.full((3, 4), 35.0),
        bias_mev=bias,
        target_names=tuple(f"J{i}" for i in range(4)),
        system_type="InhomogeneousHeisenbergChain",
    )


def test_experiment_selected_cutoff_is_applied_and_recorded():
    config = TrainingPreprocessingConfig(
        bias_cutoff_mev=37.0,
        output_points=120,
        baseline_range_mev=(0.0, 2.0),
        scale_bandwidth_mev=7.0,
    )
    raw = raw_dataset()
    prepared = prepare_training_dataset(raw, config)
    assert prepared.dataset.spectra.shape == (3, 5, 120)
    assert prepared.dataset.bias_mev[-1] == 37.0
    assert config.resolved_scale_range_mev == (30.0, 37.0)
    assert prepared.dataset.metadata["training_preprocessing"]["bias_cutoff_mev"] == 37.0
    np.testing.assert_allclose(
        prepared.dataset.spectra[0],
        prepared.preprocessor.transform_map(raw.spectra[0], raw.bias_mev),
    )


def test_cutoff_must_be_covered_by_the_training_dataset():
    config = TrainingPreprocessingConfig(bias_cutoff_mev=120.0)
    with pytest.raises(ValueError, match="does not cover"):
        prepare_training_dataset(raw_dataset(), config)

