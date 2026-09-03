import numpy as np

from hamlet.training import (
    AugmentationConfig,
    EnsembleAggregation,
    TrainingDistributionProfile,
    augment_experimental_like,
    calibrate_augmentation,
    select_ensemble_aggregation,
)
from test_guided_training import make_training_dataset


def test_distribution_profile_scores_nearest_and_round_trips(tmp_path):
    rng = np.random.default_rng(7)
    inputs = rng.normal(size=(100, 12)).astype(np.float32)
    targets = rng.uniform(30, 45, size=(100, 2)).astype(np.float32)
    profile = TrainingDistributionProfile.fit(inputs, targets, max_references=20)

    in_scores = profile.score(inputs[:5])
    out_scores = profile.score(np.full((2, 12), 20, dtype=np.float32))
    assert np.mean(out_scores) > np.mean(in_scores)
    assert profile.threshold_95 > 0

    nearest = profile.nearest(inputs[:3])
    assert nearest.inputs.shape == (3, 12)
    assert nearest.targets.shape == (3, 2)
    assert nearest.distances.shape == (3,)

    path = tmp_path / "profile.npz"
    profile.save(path)
    loaded = TrainingDistributionProfile.load(path)
    np.testing.assert_allclose(loaded.score(inputs[:5]), in_scores)


def test_augmentation_is_reproducible_and_calibration_returns_candidate():
    dataset = make_training_dataset()
    config = dict(
        seed=9,
        broadening_points=(1.0, 3.0),
        energy_shift_mev=(-1.0, 1.0),
        energy_stretch=(0.98, 1.02),
        background_quadratic=(-0.01, 0.01),
        amplitude_range=(0.9, 1.1),
    )
    first = augment_experimental_like(dataset, **config)
    second = augment_experimental_like(dataset, **config)
    np.testing.assert_allclose(first.spectra, second.spectra)

    result = calibrate_augmentation(
        dataset,
        dataset.spectra[0],
        dataset.bias_mev,
        cutoffs_mev=(50.0,),
        output_points=30,
        candidates={
            "reference": AugmentationConfig(),
            "smooth": AugmentationConfig(broadening_points=(1.0, 3.0)),
        },
    )
    assert result.selected_name in {"reference", "smooth"}
    assert set(result.candidate_metrics) == {"reference", "smooth"}


def test_aggregation_is_selected_on_validation_predictions_only():
    expected = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    predictions = np.array(
        [
            [[8.0], [8.0], [8.0]],
            [[1.0], [2.0], [3.0]],
            [[1.1], [2.1], [3.1]],
        ],
        dtype=np.float32,
    )
    rule, metrics = select_ensemble_aggregation(predictions, expected)
    assert rule.method in {"median", "validation_weighted"}
    assert metrics[rule.method]["mae"] == min(item["mae"] for item in metrics.values())
    assert metrics[rule.method]["mae"] < metrics["mean"]["mae"]
    assert np.mean(np.abs(rule.aggregate(predictions) - expected)) < 0.11

    weighted = EnsembleAggregation("validation_weighted", (0.0, 0.75, 0.25))
    np.testing.assert_allclose(
        weighted.aggregate(predictions),
        0.75 * predictions[1] + 0.25 * predictions[2],
    )
