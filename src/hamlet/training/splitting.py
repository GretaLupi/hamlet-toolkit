"""Leakage-safe splitting for global samples and local windows."""

from dataclasses import dataclass

import numpy as np

from ..data.supervised import SupervisedDataset


@dataclass(frozen=True)
class SupervisedSplit:
    train: SupervisedDataset
    validation: SupervisedDataset
    test: SupervisedDataset


def grouped_split(
    dataset: SupervisedDataset,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> SupervisedSplit:
    """Split entire simulated chains, never individual local windows."""
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions cannot be negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1")
    groups = np.unique(dataset.group_ids)
    if groups.size < 3:
        raise ValueError("at least three groups are required for train/validation/test")
    shuffled = np.random.default_rng(seed).permutation(groups)
    n_test = max(1, round(groups.size * test_fraction)) if test_fraction else 0
    n_validation = max(1, round(groups.size * validation_fraction)) if validation_fraction else 0
    if n_test + n_validation >= groups.size:
        raise ValueError("split fractions leave no training groups")
    test_groups = shuffled[:n_test]
    validation_groups = shuffled[n_test : n_test + n_validation]
    train_groups = shuffled[n_test + n_validation :]
    return SupervisedSplit(
        train=_select(dataset, train_groups),
        validation=_select(dataset, validation_groups),
        test=_select(dataset, test_groups),
    )


def _select(dataset: SupervisedDataset, groups: np.ndarray) -> SupervisedDataset:
    mask = np.isin(dataset.group_ids, groups)
    return SupervisedDataset(
        inputs=dataset.inputs[mask],
        targets=dataset.targets[mask],
        group_ids=dataset.group_ids[mask],
        target_names=dataset.target_names,
        view=dataset.view,
    )

