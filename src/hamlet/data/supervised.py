"""Convert physical datasets into explicit supervised-learning views."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .dataset import SpectroscopyDataset
from ..preprocessing.windows import make_local_windows

LearningView = Literal["local_bonds", "global"]


@dataclass(frozen=True)
class SupervisedDataset:
    inputs: NDArray[np.float32]
    targets: NDArray[np.float32]
    group_ids: NDArray[np.int64]
    target_names: tuple[str, ...]
    view: LearningView


def as_supervised(
    dataset: SpectroscopyDataset, view: LearningView
) -> SupervisedDataset:
    """Create local bond examples or one global example per chain.

    ``group_ids`` preserves simulated-chain identity so windows from the same
    chain can never leak across train, validation, and test partitions.
    """
    spectra = np.asarray(dataset.spectra, dtype=np.float32)
    targets = np.asarray(dataset.targets_mev, dtype=np.float32)
    if view == "global":
        return SupervisedDataset(
            inputs=spectra.reshape(dataset.n_samples, -1),
            targets=targets.copy(),
            group_ids=np.arange(dataset.n_samples, dtype=np.int64),
            target_names=dataset.target_names,
            view=view,
        )
    if view != "local_bonds":
        raise ValueError(f"unknown supervised view: {view}")
    if targets.shape[1] != dataset.n_sites - 1:
        raise ValueError("local_bonds requires one target per nearest-neighbor bond")

    windows = make_local_windows(spectra, window_sites=3)
    local_targets = np.stack(
        [targets[:, i : i + 2] for i in range(dataset.n_sites - 2)], axis=1
    )
    n_windows = dataset.n_sites - 2
    return SupervisedDataset(
        inputs=windows.reshape(dataset.n_samples * n_windows, -1),
        targets=local_targets.reshape(dataset.n_samples * n_windows, 2),
        group_ids=np.repeat(np.arange(dataset.n_samples), n_windows).astype(np.int64),
        target_names=("J_left", "J_right"),
        view=view,
    )

