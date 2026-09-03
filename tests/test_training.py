import numpy as np

from hamlet.data.supervised import SupervisedDataset
from hamlet.training import MinMaxTargetScaler, grouped_split


def make_dataset():
    groups = np.repeat(np.arange(10), 4)
    return SupervisedDataset(
        inputs=np.arange(120, dtype=np.float32).reshape(40, 3),
        targets=np.column_stack([groups, groups + 10]).astype(np.float32),
        group_ids=groups,
        target_names=("a", "b"),
        view="local_bonds",
    )


def test_grouped_split_has_no_chain_leakage():
    split = grouped_split(make_dataset(), validation_fraction=0.2, test_fraction=0.2)
    train = set(split.train.group_ids)
    validation = set(split.validation.group_ids)
    test = set(split.test.group_ids)
    assert train.isdisjoint(validation | test)
    assert validation.isdisjoint(test)
    assert train | validation | test == set(range(10))


def test_target_scaler_round_trip_and_metadata():
    targets = np.array([[30, 0], [45, 10], [35, 4]], dtype=np.float32)
    scaler = MinMaxTargetScaler.fit(targets)
    scaled = scaler.transform(targets)
    np.testing.assert_allclose(scaler.inverse_transform(scaled), targets)
    assert scaler.to_metadata() == {"minimum": [30.0, 0.0], "maximum": [45.0, 10.0]}

