# Supervised workflows

Dataset generation and supervised representation are separate decisions. A
single physical dataset can be saved, inspected, benchmarked, and converted to
the learning view required by a chosen model.

The same DMRGPy generation API is available from configuration using
`dataset.generate` and `hamlet generate`. The configuration workflow adds
chunk checkpoints, exact-recipe cache checking, and atomic final output; see
[project-workflow.md](project-workflow.md).

## Bond-inhomogeneous nearest-neighbor chain

Each bond has its own parameter `J_bond_i`. The three-site local view produces
two targets per window and can later be applied to chains of other lengths.

```python
from hamlet import InhomogeneousHeisenbergFamily, generate_dataset
from hamlet.data import as_supervised
from hamlet.models import create_supervised_model
from hamlet.simulation import DmrgpySimulator, SpectroscopyProtocol
from hamlet.training import MinMaxTargetScaler, grouped_split
from hamlet.training import (
    TrainingPreprocessingConfig,
    prepare_training_dataset,
)

family = InhomogeneousHeisenbergFamily(
    n_sites=12,
    coupling_range_mev=(30.0, 45.0),
)
protocol = SpectroscopyProtocol.uniform(
    bias_range_mev=(0.0, 100.0),
    points=200,
    broadening_mev=0.5,  # equivalent to delta=0.05 in DMRGPy units
    output_quantity="didv",
)
raw_dataset = generate_dataset(
    family,
    DmrgpySimulator(),
    protocol,
    n_samples=1000,
    seed=42,
    progress=lambda done, total: print(f"{done}/{total}"),
)
cutoff_config = TrainingPreprocessingConfig(
    bias_cutoff_mev=50.0,  # chosen after inspecting the usable experiment range
    output_points=200,
    baseline_range_mev=(0.0, 3.0),
    scale_range_mev=(40.0, 50.0),
)
prepared = prepare_training_dataset(raw_dataset, cutoff_config)
dataset = prepared.dataset
dataset.save("inhomogeneous_L12_cut50meV.npz")

# Public values are physical meV. The backend converts them automatically using
# 1 DMRGPy energy unit = 10 meV.

training = as_supervised(dataset, view="local_bonds")
split = grouped_split(training, validation_fraction=0.2, test_fraction=0.1)
scaler = MinMaxTargetScaler.fit(split.train.targets)
model = create_supervised_model(
    "keras_mlp",
    input_dim=training.inputs.shape[1],
    output_dim=training.targets.shape[1],
)
model.fit(
    split.train.inputs,
    scaler.transform(split.train.targets),
    validation_data=(
        split.validation.inputs,
        scaler.transform(split.validation.targets),
    ),
)
```

The grouped split guarantees that all windows from one simulated chain remain
in the same partition. Save the scaler metadata with the trained model so its
normalized predictions can always be converted back to meV.

## Homogeneous J1-J2-... chain

Here `J1` applies to every nearest-neighbor pair, `J2` to every
next-nearest-neighbor pair, and so forth. These are global targets, so the first
implementation learns from the complete fixed-length spectral map.

```python
from hamlet import HomogeneousHeisenbergFamily, generate_dataset
from hamlet.data import as_supervised
from hamlet.models import create_supervised_model
from hamlet.simulation import DmrgpySimulator, SpectroscopyProtocol

family = HomogeneousHeisenbergFamily(
    n_sites=10,
    coupling_ranges_mev=(
        (30.0, 45.0),  # J1
        (0.0, 10.0),   # J2
    ),
)
dataset = generate_dataset(
    family,
    DmrgpySimulator(),
    SpectroscopyProtocol.uniform(output_quantity="didv"),
    n_samples=1000,
    seed=42,
)
training = as_supervised(dataset, view="global")

# Lightweight baseline; alternatives include "random_forest", "keras_mlp", and
# "keras_cnn".
model = create_supervised_model(
    "ridge",
    input_dim=training.inputs.shape[1],
    output_dim=training.targets.shape[1],
)
model.fit(training.inputs, training.targets)
```

See [model-selection.md](model-selection.md) for CNN shapes, typed MLP/CNN
configuration, and a leakage-safe hyperparameter-selection example.

The global flattened representation is deliberately fixed-length. Supporting a
single homogeneous model across multiple chain lengths will require a pooling,
convolutional, or set-based architecture and will be added as a separate model
family rather than hidden behind padding.

## Anisotropic long-range pilot

The registered `heisenberg/xxz_long_range` variant uses four named targets in
a fixed order: `J1_xy`, `J2`, `J3`, `Jz`. Its nearest-neighbour term is
`J1_xy*(SxSx+SySy) + Jz*SzSz`; J2 and J3 are isotropic distance couplings. It
must not be represented as generic `J1...J4`, because that would erase the
Hamiltonian semantics. See
[`examples/heisenberg_xxz_j1j2j3_l8.yaml`](../examples/heisenberg_xxz_j1j2j3_l8.yaml).
For this anisotropic family, `total_spin` computes the weighted sum of the
`Sxx`, `Syy`, and `Szz` dynamical autocorrelators before forming the dI/dV-like
integrated response. `Sz` remains selectable for controlled ablations.
