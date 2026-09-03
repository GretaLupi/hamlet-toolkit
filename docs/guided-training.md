# Guided training and reusable artifacts

Experimentalists do not have to choose between only using our pretrained
models and building the entire ML pipeline themselves. The guided trainer is
the middle path: they choose the physical problem, experimental bias cutoff,
model family, and compute budget. The package controls leakage-safe splitting,
target scaling, early stopping, evaluation, and serialization.

## Minimal local-chain workflow

Start from a broad raw simulation dataset whose bias range covers the usable
experimental range. Select the cutoff after inspecting the experiment, then
use the same preprocessing contract for training and later inference.

```python
from hamlet.data import SpectroscopyDataset
from hamlet.training import (
    TrainingPreprocessingConfig,
    prepare_training_dataset,
    train_supervised,
)

raw = SpectroscopyDataset.load("simulations_0_to_100mev.npz")

prepared = prepare_training_dataset(
    raw,
    TrainingPreprocessingConfig(
        bias_cutoff_mev=50.0,
        output_points=200,
    ),
)

run = train_supervised(
    prepared,
    view="local_bonds",
    model="keras_mlp",       # also: keras_cnn, ridge, random_forest
    preset="standard",       # quick, standard, or research
)

print(run.metrics["validation"])
print(run.metrics["test"])
run.save("artifacts/heisenberg-local-mlp-cut50-v1")
```

The grouped split keeps every three-site window from a simulated chain in the
same partition. The scaler is fit using training chains only. Neural models
use early stopping and learning-rate reduction. All reported MAE and RMSE
values are converted back to meV.

For multi-seed runs, the trainer compares mean, median, and inverse-validation-
MAE-weighted aggregation using synthetic validation chains only. The rule with
the lowest validation MAE is frozen into the artifact before the test set is
evaluated. Experimental inference automatically uses that stored rule. Raw
member predictions and their unweighted spread are still reported, so robust
aggregation does not suppress an ensemble-disagreement warning.

## Open an artifact and analyze an experiment

First run `hamlet advise` with the manually chosen cutoff. Direct loading is
the lower-level API and assumes that this preflight has already passed.

```python
from hamlet.training import TrainingRun

run = TrainingRun.load("artifacts/heisenberg-local-mlp-cut50-v1")
analyzer = run.create_analyzer()
result = analyzer.analyze_csv("my_experimental_chain.csv")

print(result.coupling_mean)
print(result.coupling_std)
result.plot_summary().savefig("experimental_qc.png", dpi=180)
result.save_couplings_csv("inferred_couplings.csv")
result.save_report_json("inference_report.json")
```

No cutoff or output scaling is entered during inference: both come from the
artifact. This prevents accidentally applying a 50 meV model to data prepared
with a different window.

## Presets

| Preset | Seeds | Maximum epochs | Early-stopping patience | Intended use |
|---|---:|---:|---:|---|
| `quick` | 1 | 20 | 4 | API check and debugging |
| `standard` | 3 | 100 | 12 | Normal experimental analysis |
| `research` | 5 | 250 | 25 | Final benchmark or publication |

The preset is a compute budget, not a hidden hyperparameter optimization.
Model architecture remains explicit through `model_options`. For example:

```python
from hamlet.models import MLPConfig

run = train_supervised(
    prepared,
    view="local_bonds",
    model="keras_mlp",
    preset="standard",
    model_options={
        "config": MLPConfig(
            hidden_units=(512, 256, 128),
            dropout=0.25,
            learning_rate=3e-4,
        )
    },
)
```

Optuna should be added as a separate optional layer later. It should tune only
against grouped validation chains, use an explicit trial/compute budget, and
leave the untouched test partition for one final evaluation.

## Artifact contents

Each artifact directory contains one model file per seed and `manifest.json`.
The manifest records:

- model family, options, seeds, and training preset;
- system, learning view, number of sites, and target names;
- preprocessing including the chosen cutoff and output grid;
- target-scaling minima and maxima;
- validation/test metrics and split sizes;
- the validation-selected ensemble aggregation rule and candidate metrics;
- dataset metadata and the convention `1 DMRGPy unit = 10 meV`.

An artifact directory must be absent or empty when saving, which prevents an
old model and a new manifest from being mixed accidentally.
