# Experiment-selected bias cutoff

The cutoff is selected from the usable experimental spectrum and then becomes
part of the training contract. It is not fixed globally by the package.

## Workflow

First create and inspect the raw canonical experiment project:

```bash
hamlet inspect-experiment my_import.yaml \
  --mode heisenberg --variant inhomogeneous \
  --output-dir experiments/my_chain \
  --candidate-cutoffs 50 60 67 70
```

Open `experiment_inspection.html` and choose the range judged experimentally
reliable. Inspection does not crop or normalize the measurement.

Before training or inference, check registered resources explicitly:

```bash
hamlet advise experiments/my_chain/experiment_manifest.json --cutoff 67 \
  --artifact-root models/heisenberg --dataset simulations_0_to_100mev.npz
```

The advisor never substitutes weights from another cutoff. It distinguishes
artifact reuse, retraining from the broad dataset, generating an expanded
dataset, and insufficient experimental coverage.

Generate or retain synthetic spectra over a range at least this wide, then
prepare the training set with that cutoff:

```python
from hamlet.training import (
    TrainingPreprocessingConfig,
    prepare_training_dataset,
)

cutoff_config = TrainingPreprocessingConfig(
    bias_cutoff_mev=chosen_cutoff_mev,
    output_points=268,             # for example, keep 0.25 meV spacing
    baseline_range_mev=(0.0, 3.0),
    scale_range_mev=(57.0, 67.0),
)

prepared = prepare_training_dataset(raw_synthetic_dataset, cutoff_config)
training = as_supervised(prepared.dataset, view="local_bonds")
```

The prepared dataset records the cutoff and all associated preprocessing in its
metadata. The exact same object is then reused for experiment inference:

```python
analyzer = ExperimentalChainAnalyzer(
    trained_models,
    preprocessor=prepared.preprocessor,
    output_range=(30.0, 45.0),
    coupling_unit="meV",
)
from hamlet import load_experiment_project

measurement, _ = load_experiment_project(
    "experiments/my_chain/experiment_manifest.json"
)
result = analyzer.analyze_measurement(measurement)
```

Changing the cutoff creates a different model contract and therefore requires
retraining. An experiment extending beyond the trained cutoff is cropped safely;
one ending below it is incompatible with that model.

## Different cutoffs for different sites

The present MLP and CNN assume that every site channel represents the same
physical bias grid. Applying unrelated crops to individual sites would violate
that assumption even if every crop were resampled to 200 numbers.

Per-site cutoffs can be supported later with a model trained using randomized
site masks. Its inputs should include spectral values, physical bias coordinates,
and an availability mask. Until such a model is trained, the package uses one
experiment-selected cutoff per model and refuses to hide missing site ranges.
