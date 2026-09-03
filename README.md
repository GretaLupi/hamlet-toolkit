<p align="center">
  <img src="assets/logos/hamlet-logo.png" alt="HamLeT — Hamiltonian Learning Toolkit" width="360">
</p>

# HamLeT

**Hamiltonian Learning Toolkit**

> "Though this be madness, yet there is method in't." — Hamlet, Act 2.

HamLeT is an experimentalist-facing Python toolkit for generating spectroscopy
datasets, training inverse models, and reconstructing physical Hamiltonians
from measurements of spin-1/2 Heisenberg chains.

The installable distribution is `hamlet-toolkit`, the canonical Python import
is `hamlet`, and the canonical command is `hamlet`. (The bare `hamlet`
distribution name is already used by an unrelated project on
[PyPI](https://pypi.org/project/hamlet/), hence the distinct distribution
name.)

## Status

- **Bond-inhomogeneous Heisenberg chains** (`local_bonds` view) — the
  validated, ready-to-use workflow. A three-site sliding-window estimator
  predicts local exchange couplings and works on any chain length. Validated
  end-to-end against a real experimental chain and a small physical DMRGPy
  recovery benchmark.
- **Homogeneous `J1-J2` chains at L=8** (`global` view) — a first reference
  model ships in
  [models/published/](models/published/homogeneous_heisenberg_l8_random_forest_standard_v1),
  trained on 3000 exact-diagonalisation chains: 0.114 meV held-out test MAE,
  with both couplings recovered well clear of a training-mean baseline across
  five resampled splits. Validated against simulated spectra only — not yet
  against a measurement with independently known parameters. See its
  [model card](models/published/homogeneous_heisenberg_l8_random_forest_standard_v1/MODEL_CARD.md)
  for the valid parameter ranges and the conditions under which it must not be
  reused.
- **Anisotropic homogeneous extensions** (XXZ+J2+J3, and the same plus a
  uniform DMI magnitude) — implemented end to end with a configurable
  simulated observable (`Sz` or `total_spin`), and improving steadily with
  data: at 500 chains the exchange couplings reach 0.36–0.87 skill over a
  training-mean baseline. The DMI magnitude does not: it stays at or below
  that baseline even at fifteen times the pilot dataset, which suggests it is
  not identifiable from unpolarised total-spin spectra at this chain length
  rather than merely undersampled. Development-only.

Physical energy inputs and outputs are always in meV. See
[docs/user-guide.md](docs/user-guide.md) for the complete workflow.

## Install

```bash
python -m pip install -e '.[dev,io]'
pytest
```

Add the `ml` extra for Keras/scikit-learn models and `simulation` for DMRGPy
dataset generation, or install everything with the `all` extra.

## Quick start

The simplest interface is one configuration and one command:

```bash
hamlet run examples/heisenberg_train_and_analyze.yaml
```

This calibrates the manually selected cutoff against the experiment, freezes
its preprocessing contract, trains and validates the requested model,
performs inference, and exports a self-contained HTML analysis report.

The full path — importing raw per-site files, choosing a cutoff, letting the
package decide whether to reuse, retrain, or generate a model, and reading the
report — is in [docs/user-guide.md](docs/user-guide.md).

## Python API

```python
import numpy as np

from hamlet.inference import LocalChainEstimator
from hamlet.preprocessing import SpectralPreprocessor

# One experimental map with shape (sites, measured bias points).
didv = np.load("didv.npy")
measured_bias = np.load("bias_mev.npy")

preprocessor = SpectralPreprocessor(
    output_points=200,
    bias_range_mev=(0.0, 50.0),
    baseline_range_mev=(0.0, 3.0),
    scale_range_mev=(40.0, 50.0),
)
processed = preprocessor.transform_map(didv, measured_bias)

# Any object exposing predict(windows) -> (n_windows, 2) can be used.
estimator = LocalChainEstimator(model, coupling_range_mev=(30.0, 45.0))
result = estimator.predict(processed)

print(result.couplings_mev)  # length = n_sites - 1
```

## Demos

- [notebooks/04_l8_three_mode_workflow.ipynb](notebooks/04_l8_three_mode_workflow.ipynb) —
  **start here.** One L=8 chain length run through all three interpretations
  (bond-inhomogeneous, homogeneous XXZ, homogeneous XXZ+DMI), scored against known
  ground truth, including sliding-window inference on a held-out chain. Runs in
  seconds on the small datasets in [examples/l8_demo/](examples/l8_demo/).
- [notebooks/01_supervised_workflows.ipynb](notebooks/01_supervised_workflows.ipynb) —
  an executed end-to-end supervised workflow with saved plots and outputs.
- [notebooks/02_full_potential_experimental_interface.ipynb](notebooks/02_full_potential_experimental_interface.ipynb) —
  the diagnostics-first experimental workflow on a real experimental chain,
  comparing several research models.
- [notebooks/03_simulation_experiment_validation.ipynb](notebooks/03_simulation_experiment_validation.ipynb) —
  the simulation-to-experiment validation study: spectral distribution
  scoring, nearest synthetic windows, inference diagnostics, and DMRGPy
  forward residuals.

## Advanced: label-free augmentation calibration

```bash
hamlet-calibrate-augmentation \
  --reference-npz dataset_30_40.npz dataset_35_45.npz \
  --experiment ChainIH_L6_SYM.csv \
  --cutoffs 50 \
  --output calibration-cut50.json

hamlet-train-cutoff-bank \
  --reference-npz dataset_30_40.npz dataset_35_45.npz \
  --cutoffs 50 --comparison-cutoff 50 --candidates keras_mlp \
  --preset standard --augmentation-calibration calibration-cut50.json \
  --output-dir models/heisenberg-cut50
```

This optional command compares measured spectral compatibility only; it never
uses experimental Hamiltonian labels or inverse-model predictions. It's a
research utility, not required for ordinary model reuse or inference.

## Contributing

Setup, the conventions that protect scientific results, and the versioning
policy are in [CONTRIBUTING.md](CONTRIBUTING.md). Notable changes are recorded
in [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
