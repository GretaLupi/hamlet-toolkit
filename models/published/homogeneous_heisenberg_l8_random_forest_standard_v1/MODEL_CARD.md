# Model card — `homogeneous_heisenberg_l8_random_forest_standard_v1`

Reference global-view model for the `homogeneous_heisenberg` system at L = 8, trained on 3000 exact-diagonalisation chains.

## What it does

Takes one complete `(8 sites x 61 bias points)` dI/dV map of a homogeneous spin-1/2 chain and returns the global exchange parameters `J1, J2` in meV.

This is a **global-view** model: the whole site-by-bias map is a single input.
It is therefore valid only for chains of exactly **L = 8**. A different chain length needs a model trained for that length, and the package refuses the mismatch rather than padding or cropping.

## Training data

- source dataset: `homogeneous_heisenberg_l8_ed.npz` (3000 simulated chains)
- simulator: DMRGPy, `dynamics_mode=ED` (exact diagonalisation, exact at L=8)
- observable contract: `Sz`
- output quantity: `didv`
- bias window: 0 to 60 meV over 61 points
- broadening: 1.0 meV

Sampled parameter ranges — **predictions outside these ranges are extrapolation**:

| parameter | min [meV] | max [meV] |
|---|---|---|
| `J1` | 30.02 | 45.00 |
| `J2` | 0.01 | 10.00 |

## Accuracy

Model `random_forest`, preset `standard` (3 seeds), aggregation `validation_weighted`. Grouped split by simulated chain, so no chain appears in more than one partition.

- validation MAE: **0.113 meV**
- held-out test MAE: **0.114 meV**
- held-out test RMSE: 0.164 meV
- correlation fidelity: 1.000
- split sizes: {'train_groups': 2100, 'validation_groups': 600, 'test_groups': 300, 'split_seed': 42}

Per parameter on the held-out split, with skill against a training-mean
baseline (`1 - MAE_model / MAE_mean`):

| parameter | test MAE [meV] | skill |
|---|---|---|
| `J1` | 0.077 | 0.98 |
| `J2` | 0.151 | 0.94 |
Model options: `{'n_estimators': 200, 'min_samples_leaf': 2, 'n_jobs': -1}`

Chosen by `scripts/benchmark_homogeneous.py` over 5 resampled splits on validation MAE (mean validation MAE by model: `random_forest` 0.124, `ridge` 0.157). See `homogeneous_heisenberg_l8_ed_benchmark.json`.

## When NOT to reuse this model

- the chain is not L = 8
- the measurement does not cover 0 to 60 meV
- the experiment is not the `homogeneous_heisenberg` system, or not the `global` view
- the simulated observable expected by the experiment is not `Sz`
- the couplings are expected outside the sampled ranges above

`hamlet advise` checks every one of these against the stored manifest before inference, and reports `use_existing_model` only when all of them hold.

## Honest limits

The quoted accuracy is against **simulated** spectra drawn from the same generator that produced the training set. It says nothing about whether the simulator describes any particular real material, and it is not a calibrated uncertainty: ensemble spread reported at inference measures agreement between seeds, not distance from truth.

## Reproducing

```bash
python scripts/publish_homogeneous_artifact.py \
  --dataset homogeneous_heisenberg_l8_ed.npz \
  --model random_forest --preset standard \
  --output-dir models/published/homogeneous_heisenberg_l8_random_forest_standard_v1
```

Generated 2026-09-03 with hamlet 0.1.0 on Python 3.11.5.