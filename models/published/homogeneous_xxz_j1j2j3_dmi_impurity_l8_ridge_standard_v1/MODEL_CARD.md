# Model card — `homogeneous_xxz_j1j2j3_dmi_impurity_l8_ridge_standard_v1`

Reference global-view model for the `homogeneous_xxz_j1j2j3_dmi_impurity` system at L = 8, trained on 3000 exact-diagonalisation chains.

## What it does

Takes one complete `(8 sites x 61 bias points)` dI/dV map of a homogeneous spin-1/2 host chain with substituted site impurities and returns the global exchange parameters `J1_xy, J2, J3, Jz, D_z_magnitude` in meV.

This is a **global-view** model: the whole site-by-bias map is a single input.
It is therefore valid only for chains of exactly **L = 8**. A different chain length needs a model trained for that length, and the package refuses the mismatch rather than padding or cropping.

## Training data

- source dataset: `dmi_impurity_l8.npz` (3000 simulated chains)
- simulator: DMRGPy, `dynamics_mode=ED` (exact diagonalisation, exact at L=8)
- observable contract: `total_spin`
- output quantity: `didv`
- bias window: 0 to 20 meV over 61 points
- broadening: 0.25 meV
- fixed impurity configuration: site 1 (S=1, D=0 meV, E=2.0 meV, phi=0 rad), site 4 (S=1, D=0 meV, E=2.0 meV, phi=0 rad), site 6 (S=1, D=0 meV, E=2.0 meV, phi=0 rad)
- fixed transverse field: 0 meV

Sampled parameter ranges — **predictions outside these ranges are extrapolation**:

| parameter | min [meV] | max [meV] |
|---|---|---|
| `J1_xy` | 2.00 | 6.00 |
| `J2` | -1.50 | 1.50 |
| `J3` | -1.00 | 1.00 |
| `Jz` | 2.00 | 6.00 |
| `D_z_magnitude` | 0.30 | 2.50 |

## Accuracy

Model `ridge`, preset `standard` (3 seeds), aggregation `mean`. Grouped split by simulated chain, so no chain appears in more than one partition.

- validation MAE: **0.173 meV**
- held-out test MAE: **0.180 meV**
- held-out test RMSE: 0.262 meV
- correlation fidelity: 0.992
- split sizes: {'train_groups': 2100, 'validation_groups': 600, 'test_groups': 300, 'split_seed': 42}

Per parameter on the held-out split, with skill against a training-mean
baseline (`1 - MAE_model / MAE_mean`):

| parameter | test MAE [meV] | skill |
|---|---|---|
| `J1_xy` | 0.181 | 0.82 |
| `J2` | 0.095 | 0.88 |
| `J3` | 0.131 | 0.74 |
| `Jz` | 0.237 | 0.77 |
| `D_z_magnitude` | 0.257 | 0.55 |
Model options: `{'alpha': 0.001}`

Chosen by `scripts/benchmark_homogeneous.py` over 5 resampled splits on validation MAE (mean validation MAE by model: `ridge` 0.178, `random_forest` 0.234). See `dmi_impurity_l8_benchmark.json`.

## When NOT to reuse this model

- the chain is not L = 8
- the measurement does not cover 0 to 20 meV
- the experiment is not the `homogeneous_xxz_j1j2j3_dmi_impurity` system, or not the `global` view
- the simulated observable expected by the experiment is not `total_spin`
- the couplings are expected outside the sampled ranges above
- the impurity count, sites, species, anisotropies, or field differ from the fixed training conditions above

`hamlet advise` checks the stored system, view, chain length, cutoff, observable,
and parameter-range contracts, and it also compares the fixed impurity and
field conditions above. Because `system_type` is identical for every impurity
chain whatever its impurity count, sites, species or anisotropies, those
conditions must be declared explicitly via `experiment_conditions`; an
undeclared or differing configuration blocks reuse rather than being assumed
to match.

## Honest limits

The quoted accuracy is against **simulated** spectra drawn from the same generator that produced the training set. It says nothing about whether the simulator describes any particular real material, and it is not a calibrated uncertainty: ensemble spread reported at inference measures agreement between seeds, not distance from truth.

Ridge regression is deterministic, so the three saved seed members are
identical and their ensemble spread is zero. Use the five resampled-split
benchmark—not ensemble disagreement—as the stability estimate for this model.

## Reproducing

```bash
python scripts/publish_homogeneous_artifact.py \
  --dataset dmi_impurity_l8.npz \
  --model ridge --preset standard \
  --output-dir models/published/homogeneous_xxz_j1j2j3_dmi_impurity_l8_ridge_standard_v1
```

Generated 2026-09-07 with hamlet 0.1.0 on Python 3.11.5.
