# Homogeneous global-view benchmark — dmi_impurity_l8.npz

- system: `homogeneous_xxz_j1j2j3_dmi_impurity`  view: `global`
- chains: 3000  sites: 8
- targets: J1_xy, J2, J3, Jz, D_z_magnitude
- bias cutoff: 20 meV over 61 points
- split seeds: [42, 43, 44, 45, 46]  model seeds: [42, 43, 44]

Scores are the mean +/- spread over the split seeds. Selection uses the
validation column only; the test column is reported, never optimised against.

## Overall

| model | validation MAE | test MAE | train s | infer s | ens. params |
|---|---|---|---|---|---|
| `ridge` | 0.178 +/- 0.003 | 0.178 +/- 0.002 | 1 | 0.005 | n/a |
| `random_forest` | 0.234 +/- 0.004 | 0.235 +/- 0.003 | 68 | 0.359 | n/a |

**Selected on validation MAE: `ridge`**

## Per parameter

`skill` is 1 - MAE_model / MAE_training_mean. `lower` is mean - spread, and
the verdict needs lower >= 0.5 to read `resolved`, >= 0.2
to read `marginal`. A parameter that cannot clear the pessimistic end of its
own spread has not been shown to be identifiable.

### `ridge`

| parameter | test MAE [meV] | skill | lower | verdict |
|---|---|---|---|---|
| `J1_xy` | 0.169 +/- 0.004 | 0.83 +/- 0.01 | 0.82 | resolved |
| `J2` | 0.093 +/- 0.002 | 0.88 +/- 0.00 | 0.87 | resolved |
| `J3` | 0.136 +/- 0.003 | 0.73 +/- 0.00 | 0.73 | resolved |
| `Jz` | 0.235 +/- 0.007 | 0.77 +/- 0.01 | 0.76 | resolved |
| `D_z_magnitude` | 0.258 +/- 0.004 | 0.54 +/- 0.01 | 0.53 | resolved |

### `random_forest`

| parameter | test MAE [meV] | skill | lower | verdict |
|---|---|---|---|---|
| `J1_xy` | 0.214 +/- 0.004 | 0.79 +/- 0.01 | 0.78 | resolved |
| `J2` | 0.206 +/- 0.007 | 0.73 +/- 0.01 | 0.72 | resolved |
| `J3` | 0.196 +/- 0.004 | 0.61 +/- 0.01 | 0.60 | resolved |
| `Jz` | 0.263 +/- 0.006 | 0.74 +/- 0.00 | 0.73 | resolved |
| `D_z_magnitude` | 0.294 +/- 0.011 | 0.48 +/- 0.01 | 0.46 | marginal |
