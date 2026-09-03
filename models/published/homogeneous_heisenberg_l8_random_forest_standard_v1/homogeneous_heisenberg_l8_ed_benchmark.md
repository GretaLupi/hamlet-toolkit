# Homogeneous global-view benchmark — homogeneous_heisenberg_l8_ed.npz

- system: `homogeneous_heisenberg`  view: `global`
- chains: 3000  sites: 8
- targets: J1, J2
- bias cutoff: 60 meV over 61 points
- split seeds: [42, 43, 44, 45, 46]  model seeds: [42, 43, 44]

Scores are the mean +/- spread over the split seeds. Selection uses the
validation column only; the test column is reported, never optimised against.

## Overall

| model | validation MAE | test MAE | train s | infer s | ens. params |
|---|---|---|---|---|---|
| `random_forest` | 0.124 +/- 0.005 | 0.126 +/- 0.003 | 146 | 0.171 | n/a |
| `ridge` | 0.157 +/- 0.003 | 0.156 +/- 0.004 | 0 | 0.001 | n/a |

**Selected on validation MAE: `random_forest`**

## Per parameter

`skill` is 1 - MAE_model / MAE_training_mean. `lower` is mean - spread, and
the verdict needs lower >= 0.5 to read `resolved`, >= 0.2
to read `marginal`. A parameter that cannot clear the pessimistic end of its
own spread has not been shown to be identifiable.

### `random_forest`

| parameter | test MAE [meV] | skill | lower | verdict |
|---|---|---|---|---|
| `J1` | 0.086 +/- 0.002 | 0.98 +/- 0.00 | 0.98 | resolved |
| `J2` | 0.167 +/- 0.003 | 0.93 +/- 0.00 | 0.93 | resolved |

### `ridge`

| parameter | test MAE [meV] | skill | lower | verdict |
|---|---|---|---|---|
| `J1` | 0.100 +/- 0.003 | 0.97 +/- 0.00 | 0.97 | resolved |
| `J2` | 0.213 +/- 0.004 | 0.91 +/- 0.00 | 0.91 | resolved |
