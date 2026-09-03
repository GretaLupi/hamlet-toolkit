# Guided experiment projects and physical modes

Experimentalists should begin with raw per-site files, not with a hand-built
ML input CSV. They inspect the canonical signals and choose the physical
interpretation used for compatibility, training, and inference.

```bash
hamlet modes
```

This lists registered variants and states whether complete experimental
inference is available. Then run:

```bash
hamlet inspect-experiment my_import.yaml \
  --mode heisenberg \
  --variant inhomogeneous \
  --output-dir experiments/my_chain \
  --candidate-cutoffs 30 40 50 70 100
```

The import recipe still declares laboratory-specific filenames, delimiters,
exact column names, units, site-order rules, missing-data policy, and grid
policy. The mode declares what the resulting canonical measurement means
physically.

## Current mode registry

| Mode | Variant | System identifier | View | Simulated observable |
|---|---|---|---|---|
| `heisenberg` | `inhomogeneous` | `inhomogeneous_heisenberg` | `local_bonds` | `Sz` (historical approximation) |
| `heisenberg` | `homogeneous` | `homogeneous_heisenberg` | `global` | `Sz` by default |
| `heisenberg` | `xxz_long_range` | `homogeneous_xxz_j1j2j3` | `global` | `total_spin` recommended |

The inhomogeneous profile requires at least three sites because its estimator
consumes three adjacent spectra and can slide over different chain lengths.
The homogeneous profile uses the whole site-by-bias map at once and returns a
global vector `J1`, `J2`, .... Its artifact therefore records an exact chain
length: an `L=6` model accepts six experimental site files, while another
length requires a model trained for that length.

The raw inspection and channel mapping are the same for both variants. Choosing
the variant does not alter the measured data; it selects the inference view and
the model/dataset contracts that the advisor searches. It is therefore safe to
inspect first and decide between homogeneous and inhomogeneous before model
selection. To change that decision after viewing the first report, reuse the
canonical data without parsing the instrument files again:

```bash
hamlet select-experiment-mode experiments/my_chain/experiment_manifest.json \
  --mode heisenberg --variant homogeneous \
  --output-dir experiments/my_chain_homogeneous
```

The new manifest records the source manifest and `data_reimported: false`.

QPI is not registered yet. Its maps, axes, masks, targets, Fourier conventions,
and reports need a separate vertical slice before `--mode qpi` can be honest.

## Generated experiment project

```text
experiments/my_chain/
├── spectroscopy.csv
├── measurement.npz
├── import_report.json
├── import_preview.html
├── experiment_manifest.json
└── experiment_inspection.html
```

`experiment_manifest.json` is the main workflow input. It records schema
version 1, mode/variant, system identifier, supervised view, channel roles,
structural checks, warnings, per-site descriptive statistics, cutoff coverage,
and all output paths. `measurement.npz` is the richer canonical measurement.
The CSV is retained only for interoperability with older tools.

The inspection HTML embeds maps of the primary dI/dV channel and auxiliary
channels such as the second derivative. It clearly labels dI/dV as the
inference input and the second derivative as plotting/QC only.

## What inspection does—and does not do

Inspection verifies:

- the axes and meV convention required by the selected mode;
- primary-channel identity and completeness;
- minimum site count;
- a finite, strictly increasing bias grid spanning zero;
- site-level finite fraction, range, median, and standard deviation;
- whether each requested cutoff has complete common 0-to-cutoff coverage.

Covered cutoffs are choices, not recommendations. The experimentalist views
the raw signals and selects one common physically usable value. The manifest
therefore stores `selected_cutoff_mev: null`.

Inspection deliberately does not crop, subtract a baseline, normalize, clip,
or resample onto a model grid. Those transformations belong to the selected
artifact's preprocessing contract and occur only after compatibility is
established.

## Continue without handling the intermediate CSV

After choosing the cutoff:

```bash
hamlet advise experiments/my_chain/experiment_manifest.json \
  --cutoff 50 \
  --artifact-root models/heisenberg \
  --dataset data/heisenberg_simulations.npz \
  --output-dir results/my_chain_preflight
```

The mode, variant, system, and view are read from the manifest. Supplying a
conflicting `--system` or `--view` is an error.

A compatible artifact can then analyze the manifest directly:

```bash
hamlet-analyze experiments/my_chain/experiment_manifest.json \
  --artifact models/heisenberg/model-cut50 \
  --cutoff 50 \
  --output-dir results/my_chain_cut50
```

Project YAML may likewise use:

```yaml
config_schema_version: 1
name: My Heisenberg chain
system_type: inhomogeneous_heisenberg
artifact: models/heisenberg/model-cut50
experiment:
  manifest: experiments/my_chain/experiment_manifest.json
output_dir: results/my_chain_cut50
training:
  cutoffs_mev: [50]
  manual_cutoff_mev: 50
  view: local_bonds
```

Legacy `experiment.csv` and canonical `measurement.npz` inputs remain
supported. The guided manifest is preferred because it preserves the selected
physics mode and inspection provenance.
