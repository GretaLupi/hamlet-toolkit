<img src="../assets/logos/hamlet-icon.png" alt="" width="40" align="left">

# Using HamLeT

This is the complete path from raw per-site spectroscopy files to an inferred
Hamiltonian: import your experiment, choose a physically usable cutoff, let
the package decide whether to reuse, retrain, or generate a model, then read
the report. See the [README](../README.md) for install instructions.

## 1. Import and inspect your experiment

Experiments start as one text/DAT file per measured site, described by a YAML
recipe (see [`examples/import_nanonis_like_dat.yaml`](../examples/import_nanonis_like_dat.yaml)).
Copy and edit it with your input path and exact column names, then run:

```bash
hamlet modes
```

`hamlet modes` lists the registered physical interpretations:

| Mode | Variant | System identifier | View | Simulated observable |
|---|---|---|---|---|
| `heisenberg` | `inhomogeneous` | `inhomogeneous_heisenberg` | `local_bonds` | `Sz` (historical approximation) |
| `heisenberg` | `homogeneous` | `homogeneous_heisenberg` | `global` | `Sz` by default |

(Two additional development-only variants, `xxz_long_range` and `xxz_dmi`, are
listed too but are not yet release-quality — see the README.)

Then inspect the raw signals under the chosen mode:

```bash
hamlet inspect-experiment my_import.yaml \
  --mode heisenberg --variant inhomogeneous \
  --output-dir experiments/my_chain \
  --candidate-cutoffs 30 40 50 70 100
```

This creates:

```text
experiments/my_chain/
├── spectroscopy.csv           # stable long-form CSV, kept for interoperability
├── measurement.npz            # canonical measurement: axes, channels, units, masks
├── import_report.json         # file/site/column assignment, missing values, axis reversals
├── import_preview.html
├── experiment_manifest.json   # the main workflow input from here on
└── experiment_inspection.html # visual dI/dV (+ optional second derivative) per site
```

Open `experiment_inspection.html` and confirm site order, channel mapping, and
common energy coverage. dI/dV is the primary model channel; a second
derivative or similar signal is retained for plotting/QC only, never as a
model input. **No cropping, baseline subtraction, or normalization happens at
this stage** — that belongs to a specific model's preprocessing contract,
applied later.

Changed your mind about inhomogeneous vs. homogeneous after seeing the report?
Re-interpret without re-parsing the raw files:

```bash
hamlet select-experiment-mode experiments/my_chain/experiment_manifest.json \
  --mode heisenberg --variant homogeneous \
  --output-dir experiments/my_chain_homogeneous
```

## 2. Choose a cutoff and ask the advisor

The bias cutoff — how much of the measured energy range to use — is a
**scientific judgment call you make**, not something the package infers. Pick
the largest value every site covers reliably, then ask whether an existing
model, an existing dataset, or a fresh simulation run is needed:

```bash
hamlet advise experiments/my_chain/experiment_manifest.json \
  --cutoff 50 \
  --artifact-root models/heisenberg \
  --dataset data/heisenberg_simulations.npz \
  --max-test-mae 2.0 \
  --output-dir results/preflight-cut50
```

`--artifact-root` and `--dataset` are repeatable — point them at whatever
model banks or raw datasets you have on hand. The result is one of four
actions, written to `workflow_decision.json`/`.html`:

| Action | Meaning |
|---|---|
| `use_existing_model` | A trained artifact matches the system, view, exact cutoff, observable, chain-length rule, and your error limits. |
| `retrain_with_existing_dataset` | No usable exact-cutoff model exists, but a listed raw dataset covers the system and cutoff. |
| `generate_dataset_and_retrain` | Neither a model nor a compatible dataset is available. |
| `fix_experiment_or_choose_lower_cutoff` | The experiment itself doesn't cover the selected window; retraining cannot fix that. |

Weights trained at one cutoff are **never** silently substituted for another
— even with the same architecture, the physical meaning of the input changes.
Reuse always requires an exact cutoff match, plus finite stored
validation/test MAE and a standard/research (not `quick`, development-only)
training preset.

## 3. Run the workflow

### Check first: what would this run actually do?

Simulation is the expensive part of this workflow, so look before you leap:

```bash
hamlet run project.yaml --dry-run
```

This writes nothing. It reports the stages that would execute, how many chains
would be simulated and roughly what that costs, every file that would be
written, and — importantly — which existing files would make the run **refuse**
partway through, which is otherwise only discovered after the expensive part
has already happened. It exits non-zero when the real run would be refused, so
it works as a precondition check in a script.

The cost figure is an order-of-magnitude anchor from this project's own L=8
exact-diagonalisation runs, scaled by how many correlators the observable
needs (`total_spin` evaluates `Sxx`, `Syy` and `Szz`, so about three times
`Sz`). Once you have measured your own rate, pass it:

```bash
hamlet run project.yaml --dry-run --seconds-per-chain 25 --plan-json plan.json
```

`--plan-json` writes the same information as a versioned machine-readable
plan, alongside the workflow-decision JSON as something a future interface can
render directly.

### The default path: one config, one command

```bash
hamlet run project.yaml
```

A project configuration can describe everything: where to get training data,
which experiment to analyze, and how to train. For example, generating fresh
simulations and analyzing an experiment in one shot:

```yaml
config_schema_version: 1
name: Inhomogeneous Heisenberg experimental analysis
system_type: inhomogeneous_heisenberg

dataset:
  generate:
    system: inhomogeneous_heisenberg
    output: data/heisenberg_L12_3000.npz
    n_sites: 12
    n_samples: 3000
    coupling_range_mev: [30, 45]
    bias_range_mev: [0, 100]
    bias_points: 200
    broadening_mev: 0.5
    observable: Sz
    output_quantity: didv
    backend: dmrgpy
    seed: 42

experiment:
  manifest: experiments/my_chain/experiment_manifest.json

output_dir: analysis-output

training:
  cutoffs_mev: [40, 50, 70]
  manual_cutoff_mev: 50
  output_points: 200
  view: local_bonds
  model: keras_mlp
  preset: standard
```

`hamlet run` then: generates or safely reuses the configured dataset,
calibrates the manually selected cutoff (label-free, no Hamiltonian labels or
model predictions involved), stops if that cutoff fails compatibility, trains
a leakage-safe model ensemble, selects an aggregation rule on synthetic
validation chains only, evaluates the untouched test split, and analyzes the
experiment into machine-readable and HTML outputs.

Generation is restart-safe (chunk checkpoints, exact-recipe fingerprinting —
a changed recipe at the same output path is rejected, not silently
overwritten), and existing non-empty artifact/output directories are never
overwritten. Inspect first with `hamlet inspect project.yaml`, which reports
site count, energy coverage, and cutoff candidates without training anything.

More complete configurations: [`examples/heisenberg_generate_dataset.yaml`](../examples/heisenberg_generate_dataset.yaml),
[`examples/heisenberg_generate_train_analyze.yaml`](../examples/heisenberg_generate_train_analyze.yaml),
[`examples/heisenberg_train_and_analyze.yaml`](../examples/heisenberg_train_and_analyze.yaml).

### Retraining directly

When the advisor says `retrain_with_existing_dataset` or
`generate_dataset_and_retrain`, training can also be driven directly from
Python:

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
    TrainingPreprocessingConfig(bias_cutoff_mev=50.0, output_points=200),
)

run = train_supervised(
    prepared,
    view="local_bonds",          # or "global" for homogeneous chains
    model="keras_mlp",           # also: keras_cnn, ridge, random_forest
    preset="standard",           # quick, standard, or research
)

print(run.metrics["validation"])
print(run.metrics["test"])
run.save("artifacts/heisenberg-local-mlp-cut50-v1")
```

| Preset | Seeds | Max epochs | Early-stopping patience | Intended use |
|---|---:|---:|---:|---|
| `quick` | 1 | 20 | 4 | API check and debugging — development-only, not for real inference |
| `standard` | 3 | 100 | 12 | Normal experimental analysis |
| `research` | 5 | 250 | 25 | Final benchmark |

Multi-seed runs compare mean/median/inverse-validation-MAE-weighted
aggregation on synthetic validation chains only; the winning rule is frozen
into the artifact and reused automatically at inference. Raw per-seed spread
is still reported alongside it — robust aggregation never hides ensemble
disagreement.

### Reusing an existing artifact for a follow-up experiment

Skip training entirely once you have a matching artifact:

```yaml
config_schema_version: 1
name: Follow-up chain
system_type: inhomogeneous_heisenberg
artifact: models/heisenberg_local_mlp_cut50_standard_v1
experiment:
  manifest: experiments/new_chain/experiment_manifest.json
output_dir: follow-up-analysis
training:
  cutoffs_mev: [50]
  manual_cutoff_mev: 50
```

`hamlet run` re-verifies system, view, exact cutoff, observable, artifact
metrics, and chain-length rule before using it — the same checks `hamlet
advise` performs. See [`examples/heisenberg_existing_artifact.yaml`](../examples/heisenberg_existing_artifact.yaml).

## 4. Read the report

```text
analysis-output/
├── resolved_project_config.json
├── experiment_inspection.json
├── preflight/                 # present for existing-artifact inference
│   ├── workflow_decision.json
│   └── workflow_decision.html
├── calibration/
│   ├── cutoff-50.json
│   └── summary.json
├── artifact/
│   ├── manifest.json
│   ├── model_seed_*.keras
│   └── training_distribution.npz
├── analysis/
│   ├── couplings.csv
│   ├── report.json
│   ├── report.html
│   └── summary.png
└── project_summary.json
```

`analysis/report.html` is self-contained (the QC plot is embedded), so it can
be opened locally or shared without a running Python server. It records
coupling estimates, per-seed ensemble spread, overlapping-window consistency
checks, out-of-distribution warnings, units, preprocessing, model metrics,
and artifact provenance.

**Ensemble spread and overlap disagreement are diagnostics, not a calibrated
confidence interval** — see the README for the current scientific status and
limitations before treating a result as final.
