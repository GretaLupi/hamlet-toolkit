# Unified project workflow

`HamiltonianLearningProject` is the high-level interface for experimentalists.
It joins experiment inspection, label-free preprocessing calibration, a
manually selected cutoff, supervised training, artifact creation, inference, and a
self-contained HTML report.

## Generate simulations from configuration

An existing dataset is no longer required. A project may describe its training
simulations directly:

```yaml
config_schema_version: 1
name: Generate Heisenberg simulations
system_type: inhomogeneous_heisenberg
output_dir: analysis-output

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
    max_bond_dimension: 20
    kpm_max_bond_dimension: 20
    seed: 42
    checkpoint_every: 25
```

Generate only the portable NPZ dataset with:

```bash
hamlet generate project.yaml
```

The experiment section is optional for `generate`. For a full project, include
the experiment and training sections and run `hamlet run project.yaml`;
generation then happens automatically before calibration.

Generation is restart-safe. Every completed chunk is saved in a hidden
checkpoint directory. After interruption, an identical recipe reuses complete
chunks and simulates only the missing ones. Once the final dataset is written
atomically, chunk files are removed. A `.generation.json` sidecar remains with
the exact recipe fingerprint. A completed matching dataset is a cache hit; a
different recipe at the same output path is rejected instead of overwritten.

For homogeneous longer-range systems, replace the system and ranges:

```yaml
dataset:
  generate:
    system: homogeneous_heisenberg
    output: data/homogeneous_J1J2.npz
    n_sites: 10
    n_samples: 2000
    coupling_ranges_mev:
      - [30, 45]  # J1
      - [0, 10]   # J2
    bias_range_mev: [0, 100]
    bias_points: 200
    backend: dmrgpy
```

`observable: Sz` retains the original approximation. New anisotropic or DMI
studies should normally declare the total unpolarized response explicitly:

```yaml
    observable: total_spin
    observable_weights: [1, 1, 1]  # Sxx, Syy, Szz
```

This choice is stored in the dataset and artifact. HamLeT does not silently
substitute an `Sz` artifact for an experiment mode expecting `total_spin`.

The generated global dataset, saved artifact, experimental analyzer, and report
are supported. The analyzer consumes the full fixed-length map and reports one
value and ensemble spread for every named global target (`J1`, `J2`, ...).

## One configuration and one command

```yaml
config_schema_version: 1
name: Inhomogeneous Heisenberg experimental analysis
system_type: inhomogeneous_heisenberg

dataset:
  format: reference_heisenberg
  paths:
    - dataset_30_40.npz
    - dataset_35_45.npz

experiment:
  manifest: experiments/chain_ih/experiment_manifest.json

output_dir: analysis-output

training:
  cutoffs_mev: [40, 50, 70]
  manual_cutoff_mev: 50
  output_points: 200
  view: local_bonds
  model: keras_mlp
  preset: standard
  verbose: 1
```

Paths are interpreted relative to the configuration file. Run the full
workflow with:

```bash
hamlet inspect project.yaml
hamlet run project.yaml
```

`inspect` reads only the canonical experiment project and records its site count, energy
coverage, grid size, and configured cutoff candidates. The user inspects this
information and records `manual_cutoff_mev`. `run` then:

1. generates or safely reuses the configured simulation dataset when needed;
2. calibrates the manually selected cutoff without Hamiltonian labels or model predictions;
3. stops if that selected cutoff fails the compatibility gate;
4. applies the selected nuisance model to the simulations;
5. trains the requested leakage-safe model ensemble;
6. selects ensemble aggregation on synthetic validation chains;
7. evaluates the untouched test split;
8. analyzes the experiment and creates machine-readable and HTML outputs.

If the manual cutoff does not pass, training stops and the calibration JSON files remain
available for inspection. Existing nonempty artifact directories are never
silently overwritten. The complete path-resolved configuration is stored as
`resolved_project_config.json`; reusing the output directory with a different
configuration is refused.

## Python workflow

The same stages are available interactively:

```python
from hamlet import HamiltonianLearningProject

project = HamiltonianLearningProject.from_config("project.yaml")
project.inspect_experiment()
project.calibrate_preprocessing()
project.prepare_training_data()
run = project.train()
result = project.infer()

print(project.selected_cutoff_mev)
print(run.aggregation.method)
print(result.coupling_mean)
```

This staged form is useful in a notebook because the user can inspect the
calibration results before authorizing training.

## Analyze with an existing artifact

Training does not have to be repeated for every experiment. Remove the
`dataset` section and specify an artifact:

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

`hamlet run` verifies the physical mode, learning view, exact manual cutoff,
observable, artifact metrics, and chain-length rule. It then loads the stored
preprocessing, target scaling, and aggregation rule for inference.

## Output directory

The complete training workflow produces:

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

`report.html` is self-contained: the quality-control plot is embedded in the
file, so it can be opened locally or shared without a running Python server.
It records estimates, member uncertainty, consistency checks, warnings, units,
preprocessing, model metrics, and artifact provenance.

Complete examples are provided in
[`examples/heisenberg_generate_dataset.yaml`](../examples/heisenberg_generate_dataset.yaml),
[`examples/heisenberg_generate_train_analyze.yaml`](../examples/heisenberg_generate_train_analyze.yaml),
[`examples/heisenberg_train_and_analyze.yaml`](../examples/heisenberg_train_and_analyze.yaml)
and
[`examples/heisenberg_existing_artifact.yaml`](../examples/heisenberg_existing_artifact.yaml).
