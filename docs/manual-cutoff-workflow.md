# Manual-cutoff reuse or retraining workflow

This is the recommended entry point for the limited inhomogeneous-Heisenberg
package. The experimentalist chooses the physically usable cutoff; the package
does not replace that scientific judgment with a model prediction.

## 1. Import and inspect the experiment

Choose the physics profile while converting one file per site:

```bash
hamlet inspect-experiment my_import.yaml \
  --mode heisenberg --variant inhomogeneous \
  --output-dir experiments/my_chain
```

Open `experiment_inspection.html`, inspect the raw dI/dV and optional second
derivative, confirm site order, and choose a common cutoff that every site
covers reliably. The cutoff is a model
contract: weights trained at 40 meV are different from weights trained at
50 meV even when the architecture and input tensor size are unchanged.

## 2. Ask for a preflight decision

```bash
hamlet advise experiments/my_chain/experiment_manifest.json \
  --cutoff 50 \
  --artifact-root models/heisenberg_cutoff_banks \
  --dataset data/heisenberg_simulations.npz \
  --max-test-mae 2.0 \
  --output-dir results/preflight-cut50
```

`--artifact-root` and `--dataset` are repeatable. An artifact root may be one
artifact, a cutoff bank, or a directory containing several banks. The command
creates both `workflow_decision.json` and `workflow_decision.html`.

The decision is one of four actions:

| Action | Meaning |
|---|---|
| `use_existing_model` | A trained artifact matches the system, learning view, exact cutoff, observable, chain-length rule, and requested error limits. |
| `retrain_with_existing_dataset` | No usable exact-cutoff model exists, but a listed raw simulation dataset covers the system and cutoff. |
| `generate_dataset_and_retrain` | Neither a model nor a compatible dataset is available. |
| `fix_experiment_or_choose_lower_cutoff` | The experiment itself does not cover the selected window or is structurally invalid; retraining cannot fix it. |

## What “use existing model” checks

Automatic reuse requires all of the following:

- exact system and supervised view;
- weights trained at exactly the manual cutoff;
- complete experimental coverage of the artifact preprocessing window;
- a loadable artifact and supported schema;
- finite stored validation and held-out test MAE;
- a standard/research artifact rather than a `quick` development run.

Validation/test MAE acceptance is physics-dependent, so the package does not
invent one universal number. Use `--max-validation-mae` and/or
`--max-test-mae` when your analysis has a defined error budget. Without them,
the stored values are displayed as an explicit warning rather than silently
treated as scientifically sufficient.

For local inhomogeneous models, the three-site sliding-window architecture can
analyze any chain with at least three sites. For global homogeneous models,
the experimental chain length must exactly match the length used for training.

These checks are deliberately performed before model predictions are used.
Passing them means the stored model is technically suitable for inference; it
does not prove the simulator is complete or turn ensemble disagreement into a
confidence interval.

Use `--allow-development-artifacts` only for pipeline testing. It must not be a
default in an experimental analysis.

## 3A. Reuse the selected artifact

The JSON report records `selected_artifact`. Analyze with that exact path:

```bash
hamlet-analyze experiments/my_chain/experiment_manifest.json \
  --artifact models/.../selected-artifact \
  --cutoff 50 \
  --output-dir results/analysis-cut50
```

Alternatively, place the artifact and the exact manual cutoff in a project
configuration. `hamlet run` performs the preflight again before existing-
artifact inference:

```yaml
config_schema_version: 1
artifact: models/heisenberg_local_mlp_cut50_standard_v1
experiment:
  manifest: experiments/my_chain/experiment_manifest.json
training:
  cutoffs_mev: [50]
  manual_cutoff_mev: 50
```

The project refuses an artifact trained at another cutoff.

## 3B. Retrain from an existing dataset

When the decision is `retrain_with_existing_dataset`, use the selected portable
dataset in a project configuration. Training uses the exact manual cutoff
selected for the experiment:

```yaml
config_schema_version: 1
dataset:
  format: portable
  path: data/heisenberg_simulations.npz
experiment:
  manifest: experiments/my_chain/experiment_manifest.json
training:
  cutoffs_mev: [50]
  manual_cutoff_mev: 50
  model: keras_mlp
  preset: standard
```

Validate the resulting model on held-out simulated chains before registering
it for reuse.

## 3C. Generate and retrain

`generate_dataset_and_retrain` means the package has no listed raw simulation
dataset with the correct system, observable, target structure, and bias range.
Start from `examples/heisenberg_generate_train_analyze.yaml`. Generation is
seeded, checkpointed, resumable, and protected by an exact recipe fingerprint.

For each cutoff, train fresh weights. A future bank may reuse the architecture
choice, but never the weights from a different cutoff.

## Python interface

```python
from hamlet import advise_experiment

decision = advise_experiment(
    "experiments/my_chain/experiment_manifest.json",
    manual_cutoff_mev=50,
    artifact_roots=["models/heisenberg"],
    dataset_paths=["data/heisenberg_raw.npz"],
)
print(decision.action)
print(decision.next_steps)
```

The JSON form is intended to remain stable enough for a later graphical user
interface. The future UI should display this decision rather than implement a
second independent compatibility algorithm.
