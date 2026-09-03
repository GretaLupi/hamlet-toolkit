# Bias-cutoff model banks

A cutoff bank uses one selected architecture but independently trained weights
for every bias cutoff. The models are not fine-tuned from one another. Each
cutoff changes the physical meaning of the 200 input samples, so sharing
weights would be scientifically unjustified even though the tensor shape is
unchanged.

## Train from the original Heisenberg datasets

```bash
hamlet-train-cutoff-bank \
  --reference-npz \
    reference/Inhomogeneous-Heisenberg-HL/data/synthetic/dataset_30_40.npz \
    reference/Inhomogeneous-Heisenberg-HL/data/synthetic/dataset_35_45.npz \
  --output-dir models/heisenberg_cutoff_bank_research \
  --cutoffs 30 40 50 70 100 \
  --comparison-cutoff 50 \
  --candidates keras_mlp keras_cnn \
  --preset research
```

The command performs three controlled stages:

1. Train a fresh MLP and CNN at 50 meV using the same grouped split.
2. Select the architecture with the lowest validation MAE in meV.
3. Initialize and train fresh weights with that architecture at every other
   cutoff. The winning 50 meV run is retained as the 50 meV bank member.

By default, broadening, noise, offset, and drift are applied once to the broad
raw simulations using augmentation seed 42. Thus all cutoffs see the same
physical examples and perturbation realization. Use `--theory` only for an
ideal-spectra benchmark.

`quick` is a one-seed pipeline check, `standard` is a three-seed working
ensemble, and `research` is the five-seed final benchmark. Only research runs
that pass the artifact acceptance checklist should be distributed as reference
models.

## Exact-cutoff model selection for an experiment

```bash
hamlet-analyze experiments/my_chain/experiment_manifest.json \
  --artifact-bank models/heisenberg_cutoff_bank_research \
  --cutoff 50 \
  --output-dir analysis-result
```

`--cutoff` is the experimentalist's manual choice and requires independently
trained weights at exactly that value. The package first rejects artifacts whose bias window is not fully covered by
the experimental grid. Among compatible artifacts it defaults to the lowest
stored validation MAE, so a poorly converged wider model is not preferred just
because more bias is available. Programmatic selection is also available:

```python
from hamlet.training import recommend_artifact, TrainingRun

choice = recommend_artifact(
    "models/heisenberg_cutoff_bank_research",
    bias_mev,
    required_cutoff_mev=50,
)
print(choice.reason)
analyzer = TrainingRun.load(choice.artifact_path).create_analyzer()
```

Pass `strategy="largest_covered"` to `recommend_artifact` when deliberately
choosing the widest compatible window regardless of validation performance.
The supplied bias grid must represent the experimentally *usable* range, not
merely every recorded point.
