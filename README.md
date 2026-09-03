<p align="center">
  <img src="assets/logos/hamlet-logo.png" alt="HamLeT — Hamiltonian Learning Toolkit" width="360">
</p>

# HamLeT

**Hamiltonian Learning Toolkit**

> "Though this be madness, yet there is method in't." — Hamlet, Act 2.


## Fast validation

Before extending HamLeT with another physical system, run the small end-to-end
release gate:

```bash
./scripts/run_smoke_tests.sh
```

It generates homogeneous and inhomogeneous Heisenberg datasets at several
chain lengths, checks manual-cutoff preprocessing, trains the fast classical
models, round-trips artifacts, and exercises local experimental inference.
Neural-model shape checks run when TensorFlow/Keras is installed. The complete
test suite remains `python -m pytest`.

HamLeT is an experimentalist-facing Python toolkit for generating spectroscopy datasets,
training inverse models, benchmarking them, and reconstructing physical
Hamiltonians from measurements.

The installable distribution is `hamlet-toolkit`, the canonical Python import
is `hamlet`, and the canonical command is `hamlet`. The bare `hamlet`
distribution name is already used by an unrelated project on
[PyPI](https://pypi.org/project/hamlet/), so HamLeT uses a unique distribution
name.

The simplest interface is one configuration and one command:

```bash
hamlet run examples/heisenberg_train_and_analyze.yaml
```

This calibrates the manually selected cutoff against the experiment, freezes
its preprocessing contract, trains and validates the requested model, performs
inference, and exports a self-contained HTML analysis report. See
[docs/user-guide.md](docs/user-guide.md) for the complete workflow, from raw
per-site files through retraining to the final report.

New simulation datasets can also be generated directly from YAML:

```bash
hamlet generate examples/heisenberg_generate_dataset.yaml
```

Generation uses physical meV inputs, deterministic seeds, exact-recipe cache
validation, and chunk checkpoints for safe continuation after interruption.
`hamlet run` automatically performs this generation stage when its project
configuration contains `dataset.generate`.

Experimental spectra can start as one text/DAT file per measured site. A
configurable recipe first makes the laboratory data inspectable; the selected
mode then says how HamLeT will interpret the same inspected site map:

```bash
hamlet inspect-experiment my_import.yaml \
  --mode heisenberg --variant inhomogeneous \
  --output-dir experiments/my_chain
```

It validates the laboratory import plus mode-specific axes, site count,
channels, bias coverage, and cutoff choices, then creates a versioned
experiment manifest and visual HTML inspection. dI/dV is the primary model
channel; optional signals such as a second derivative remain available for
plotting and QC. No model normalization or cropping happens at this stage. See
[docs/user-guide.md](docs/user-guide.md) and
[examples/import_nanonis_like_dat.yaml](examples/import_nanonis_like_dat.yaml).

The interpretation can be changed after looking at the inspection without
re-importing the raw files:

```bash
hamlet select-experiment-mode experiments/my_chain/experiment_manifest.json \
  --mode heisenberg --variant homogeneous \
  --output-dir experiments/my_chain_homogeneous
```

The current suite contains 125 passing tests, covering engineering,
importer, and identifiable known-Hamiltonian recovery gates plus a small
DMRGPy recovery gate.

After inspecting an experiment and choosing its cutoff manually, ask the
package whether to reuse, retrain, regenerate, or revise the measurement:

```bash
hamlet advise experiments/my_chain/experiment_manifest.json --cutoff 50 \
  --artifact-root models/heisenberg --dataset data/heisenberg.npz
```

The decision checks the system, learning view, exact cutoff, observable,
artifact metrics, and chain-length rule before recommending an existing model.
See [docs/user-guide.md](docs/user-guide.md) for the full reuse/retrain/
regenerate decision table.

The supported Heisenberg slice now has two inference views. The
bond-inhomogeneous spin-1/2 chain was developed in
[Inhomogeneous-Heisenberg-HL](https://github.com/GretaLupi/Inhomogeneous-Heisenberg-HL).
Its local model consumes spectra from three adjacent sites and predicts the two
enclosed exchange couplings. Sliding this fixed-size estimator over a chain
makes inference independent of total chain length.

For a homogeneous chain, HamLeT instead sends the entire fixed-length map to a
global model and returns named parameters such as `J1`, `J2`, and `J3`. A
global artifact is reusable only when its trained chain length and cutoff match
the experiment. Thus the six supplied site spectra are one `L=6` input, not
four local sliding windows.

An additional development mode, `heisenberg/xxz_long_range`, defines the
four-parameter `J1_xy, J2, J3, Jz` Hamiltonian explicitly. Its first genuine
L=8 DMRGPy pilot is retained under `results/xxz_j1j2j3_l8_pilot`; it identifies
J3 as the weakest target and is not a release-quality experimental artifact.

Simulation observables are explicit model contracts. `Sz` preserves the
original workflow; `total_spin` produces a configurable weighted sum of
`Sxx + Syy + Szz` and is recommended for new anisotropic/DMI studies. The
advisor rejects silent observable substitutions.

A paired L=8 exact-diagonalization screen shows why both observables remain:
total spin improved J2/J3, while Sz improved J1_xy/Jz. A separate five-target
total-spin DMI screen found that J3 and DMI magnitude were not yet reliably
identifiable. Both artifacts remain development-only.

Physical energy inputs and outputs use meV. The DMRGPy adapter applies the
project convention `1 DMRGPy energy unit = 10 meV` internally.

## Current API

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

Install the lightweight core for development with:

```bash
python -m pip install -e '.[dev,io]'
pytest
```

The original source-only research repository is retained under `reference/` for
behavioral comparisons during migration. Large datasets and trained models are
not duplicated here.

Dataset generation supports both bond-inhomogeneous chains and homogeneous
global `J1-J2-...` chains through the same simulator-independent interface.
A generated dataset converts to either a three-site local-bond view or a
full-chain global view, then trains `keras_mlp`, `keras_cnn`, `ridge`, or
`random_forest`. The bias cutoff is selected from the experiment and stored
as part of the training/inference contract. See
[docs/user-guide.md](docs/user-guide.md) for the complete workflow.

Experimentalists can train their own leakage-safe single models or
multi-seed ensembles with `quick`, `standard`, and `research` presets. The
saved artifact bundles the trained models with cutoff, preprocessing, target
scaling, units, seeds, and physical-unit metrics, and opens directly as an
experimental analyzer.

Experimentalists can run a trained local ensemble from Python with
`ExperimentalChainAnalyzer` or use the `hamlet-analyze` command. Both paths
produce bond estimates, ensemble uncertainty, local-window consistency checks,
CSV/JSON exports, a quality-control figure, and a self-contained HTML report.

Multi-seed training automatically chooses mean, median, or validation-weighted
aggregation on grouped synthetic validation data, freezes that rule in the
artifact, and leaves raw member disagreement visible during experiment checks.

For an executed end-to-end demonstration with saved plots and outputs, open
[notebooks/01_supervised_workflows.ipynb](notebooks/01_supervised_workflows.ipynb).

For the diagnostics-first experimental workflow using the real
`ChainIH_L6_SYM.csv` data and three enhanced research models, open
[notebooks/02_full_potential_experimental_interface.ipynb](notebooks/02_full_potential_experimental_interface.ipynb).

The executed simulation-to-experiment research study—including spectral
distribution scoring, nearest synthetic windows, inference diagnostics, and DMRGPy forward residuals—
is [notebooks/03_simulation_experiment_validation.ipynb](notebooks/03_simulation_experiment_validation.ipynb).

Label-free nuisance calibration is available from the command line:

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

The optional calibration command uses measured spectral compatibility only; it
never uses experimental Hamiltonian labels or inverse-model predictions. It is
an advanced research utility and is not required for ordinary model reuse or
inference.
