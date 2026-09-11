<p align="center">
  <img src="https://raw.githubusercontent.com/GretaLupi/hamlet-toolkit/main/assets/logos/hamlet-logo.png" alt="HamLeT — Hamiltonian Learning Toolkit" width="390">
</p>

# HamLeT

[![PyPI](https://img.shields.io/pypi/v/hamlet-toolkit.svg)](https://pypi.org/project/hamlet-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/hamlet-toolkit.svg)](https://pypi.org/project/hamlet-toolkit/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-087f8c.svg)](LICENSE)

HamLeT (Hamiltonian Learning Toolkit) is an experimental-facing framework for
inferring Hamiltonians and physical parameters from scanning tunnelling
microscopy and spectroscopy data. It connects raw-data inspection and
preprocessing to simulated dataset generation, supervised model training,
model matching, inference, and reproducible reports.

Spin chains are HamLeT's first fully implemented application. Its modular
experiment → simulation → learning → inference workflow is designed for
Hamiltonian-learning problems based on STM/STS measurements.

HamLeT **0.1.0 is a complete, working toolkit for spin-chain Hamiltonian
inference**. Its spin-chain workflows are implemented end to end and extensively
tested: experimentalists can import site-resolved STM/STS measurements, inspect
and crop spectra, match compatible pretrained models, generate simulated
training data, train their own estimators, infer couplings, and export results.
Physical energies are expressed in meV; at the DMRGPy simulator boundary,
`1 DMRGPy energy unit = 10 meV`.

## How it works, in one picture

<p align="center">
  <img src="https://raw.githubusercontent.com/GretaLupi/hamlet-toolkit/main/assets/figures/pipeline.png" alt="Simulate, measure, infer: a dynamical correlator becomes per-site dI/dV, and a model reads the couplings back out" width="100%">
</p>

**Simulate.** For a spin chain whose couplings you choose, DMRG or exact
diagonalisation gives the site-resolved dynamical correlator — the left panel
is one such chain, sites up the axis and bias across it.

**Measure.** That correlator is what an STM sees as d*I*/d*V*: one spectrum per
site, each step marking an excitation. This is the form your own data arrives
in, and the form the simulated dataset is built to match.

**Infer.** A model trained on thousands of such simulated chains reads the
couplings back out of a single measured spectrum. The right panel recovers
J1<sub>xy</sub>, J2, J3 and Jz for a chain the model never saw during
training — grey is the truth, teal what it returned.

Every number in that figure is produced by this package; regenerate it with
`python scripts/make_readme_figures.py`.

## Start with the interface

```bash
python -m pip install hamlet-toolkit
hamlet gui
```

The local browser interface guides the full workflow:

1. select a prepared measurement or a folder containing one Nanonis
   `.dat`/`.txt` spectrum per site;
2. inspect every site and choose the visible bias cutoff;
3. check whether a shipped or locally trained model matches the system, chain
   length, view, cutoff, and preprocessing contract;
4. reuse it, or generate a dataset and train ridge, random-forest, MLP, or CNN
   models from the interface;
5. infer the couplings and export a LaTeX/PDF summary for a colleague,
   an HTML report, a plot, a CSV, and a JSON result.

It binds to localhost and does not upload measurements elsewhere. Long jobs run
in the background. Close it with the **Stop server** button or `Ctrl+C`.

The *Start here* page lists every folder it writes to, and `hamlet where`
prints the same layout from the terminal: `results/` beside a source checkout,
`~/.hamlet/workspace/` for an installed package, or `HAMLET_WORKSPACE`.

Dataset generation is the stage that costs hours, and every chain is
independent, so set **chains at once** (`dataset.generate.workers`, or `0` for
one per core) to shorten it. The result does not depend on it — the dataset is
bit-identical however many run at a time. See
[Making generation faster](docs/user-guide.md#making-generation-faster),
which also covers why a GPU makes this stage *slower*.

## Installation options

The base installation includes the interface, experimental I/O, plotting,
ridge and random-forest models, and the published reference-model catalog.

```bash
python -m pip install "hamlet-toolkit[ml]"          # Keras MLP and CNN
python -m pip install "hamlet-toolkit[simulation]"  # DMRGPy generation
python -m pip install "hamlet-toolkit[tune]"        # Optuna search
python -m pip install "hamlet-toolkit[all]"         # all optional features
python -m pip install "hamlet-toolkit[gpu]"         # CUDA TensorFlow, Linux only
```

`[gpu]` is not part of `[all]`, and is worth reading about before installing:
the CUDA wheels are several gigabytes, only the Keras models can use a card,
and dataset generation — the long stage — is CPU-bound and gains nothing from
one. `hamlet compute` reports what this machine will actually use for each
stage, and why. See [Using a GPU](docs/user-guide.md#using-a-gpu).

For development from a clone:

```bash
python -m pip install -e ".[dev,ml,simulation,tune]"
pytest
```

## Supported workflows

| Workflow | Model view | Current status |
| --- | --- | --- |
| Bond-inhomogeneous Heisenberg | three-site local windows; variable chain length | validated workflow; published Keras ensemble |
| Homogeneous Heisenberg `J1-J2` | complete chain; exactly L=8 | published random-forest reference model; simulation-validated |
| XXZ `J1-J2-J3` with impurity-assisted DMI | complete chain; exactly L=8 | published ridge reference model; experimental-design workflow |

Global models are chain-length specific. Local sliding-window models can be
applied to other lengths when the stored model contract is otherwise
compatible. HamLeT reports a mismatch instead of padding spectra or silently
changing the cutoff.

Each shipped artifact has a model card under
[`src/hamlet/resources/models`](src/hamlet/resources/models) documenting its
training distribution, expected observable, accuracy, and limitations.

The pretrained bond-inhomogeneous model is the model developed for:

> G. Lupi, S. Ravuri, C. Zhao, et al., *Learning Inhomogeneous Heisenberg
> Hamiltonians in Nanographene Spin Chains* (2026).

Please cite that work when using this model. Its
[model card](src/hamlet/resources/models/inhomogeneous_heisenberg_l12_keras_mlp_standard_v1/MODEL_CARD.md)
contains the full author list, BibTeX entry, training provenance, and validity
range.

### Work this package builds on

The uniform spin-chain Hamiltonians and the impurity-tomography route to
parameters that are otherwise hidden come from:

> G. Lupi and J. L. Lado, *Hamiltonian-learning quantum magnets with nonlocal
> impurity tomography*, [Phys. Rev. Applied **23**, 054077
> (2025)](https://doi.org/10.1103/PhysRevApplied.23.054077).

> N. Karjalainen, G. Lupi, R. Koch, A. O. Fumega and J. L. Lado, *Hamiltonian
> learning quantum magnets with dynamical impurity tomography*, [Phys. Rev.
> Research **8**, 033281 (2026)](https://doi.org/10.1103/cw27-2qqd).

**The models trained for those papers are not distributed with this package.**
HamLeT implements the same families of Hamiltonians and the same measurement
logic, so you can generate a dataset and train an equivalent model yourself,
but any pretrained artifact shipped here is listed under
[`src/hamlet/resources/models`](src/hamlet/resources/models) and is not one of
theirs. Please cite these papers if you use the impurity route or the uniform
chain families; the estimators you train are your own.

## Command line

The same workflow can be scripted with YAML configurations:

```bash
hamlet modes
hamlet inspect-experiment path/to/measurement.csv
hamlet run examples/quickstart_l8.yaml --dry-run
hamlet run examples/quickstart_l8.yaml
```

See the [user guide](docs/user-guide.md) for raw-file import, cutoff selection,
dataset generation, model tuning, cluster execution, inference, and reports.
The small [L=8 notebook](notebooks/04_l8_three_mode_workflow.ipynb) is the
reproducible Python example; notebooks are tutorials, not the test suite.

## Scientific scope

Reference-model scores describe held-out simulated data unless a model card
explicitly says otherwise. Ensemble spread measures disagreement between
trained members; it is not a calibrated confidence interval. A measurement
must satisfy the artifact contract before its predictions are physically
interpretable.

Uniform z-directed DMI cannot be identified from the supported on-site
autocorrelator in a symmetry-preserving chain. HamLeT therefore exposes the
impurity-assisted design route instead of offering a misleading ordinary DMI
model. See the [DMI experiment specification](docs/dmi-experiment-spec.md).

## Project information

- Changes: [CHANGELOG.md](CHANGELOG.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Releasing: [RELEASING.md](RELEASING.md)
- Security reports: [SECURITY.md](SECURITY.md)
- Citation metadata: [CITATION.cff](CITATION.cff)

HamLeT is distributed under the [GNU General Public License v3.0 or
later](LICENSE).
