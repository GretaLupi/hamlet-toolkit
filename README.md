<p align="center">
  <img src="assets/logos/hamlet-logo.png" alt="HamLeT — Hamiltonian Learning Toolkit" width="390">
</p>

# HamLeT

[![CI](https://github.com/GretaLupi/hamlet-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/GretaLupi/hamlet-toolkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hamlet-toolkit.svg)](https://pypi.org/project/hamlet-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/hamlet-toolkit.svg)](https://pypi.org/project/hamlet-toolkit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-087f8c.svg)](LICENSE)

HamLeT (Hamiltonian Learning Toolkit) is an experimental-facing framework for
inferring Hamiltonians and physical parameters from scanning tunnelling
microscopy and spectroscopy data. It connects raw-data inspection and
preprocessing to simulated dataset generation, supervised model training,
model matching, inference, and reproducible reports.

Spin chains are HamLeT's first fully implemented application. Its modular
experiment → simulation → learning → inference workflow is designed for
Hamiltonian-learning problems based on STM/STS measurements.

HamLeT **1.0 is a complete, working toolkit for spin-chain Hamiltonian
inference**. Its spin-chain workflows are implemented end to end and extensively
tested: experimentalists can import site-resolved STM/STS measurements, inspect
and crop spectra, match compatible pretrained models, generate simulated
training data, train their own estimators, infer couplings, and export results.
Physical energies are expressed in meV; at the DMRGPy simulator boundary,
`1 DMRGPy energy unit = 10 meV`.

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
5. infer the couplings and export an HTML report, plot, CSV, and JSON result.

It binds to localhost and does not upload measurements elsewhere. Long jobs run
in the background. Close it with the **Stop server** button or `Ctrl+C`.

## Installation options

The base installation includes the interface, experimental I/O, plotting,
ridge and random-forest models, and the published reference-model catalog.

```bash
python -m pip install "hamlet-toolkit[ml]"          # Keras MLP and CNN
python -m pip install "hamlet-toolkit[simulation]"  # DMRGPy generation
python -m pip install "hamlet-toolkit[tune]"        # Optuna search
python -m pip install "hamlet-toolkit[all]"         # all optional features
```

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

HamLeT is distributed under the [MIT License](LICENSE).
