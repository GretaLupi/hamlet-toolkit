<p align="center">
  <img src="assets/logos/hamlet-logo.png" alt="HamLeT — Hamiltonian Learning Toolkit" width="390">
</p>

# HamLeT

[![CI](https://github.com/GretaLupi/hamlet-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/GretaLupi/hamlet-toolkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hamlet-toolkit.svg)](https://pypi.org/project/hamlet-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/hamlet-toolkit.svg)](https://pypi.org/project/hamlet-toolkit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-087f8c.svg)](LICENSE)

HamLeT (Hamiltonian Learning Toolkit) is an experimentalist-facing Python
package for generating spin-chain spectroscopy datasets, training supervised
inverse models, and inferring Hamiltonian parameters from measured dI/dV maps.

The package is an **alpha release**: the complete Heisenberg workflow works,
while additional physical systems and experimental formats are still being
added. Physical energies are expressed in meV; at the simulator boundary,
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
