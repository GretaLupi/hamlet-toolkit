# Changelog

Notable changes to HamLeT. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
the policy in [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-09-14

First public release. HamLeT infers spin-chain Hamiltonians from scanning
tunnelling spectroscopy by simulating spectra for known couplings, training a
model to invert that map, and applying it to a measurement.

Everything below is what the release contains. The development that produced
it, including the internal 0.1.0 milestone, is in the git history.

### Added

- **A local browser interface, `hamlet gui`.** The whole workflow without
  writing a configuration: choose a measurement, inspect every site and pick
  the bias cutoff, check whether an existing model matches, generate a dataset
  and train, infer couplings, and export the results. Standard library only,
  bound to localhost, with long jobs on background threads that can be stopped
  and resumed.
- **Three spin-chain workflows**, each with a published reference model and a
  model card recording its training distribution, observable, accuracy and
  limits: bond-inhomogeneous Heisenberg (local three-site windows, variable
  length), homogeneous Heisenberg J1-J2 (L=8), and XXZ J1-J2-J3 with
  impurity-assisted DMI (L=8).
- **Experimental import and inspection.** One Nanonis `.dat`/`.txt` per site, a
  CSV of site/bias/dI-dV, or a canonical NPZ; site-resolved plots, and a bias
  cutoff chosen against the data rather than assumed.
- **A reuse advisor.** Before training anything, HamLeT checks a measurement
  against every model it can see -- system, chain length, view, exact cutoff,
  observable, preprocessing -- and reports a mismatch instead of padding
  spectra or silently changing the cutoff.
- **Dataset generation** through DMRGPy, choosing exact diagonalisation or DMRG
  by Hilbert dimension. Checkpointed per chunk, so an interrupted run resumes,
  and fingerprinted by recipe, so an existing dataset is reused only when it
  genuinely matches.
- **Generation runs on many cores.** `dataset.generate.workers`, or *cores to
  use* in the form; measured 2.7x on four. The dataset is bit-identical however
  many run at once, because chunk seeds come from position rather than order.
- **Chains of any spin from S=1/2 to S=5/2**, per site, with impurities that
  differ from the chain and from each other.
- **Four estimators** -- ridge, random forest, Keras MLP, Keras CNN -- with
  quick/standard/research presets, multi-seed ensembles whose aggregation rule
  is selected on validation data and frozen into the artifact, and optional
  hyperparameter search (Optuna when installed, random sampling otherwise)
  that evaluates the library defaults first and keeps them if nothing beats
  them.
- **Reports.** Every analysis writes a LaTeX summary and a PDF beside it when a
  TeX toolchain is present -- the Hamiltonian written out, the method, one
  table of couplings -- along with a self-contained HTML report, a
  quality-control figure, `couplings.csv` and `report.json`.
- **DMI sample design.** Screen impurity arrangements before preparing a
  sample: a free symmetry verdict that rules out hopeless designs without
  simulating, a measured imprint for the rest against a calibration table, and
  a warning when the symmetry count is optimistic because one global rotation
  can undo the impurities.
- **Cluster submission** over your own ssh, with Slurm, PBS, LSF and Grid
  Engine. Key-based access is required and checked, the run directory can be
  browsed rather than typed, and the connection test confirms that the cluster
  can actually import HamLeT before a job is queued.
- **`hamlet compute`**, which reports the cores and accelerators available and
  says which stage each affects -- generation is CPU-bound and cannot use a
  GPU; only the Keras models can.
- **`hamlet where`**, which prints the folders results are written to. They
  differ by install: beside a source checkout, under the home directory for an
  installed package, or wherever `HAMLET_WORKSPACE` points.
- **A command line for everything the interface does**: `modes`,
  `inspect-experiment`, `advise`, `generate`, `run`, `screen-dmi`, `compute`,
  `where`, `cluster`, and `gui`. A configuration written by the form is a file
  the command line re-runs unchanged.
- **Optional extras**: `ml` (TensorFlow), `simulation` (DMRGPy), `tune`
  (Optuna), `gpu` (CUDA TensorFlow on Linux), and `all`.
- **A line of Shakespeare while a job runs.** The package is called HamLeT and
  the waits are long.

### Notes on scientific scope

- Reference-model scores describe held-out simulated data unless a model card
  says otherwise.
- Ensemble spread measures disagreement between trained members. It is not a
  calibrated confidence interval, and a narrow spread is not by itself
  evidence that an estimate is correct.
- A measurement must satisfy an artifact's contract before its predictions are
  physically interpretable. HamLeT refuses rather than extrapolating.
- Uniform z-directed DMI cannot be identified from the supported on-site
  autocorrelator in a symmetry-preserving chain. The impurity-assisted design
  route is offered instead of a model that would look like it worked.
- Physical energies are in meV. At the DMRGPy boundary, 1 DMRGPy energy unit =
  10 meV.

### Requirements

- Python 3.10 to 3.13. Licensed under the GNU General Public License v3.0 or
  later.
