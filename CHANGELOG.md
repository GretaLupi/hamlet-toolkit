# Changelog

Notable changes to HamLeT. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
the policy in [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

### Fixed

- **A GPU is no longer invisible because of one library.** TensorFlow's
  `and-cuda` wheels ship `libcusolver` in a directory that is on the RUNPATH of
  `libtensorflow_cc.so.2` but not of `libtensorflow_framework.so.2`, where the
  load actually happens, so TensorFlow reported no GPU at all on machines whose
  driver and CUDA wheels were both fine. HamLeT now loads it into the process
  before importing TensorFlow, which needs no environment variable and no
  action from the user. Every TensorFlow and Keras import in the package goes
  through one helper so a new call site cannot skip it.
- **XLA no longer stops `keras_cnn` training on older cards.** Keras defaults
  `jit_compile` to `"auto"`, which turns XLA on for TensorFlow on a GPU; its
  autotuner then finds no supported configuration for a convolution on
  pre-Volta cards and training dies before the first epoch. These models are
  small enough that XLA wins little, so it is off by default, and
  `jit_compile: true` in a model's options turns it back on.
- **`hamlet compute` trains a convolution on the card instead of trusting the
  device list.** A visible GPU proves the driver loaded and nothing more; a
  cuDNN too new for the card fails only once real work starts, which without
  this check is after the dataset has been generated. When the probe fails the
  command prints what to try, including the cuDNN pin for older GPUs.
- The note explaining a missing GPU no longer suggests reinstalling
  `[gpu]`, which could not have fixed a machine where every wheel was already
  installed. It names the driver, and how to check for one.

All three were found and diagnosed by [@joselado](https://github.com/joselado)
in [#1](https://github.com/GretaLupi/hamlet-toolkit/issues/1), on a GTX 1060.

## [0.1.1] - 2026-09-16

Corrections found while publishing 0.1.0, on the same day.

### Fixed

- **The reuse check no longer assumes which family of Hamiltonian produced a
  measurement.** A measurement declares no family, and the advisor defaulted to
  `inhomogeneous_heisenberg`, then rejected every homogeneous model for a
  "system mismatch" the data had never asserted -- while inference itself ran
  those same models happily. Whether a chain is treated as homogeneous or
  bond-inhomogeneous is a modelling choice, so an undeclared family now
  constrains nothing and each model is judged against the view it was trained
  for. Naming a family, as every project configuration and `--system` does,
  still narrows the search exactly as before.
- **What a model needs of the *sample* can now be declared.** Impurities are
  never recognised from data -- nothing in a dI/dV map reveals them -- and the
  only way to declare them was a checkbox confirming a model's own list, on the
  analysis page. The reuse check had no way at all, so a model trained with
  impurities was refused there every time, with a reason the page gave no means
  of answering. Both pages now carry one declaration of the measured chain: a
  clickable chain diagram with each impurity's spin, axial and transverse
  anisotropy and angle, plus a transverse field. A sample declared as clean is
  told so, rather than being told it "has not declared".
- **A model's required sample is shown as the chain it describes**, with a row
  per impurity, instead of the training recipe printed as JSON, and each row is
  marked against the declared sample. A one-line form appears on the model
  cards and in the compatibility table, so what a model demands is visible
  before it is selected.
- **Timing claims name the right stage.** Three places said training takes
  hours. Generating the dataset is the slow stage; fitting a model to it is
  minutes.
- Coupling estimates are no longer printed on top of their own error bars.
- Formulas in `docs/theory.md` used LaTeX thin spaces, which GitHub renders as
  commas and semicolons.

### Changed

- The README states which install gives which capability, next to the install
  command rather than forty lines below it. The base package can use the ridge
  and random-forest reference models but cannot generate a dataset; `[ml]` adds
  the Keras models including the variable-length local-window one, and
  `[simulation]` adds generation. An environment that already has TensorFlow or
  DMRGPy needs no extra.

## [0.1.0] - 2026-09-16

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
  against every model it can see -- chain length, view, exact cutoff,
  observable, preprocessing -- and reports a mismatch instead of padding
  spectra or silently changing the cutoff. Which family of Hamiltonian to
  assume is left to you, since spectra do not decide it; what a model requires
  of the *sample* is not, so a model trained with impurities at given sites
  stays refused until the measured chain is declared to have them. That
  declaration is made once, on a chain diagram, and both the reuse check and
  the inference run read it.
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
  Engine, chosen and sent from the training page itself. Generation goes as a
  scheduler array -- one task per simulated chain, each in its own scratch
  directory -- and a dependent job joins the checkpoints and trains only if
  every task succeeded. Key-based access is required and checked, the run
  directory can be browsed rather than typed, job logs are collected in
  `logs/`, and the connection test confirms the cluster can import a HamLeT
  new enough to run the array before anything is copied.
- **Held-out evaluation before a model is used.** Training reserves whole
  simulated chains from both fitting and selection, then reports MAE, RMSE,
  fidelity and skill against the training mean -- overall and per learned
  parameter -- in the interface, in `held_out_evaluation.json`, and in the
  PDF report, which also flags any estimate that falls outside the range the
  model was trained on.
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
- **Theory notes** ([docs/theory.md](docs/theory.md)) setting out the
  Hamiltonians as implemented, the correlator and its relation to dI/dV, the
  DMI identifiability argument, the metric definitions, and the assumptions
  behind all of them.
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
