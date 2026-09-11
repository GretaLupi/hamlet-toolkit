<img src="../assets/logos/hamlet-icon.png" alt="" width="40" align="left">

# Using HamLeT

HamLeT provides a general experiment-to-inference workflow for STM/STS data.
This release implements that workflow for Heisenberg and related spin-chain
Hamiltonians; spin chains are the first physics mode, and the package structure
is intended to support additional observables, simulators, and inverse problems
without changing the experimental workflow.

## The browser interface

If you are not sure which workflow applies to your situation -- the usual
problem, and not only for new users -- start here instead of with the CLI:

```bash
hamlet gui
```

It presents the package as a set of situations ("I have measured data and want
to know if an existing model fits it") rather than a set of commands, and every
screen states what it is about to do, what it needs, and how long it takes
before doing it. Everything below is available through it, and the interface
calls the same library functions, so nothing is available in one and not the
other.

### Giving it your data

Nowhere in the interface do you type a path. Every page that needs a
measurement takes it three ways:

- **drag it onto the page**, or **Choose a file…** — sends one prepared file to the
  local server, which keeps a copy under `results/gui-uploads/` and uses that
  path from then on. Use this when the browser and the data are on the same
  machine;
- **Choose a folder…** — sends all `.dat`/`.txt` site spectra in a local folder,
  with upload progress shown on the page. The files are kept together as one
  experiment under `results/gui-uploads/folders/`;
- **Browse this machine…** — a chooser over the machine running the server. It
  can select either one prepared measurement or a directory containing one
  Nanonis `.dat`/`.txt` STS export per chain site. Use this over an SSH tunnel,
  where the data is on the far end and never needs to travel.

For a raw folder, **Use this folder** appears when HamLeT finds STS files in
the current directory. Files are placed at sites 1...N in natural filename
order (`site_2` comes before `site_10`). The automatic importer reads
`Bias calc (V)` (or `Bias (V)`) and `LI Demod 1 X (A)`, converts volts to meV,
and writes the canonical NPZ and CSV under `results/gui-experiments/`. If
`LI Demod 2 X (A)` is present in every file, it is retained for plotting and
quality control only; inference continues to use dI/dV. A folder should hold
one measurement/chain. Other laboratory text formats remain supported through
a text-import recipe, where their column names and delimiter are explicit.

Inspection plots all sites with labelled axes and a site legend. When the
second-derivative channel is available, buttons above the plot switch between
dI/dV and d²I/dV². Moving the cutoff control updates the shaded analysis window
on the plot before model matching.

Once a file is chosen on one page, the others pick it up, so inspecting a
measurement and then asking whether a model fits it does not mean answering the
same question twice.

### Generating a dataset and training, without writing a configuration

The **Train a model** page is a form, not a file to edit. It asks, in order:

1. **Which system** you are measuring — each option says what it recovers and
   when to pick it. An XXZ chain carrying DMI with nothing to break the
   symmetry is not offered at all: `D_z` is exactly unidentifiable there, so
   every model trained on it scores about zero `D_z` skill by construction.
   Measuring DMI means the impurity system, screened first on the **DMI sample
   design** page.
2. **The chain and its couplings** — number of sites, how many chains to
   simulate, and the range each coupling is drawn from. For a system that takes
   impurities the chain is drawn as balls and sticks: click a site to put one
   there, click it again to take it off. A new impurity copies the system's own
   default properties, so it can actually expose DMI rather than sitting there
   inert; the table below carries each one's measured spin and anisotropies.
3. **The measurement to simulate** — bias window, points, broadening,
   observable, and the analysis cutoff.
4. **A sample check** — simulates two or three chains with exactly those
   settings and plots them, so you can see whether the bias window contains the
   excitations and the broadening is not washing them out. About a minute per
   chain, against hours for a full run.
5. **The model, and how big** — ridge, random forest, or a neural network.
   **Design the network** opens the full hyperparameter set: for the MLP, one
   row per hidden layer with its width, added and removed like any other list,
   plus activation, dropout, weight decay, learning rate, batch normalisation
   and the Huber delta; for the CNN, the convolution blocks, kernel width and
   dense layers. Every field starts at the library default, and clearing one
   returns that single setting to it. Values the library would reject are
   refused here rather than after generation. Models needing TensorFlow are
   disabled when it is not installed.
   Instead of choosing by hand you can tick **search before training** (see
   below).
6. **Review and run** — the plan lists every file it would write and a compute
   estimate before anything happens.

Settings that cannot work are refused at step 6 rather than hours into the run:
a cutoff outside the simulated window, more model input points than simulated
points, impurity sites off the end of the chain, or too few sites for a local
sliding window.

Your answers are saved as a configuration file, so the same run can be repeated
or submitted to a cluster with `hamlet run <path>`. The page shows that path;
you never have to open the file.

### Letting a search choose the hyperparameters

Ticking **search before training** adds a search stage between generation and
training. Each trial is one short training run at the `quick` preset, so twenty
trials cost roughly twenty quick runs — affordable next to generation, which
has already happened by then.

Two rules make it safe to leave on:

- the **library defaults are trial zero**, and are kept if nothing beats them,
  so a search can never produce a worse model than not searching;
- selection reads the **validation split only**. The test split is never
  touched during the search, so the artifact's held-out MAE stays an estimate
  of a model whose hyperparameters it did not choose.

Anything you set by hand that the search does not vary is held fixed. Every
trial, its settings and its score are written to `tuning.json` beside the
artifact, along with why the winner won.

With [Optuna](https://optuna.org) installed (`pip install
"hamlet-toolkit[tune]"`) the search uses its TPE sampler, which spends later
trials near the good region. Without it the same space is sampled at random,
which is worse but still works — the page says which is in use.

From a configuration, the same thing is `training.tuning`:

```yaml
training:
  model: keras_mlp
  preset: standard
  tuning:
    n_trials: 20
    preset: quick          # the budget of one trial, not of the final model
    timeout_seconds: 3600  # optional
```

### Getting the couplings out

**Get my couplings** applies a trained model — published or your own — to a
measurement and writes the answers. You choose the measurement and the model;
you are not asked for a cutoff, because the cutoff is part of the model's
contract and is never substituted. The model is re-checked against your data
before it runs, so a mismatch is refused rather than answered.

It writes, into a timestamped directory under `results/gui-analyses/`:

| File | What |
| --- | --- |
| `analysis/report.html` | the full report, self-contained |
| `analysis/summary.png` | quality-control figure |
| `analysis/couplings.csv` | the coupling table |
| `analysis/report.json` | the same numbers, machine-readable |
| `analysis.yaml` | the configuration, so `hamlet run` repeats it |

The report, the figure and the table open straight from the page.

For local-bond inference, the browser presents the answer as the physical spin
chain: numbered site circles joined by bonds labelled with the inferred
coupling and model spread. Bond thickness compares absolute coupling strength
within that chain, while colour/dashing distinguishes the sign. The exact
machine-readable values remain directly below it in the table.

When the run finishes, the model appears on the **Existing models** page marked
*you trained this*, on **Get my couplings**, and in the reuse advisor alongside
the published ones.

A note on the sample check: simulation cost is set by how many correlators are
evaluated, which is one per site per observable component, and *not* by the bias
resolution -- 81 bias points cost the same as 21. The preview therefore
simulates a few representative sites rather than the whole chain, and switches
from exact diagonalisation to DMRG once the basis grows past a few thousand
states. It says which sites and which method it used.

### Designing a DMI sample

The **DMI sample design** page is a form as well: the chain you can build (its
length, `J_z`, `J_2`, `J_3`, the exchange scale `sqrt(J1_xy^2 + D_z^2)` and the
`D_z` you are trying to resolve), the measurement you can take, and one row per
candidate arrangement — impurity sites, spin, transverse and axial anisotropy,
and any transverse field.

Each candidate is a card with its own chain diagram — click the sites you would
put impurities on — and its verdict updates underneath as you click, since the
symmetry rule costs nothing to apply.

**Check symmetry** is free and exact: it says which arrangements can break the
symmetry that hides `D_z` at all, without simulating anything. One impurity
never can. **Run the full screening** then simulates a gauge pair per surviving
candidate, about a minute each, and ranks them by imprint against the
calibration table shown on the page — each threshold anchored to the `D_z`
skill a model trained on that design actually reached.

The design is saved as a screening configuration, so
`hamlet screen-dmi <path>` repeats it.

### Where your files are saved

The *Start here* page lists every folder the interface writes to, and so does

```bash
hamlet where
```

The location depends on how HamLeT was installed, which is why it is worth
asking rather than assuming:

| Install | Workspace |
| --- | --- |
| a source checkout | `results/` beside the project |
| an installed package | `~/.hamlet/workspace/` |
| `HAMLET_WORKSPACE` set | wherever it points |

An installed package deliberately does not write beside itself: that is inside
`site-packages`, which pip replaces on upgrade, and your measurements would go
with it.

Within the workspace: `gui-uploads/` holds copies of dropped files,
`gui-experiments/` the folders of raw per-site spectra once converted,
`gui-projects/` one folder per training run (configuration, dataset, trained
artifact), `gui-analyses/` one folder per inference (`report.html`,
`summary.png`, `couplings.csv`, `report.json`), and `gui-screenings/` the saved
DMI designs. Every path a run reports is inside one of these.

### While you wait

Generation and training are long — a single simulated chain is around a minute
— so once a run has been going for a few seconds the **Running** tab, and the
page that started it, show a line of Shakespeare under the progress. The
package is called HamLeT; the lines are mostly from the play. They rotate as
the wait goes on, and there are a few reserved for a run that has been going
a quarter of an hour.

If you would rather not have them, the **hide** link beside a quote turns them
off for good in that browser.

### Stopping it

The interface is a server, so **closing the browser tab does not stop it** --
it keeps running and keeps its port. Any of these will stop it:

- the **Stop server** button on the page, which warns first if a run is still
  going and would be lost
- `Ctrl+C` in the terminal where you started it
- `pkill -f "hamlet gui"` if you have lost track of it

If you suspended it with `Ctrl+Z`, it is still holding its port while answering
nothing: bring it back with `fg` (then `Ctrl+C`), or list suspended jobs with
`jobs`. Starting a second interface is harmless -- it takes the next free port
and tells you which.

### On a remote machine

With no display there is nothing to open, so the interface prints an SSH
port-forward command instead. Run that on your laptop and open the URL there.

The rest of this guide covers the command line, which is what you want for
scripted or cluster runs.

This is the complete path from raw per-site spectroscopy files to an inferred
Hamiltonian: import your experiment, choose a physically usable cutoff, let
the package decide whether to reuse, retrain, or generate a model, then read
the report. See the [README](../README.md) for install instructions.

## The report you send to someone else

Every analysis writes its results five ways into the analysis folder:

| File | For |
| --- | --- |
| `report.pdf` | a paper-shaped summary: the Hamiltonian, the method, one table |
| `report.tex` | its LaTeX source, to edit or paste into a draft |
| `report.html` | the full report, with every diagnostic, self-contained |
| `summary.png` | the quality-control figure on its own |
| `couplings.csv`, `report.json` | the numbers, for a script |

The PDF is the one to send to a collaborator. It writes out the Hamiltonian
that was inferred — transcribed from what the simulator actually builds, not a
textbook form that resembles it — states the method in a paragraph, gives the
couplings as a table, and carries the caveat that belongs with every estimate:
the ensemble spread measures disagreement between trained members and is not a
calibrated confidence interval.

`report.tex` is written whether or not you have LaTeX installed, because it is
the durable thing: upload it and its `-summary.png` to Overleaf and it
compiles with nothing installed locally. For a PDF here, any of `tectonic`,
`latexmk`, or `pdflatex` will do — HamLeT uses whichever it finds, and says
which in the result. It needs only `amsmath`, `graphicx`, and `geometry`, so a
minimal TeX install is enough.

From Python:

```python
result.save_latex_report("report.tex", title="Chain S1")
```

## Sending a run to a cluster

Two facts are needed: the address you `ssh` to, and which batch system the
cluster runs. Both go in the form under *Where it runs*.

**Access must be key-based.** Runs are submitted from a background thread with
no terminal attached, so nothing can answer a password prompt — HamLeT passes
`BatchMode=yes` to `ssh`, which makes it fail with a message instead of
hanging on a question nobody will see. Set a key up once:

```bash
ssh-keygen -t ed25519          # only if you have no key yet
ssh-copy-id you@cluster.example.edu
ssh you@cluster.example.edu true
```

That last line must succeed without asking you for anything. If your key has a
passphrase, load it into `ssh-agent` first. **Test the connection** on the page
distinguishes a refused key from an unreachable host, because the fixes are
different.

Everything else is optional: cpus, gpus, memory, walltime, queue and account
are passed to the scheduler if you set them and left to the site default if you
do not, and the setup lines are whatever your site needs to make
`python -m hamlet` work — usually a `module load` and a virtualenv.

The batch systems offered are Slurm, PBS/Torque, LSF, Grid Engine, and *no
scheduler* (which just runs the job in the background on that machine). You do
not configure how they are spoken to; picking the name is the whole decision.
If your site runs something else, write a `scheduler:` block by hand in
`cluster.yaml` — `hamlet where` prints its location — and the form will tell
you it cannot show it rather than quietly replacing it.

### Windows

This is how a Windows user reaches a GPU. TensorFlow cannot use one on native
Windows at all, so the interface does not offer **Require a GPU** there;
*Automatic* already takes a GPU whenever the job lands on a node that has one,
which is exactly what happens on a cluster. Ask for one with `gpus: 1` in the
cluster form.

## Making generation faster

Generation is the stage that costs hours. Every chain is an independent
simulation, so the way to shorten it is to run several at once:

```yaml
dataset:
  generate:
    n_samples: 3000
    workers: 8      # chains at once; 0 means one per core
```

or **chains at once** on the *Train a model* page. The plan divides its
estimate by that number before you commit to the run.

This cannot change the dataset. Each chunk's seed is derived from the run's
seed by position, so a chunk is identical whenever and wherever it is
simulated, and the samples are assembled in recipe order rather than
completion order — the result is bit-identical on one core and on twelve.
`workers` is therefore not part of the recipe fingerprint: you can stop a run
on four cores and resume it on sixteen, and a dataset generated on one machine
stays valid on another.

Three practical limits. Each worker is a separate process holding its own
simulation, so memory use scales with the count — on a shared login node,
`workers: 0` is a way to annoy your colleagues. On Windows and macOS, Python
starts worker processes by re-importing your script, so a `.py` file that
generates a dataset needs the usual guard, or it will try to start the run
again inside every worker:

```python
if __name__ == "__main__":
    main()
```

`hamlet run`, `hamlet gui` and notebooks need nothing — this applies only to
your own scripts. And progress arrives per
checkpoint chunk rather than per chain when workers are in use, which is also
the granularity at which a run can be stopped: **Stop** finishes the chunks
already running and discards the ones not yet started, keeping every chunk
already written.

### Why generation cannot use a GPU

It is DMRG and exact diagonalisation, and neither has a CUDA path in the
simulator HamLeT uses. Recent DMRGPy can put its pure-Python backend's tensors
on a GPU through JAX, but its own published benchmarks put the crossover far
above where this package operates: for a KPM dynamical correlator the device is
**7.7× slower** than one CPU core at a bond dimension of 40, roughly breaks
even near 80, and only wins from about 160 upward. HamLeT's default bond
dimension is 20, and the supported spin-chain workflows are small enough
(Hilbert dimension ≤ 2048) that they use exact diagonalisation, where there is
no GPU path at all.

So a GPU would make this stage slower, not faster. Cores are what help here; a
card only helps the Keras training that follows, which takes minutes.

## Using a GPU

Ask first, before installing anything:

```bash
hamlet compute
```

It lists the cores and cards it can see, says what each model will use, and —
when it sees no GPU — why not. That last part matters, because "no GPU
visible" has several different causes and only some of them are worth acting
on.

**A GPU may not be what you need.** Only `keras_mlp` and `keras_cnn` can use
one; `ridge` and `random_forest` are scikit-learn and run on the CPU whatever
hardware is present. And dataset generation — DMRG and exact diagonalisation,
the stage that takes hours rather than minutes — is CPU-bound and
single-threaded per chain, so a card does nothing for it. More cores shorten a
run here; a faster accelerator usually does not. Check which stage is actually
costing you time before spending an afternoon on drivers.

**On Linux**, the plain `tensorflow` wheel that `[ml]` and `[all]` install is
built with CUDA but ships none of the CUDA runtime libraries, so it finds no
card unless your system already provides them. To let pip install them:

```bash
python -m pip install "hamlet-toolkit[gpu]"
```

The NVIDIA driver still has to come from the system; pip cannot supply it.

**On native Windows there is no GPU path at all.** TensorFlow dropped Windows
GPU support at version 2.11, and its Windows wheels have been CPU-only since.
No driver update, CUDA install, or environment variable changes that, and the
DirectML plugin sometimes suggested instead is pinned to TensorFlow 2.10 and
Python ≤ 3.10 and cannot work with a current install. The interface therefore
does not offer **Require a GPU** on Windows or macOS — it offers only choices
it can honour.

The straightforward answer is to
[send the run to a cluster](#sending-a-run-to-a-cluster), where *Automatic*
picks up the GPU on the node. If you want one on the machine in front of you,
the only route is WSL2:

```powershell
wsl --install -d Ubuntu
```

then, inside Ubuntu — installing the CUDA TensorFlow **first**, because
otherwise HamLeT's plain `tensorflow` requirement is already satisfied and pip
will not add the CUDA packages:

```bash
python3 -m venv ~/.venvs/hamlet && source ~/.venvs/hamlet/bin/activate
python -m pip install "tensorflow[and-cuda]"
python -m pip install "hamlet-toolkit[all]"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Your Windows NVIDIA driver serves WSL2 — do not install a driver inside
Ubuntu. On macOS there is no CUDA at all; Apple's `tensorflow-metal` plugin is
the only accelerator option, and HamLeT neither requires nor tests it.

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

For impurity-assisted DMI, select
`homogeneous_xxz_j1j2j3_dmi_impurity` and provide `impurities` under
`dataset.generate`. The list can contain any number of distinct, zero-based
sites. Each entry accepts `site`, `spin`, `axial_mev`, `transverse_mev`, and
`transverse_angle_rad`, so the simulation can reproduce the experimentally
measured impurity configuration instead of assuming the validated three-site
recipe. The exact list is part of the dataset fingerprint and model contract;
changing it requires a matching dataset and artifact. See
[`examples/heisenberg_xxz_dmi_impurities_l8.yaml`](../examples/heisenberg_xxz_dmi_impurities_l8.yaml).

Ready-to-run configurations:
[`examples/quickstart_l8.yaml`](../examples/quickstart_l8.yaml),
[`examples/heisenberg_generate_dataset.yaml`](../examples/heisenberg_generate_dataset.yaml),
and [`examples/heisenberg_xxz_dmi_impurities_l8.yaml`](../examples/heisenberg_xxz_dmi_impurities_l8.yaml).

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

## Measuring DMI

DMI needs a purpose-built sample rather than a different analysis: a DM vector
along `z` is exactly unidentifiable in any chain that conserves total `S^z`,
measured here as agreement to one part in 10^13 between chains differing only
in how `sqrt(J1_xy^2 + D_z^2)` splits between exchange and DMI.

Impurities carrying transverse magnetic anisotropy break that symmetry and make
`D_z` recoverable, with no magnetic field required. Which arrangement works best
depends on your chain, so the package screens candidates for you:

```bash
cp examples/dmi_screening.yaml my_screening.yaml   # edit chain and candidates
hamlet screen-dmi my_screening.yaml
```

Each candidate is ranked by how well it separates a gauge pair, with verdicts
calibrated against the `D_z` skill models trained on such designs actually
reached. Arrangements that cannot break the symmetry are identified and skipped
without simulating, so sweeping counts and positions is cheap. The sample requirements,
the measurement protocol, and the precision needed on the impurity
characterisation are in
[dmi-experiment-spec.md](dmi-experiment-spec.md). Read the precision section
before committing beam time: a 10% error in the impurity transverse anisotropy
removes all DMI skill, and out-of-distribution detection does not catch it.
