# Changelog

Notable changes to HamLeT. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
the policy in [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

### Added

- Chains of spin greater than one half. `dataset.generate.site_spin`, or
  **spin per site** in the form, accepts S=1/2 through S=5/2 and carries
  through every system family into the simulated Hilbert space. It is part of
  the recipe fingerprint, unlike `workers`, because it changes every number in
  the dataset -- eight spin-1 sites is 6561 states against 256, which also
  moves the simulator from exact diagonalisation to DMRG on its own.
- An impurity may no longer carry the chain's own spin. A substituted site
  matching the chain is not a substitution, and the likelier mistake this
  catches is raising the chain's spin while leaving the impurities at S=1. It
  is refused at the form and again when the family is built, rather than
  partway through an hours-long generation run. Spin magnitude alone still
  does not expose `D_z` -- the transverse anisotropy does -- so this changes
  what counts as an impurity, not the DMI physics.
- A figure in the README showing what the package does: a simulated dynamical
  correlator, the per-site dI/dV an STM would measure, and the couplings a
  trained model reads back out of it. Every number in it comes from the
  package, and `scripts/make_readme_figures.py` regenerates it.
- The two papers whose Hamiltonians and impurity-tomography route this package
  implements are cited in the README and `CITATION.cff`, with the models
  trained for them explicitly noted as not distributed here.
- A LaTeX summary of every analysis, and a PDF beside it when a TeX toolchain
  is installed: `report.tex`, `report.pdf`, and the figure they reference,
  written into the analysis folder alongside the HTML report. It is the
  artifact that leaves the machine -- the Hamiltonian written out, the method
  in a paragraph, one table of couplings, and the caveat that ensemble spread
  is not a calibrated confidence interval. Every formula is transcribed from
  what `DmrgpySimulator` actually assembles, so the document describes the
  model the numbers came from rather than a textbook form that resembles it.
  The `.tex` is written whether or not LaTeX is present, because it compiles
  on Overleaf with nothing installed locally; `tectonic`, `latexmk` and
  `pdflatex` are each used if found, and only `amsmath`, `graphicx` and
  `geometry` are required. Build files are cleaned up on success, and a figure
  that cannot be drawn costs a paragraph rather than the report.
  `save_latex_report()` on both result types backs it.
- Every neural-network setting on the training page now explains what it does
  and which way to move it. The audience measures spectra; a field labelled
  "Huber delta" with no explanation is one nobody touches, or worse, one
  somebody changes at random.
- The inferred chain now inks each bond by coupling strength as well as
  thickening it, so the weak bonds recede and the strong ones carry the eye,
  with a key showing the scale and a tooltip giving each bond as a percentage
  of the strongest. Stroke opacity rather than a computed colour, because the
  hue still has to come from CSS to keep meaning the sign of J.
- The random forest exposes its core count. It ran on every core
  unconditionally, which is the fastest and least polite thing to do on a
  shared login node.

### Changed

- The README states the Python requirement instead of leaving it blank. The
  badge read the supported versions from PyPI, where the package is not
  published, so the one place a reader looks for it showed nothing. It is now
  a static badge naming 3.10 to 3.13, checked against `requires-python` and
  the classifiers by a test so the two cannot drift. Python 3.13 is added to
  the classifiers: it was already inside `requires-python`, and an install
  there is known to work.
- The nanographene paper is cited with its preprint identifier,
  [arXiv:2606.29281](https://arxiv.org/abs/2606.29281), in the README, the
  model card's BibTeX entry, and `CITATION.cff`. It had been cited by title
  and year alone, which is not enough to find it.
- `RELEASING.md` is written for someone who has not published a package
  before: what PyPI and Trusted Publishing are, every field of the pending
  publisher, where each button is, and what the common failures look like.
- Bond shading in the inferred chain spans the couplings actually present
  rather than zero to the largest. Real chains sit in a narrow band -- 32 to
  38 meV is an ordinary result -- and against an absolute scale every bond
  inked within a tenth of full, so the picture said "all the same" about a set
  varying by a fifth. Thickness stays absolute, so genuinely equal couplings
  still look equal, and the key prints the two end values so full contrast
  over a narrow range cannot mislead.
- README images use absolute URLs. PyPI resolves relative links against
  `pypi.org`, so the logo and the new figure would have been broken icons on
  the project page.
- A line of Shakespeare is now always shown while a job runs. The opt-out and
  its stored preference are gone: a decoration that can be switched off
  permanently by a stray click, in browser storage that this session cannot
  see, is a decoration that silently stops existing.
- The project is licensed under the GNU General Public License v3.0 or later,
  replacing the MIT licence.
- The continuous-integration workflow has been removed. The release workflow
  that publishes to PyPI is kept.
- **Require a GPU** is offered on Linux only. It is the only platform where
  TensorFlow can use one -- native Windows has had no GPU build since 2.11 and
  macOS has no CUDA path -- and offering a choice that can never be honoured,
  even labelled as such, is an invitation to spend an afternoon on drivers.
  The form refuses `device: gpu` there rather than downgrading it silently,
  and points at the cluster instead: *Automatic* already takes a GPU whenever
  the job lands on a node that has one, which is how a Windows user reaches
  a card.
- Cluster access requires a key. `ssh` is invoked with `BatchMode=yes` and a
  connect timeout, so it fails with a message rather than waiting on a
  password prompt that nobody can answer -- every cluster operation runs on a
  background thread with no terminal attached, where a prompt is an
  indefinite hang that looks like a broken job. **Test the connection** now
  tells a refused key apart from an unreachable host, because the fixes are
  different, and prints the three `ssh-keygen`/`ssh-copy-id` commands that
  resolve the first.
- The cluster is configured by a form rather than by writing YAML. It asks for
  the address you ssh to and which batch system the site runs; the rest --
  cpus, gpus, memory, walltime, queue, account, setup lines -- is optional and
  left to the site default when blank. The table of scheduler profiles and the
  raw editor are gone from the page: which directives Slurm takes was never a
  decision anyone made, and picking the name is the whole of it. A site whose
  scheduler is none of the five still writes a `scheduler:` block in
  `cluster.yaml` by hand, and the form reports that it cannot show one instead
  of quietly replacing it. `GET /api/cluster-form` backs it.

### Added

- Dataset generation runs several chains at once: `dataset.generate.workers`
  in a configuration, **chains at once** on the training page, `0` for one per
  core. It is the stage that costs hours and every chain is independent, so
  this is the only setting that shortens it -- measured at 2.7x on four
  workers. It cannot change the result: chunk seeds are derived from the run's
  seed by position and samples are assembled in recipe order, so the dataset
  is bit-identical on one core and on twelve, and `workers` is excluded from
  the recipe fingerprint -- a run stopped on four cores resumes on sixteen,
  and a dataset generated on one machine stays valid on another. Each worker
  simulates in its own working directory, because DMRGPy derives its scratch
  paths from the working directory and two workers sharing one would overwrite
  each other's wavefunctions and produce a quietly wrong dataset. Progress and
  cancellation move to chunk granularity while workers are in use. The
  platform's default process start method is used rather than a pinned one:
  spawn and forkserver both re-import `__main__` in every worker and so fail
  wherever it is not importable, including notebooks and `python -c`, which
  would trade an unlikely fork/thread deadlock for certain breakage of a
  documented workflow.
- The interface says where it writes things, on the *Start here* page and from
  the new `hamlet where`. The workspace is `results/` beside a source
  checkout, `~/.hamlet/workspace/` for an installed package, or wherever
  `HAMLET_WORKSPACE` points, and the report says which case applies and why --
  an installed package must not write beside itself, because that is inside
  `site-packages` and pip replaces it on upgrade. Each folder is listed with
  what lands in it. `GET /api/locations` backs it.
- The device cards mark a choice this machine cannot honour. Picking **Require
  a GPU** where none is visible used to highlight the card and say nothing
  until the plan, two screens later. It is now labelled *not on this machine*
  with the reason, and warns that training here will use the CPU -- without
  forbidding it, since a configuration built here is often bound for a cluster
  that does have a card.
- A `gpu` extra, `pip install "hamlet-toolkit[gpu]"`, which installs
  `tensorflow[and-cuda]` on Linux. Neither `[ml]` nor `[all]` ever produced a
  GPU on any platform: they require the plain TensorFlow wheel, which on Linux
  is built with CUDA but ships none of the CUDA runtime libraries and so finds
  no card unless the system already provides them. Deliberately not folded
  into `[all]`, because those wheels are several GB and do nothing on Windows
  or macOS. The floor is 2.14 -- the release where the `and-cuda` extra first
  exists -- since asking 2.13 for it only warns and then installs a
  TensorFlow that can never find the card.
- `hamlet compute` now says *why* it sees no GPU, not just that it does not.
  The bare fact reads as a driver or hardware fault on every platform, and on
  native Windows it is neither: TensorFlow has shipped no Windows GPU support
  since 2.11 and its Windows wheels are CPU-only, so the report says so and
  points at WSL2 instead of leaving someone to update drivers that were never
  the problem. A CUDA-built TensorFlow that sees no card is distinguished from
  a build that never could, and macOS is not sent after a CUDA install it
  cannot use. The reason appears wherever the report does -- `hamlet compute`
  and the interface's compute page.
- A **Using a GPU** section in the user guide: which stages a card can help
  (only Keras training; not generation, which is the long one), the Linux
  install, and the WSL2 route for Windows, including the ordering that makes
  it work -- `tensorflow[and-cuda]` before HamLeT, or pip sees the plain
  `tensorflow` requirement already satisfied and adds no CUDA packages.
- A line of Shakespeare while a job runs. The waits here are long -- one
  simulated chain is about a minute and a dataset is hours -- and all the page
  could offer for that was `142 of 3000 chains`. The package is called HamLeT,
  so the wait now quotes the play: the **Running** tab and the page that
  started the run show an attributed line once a job is more than eight seconds
  old, rotating every thirty seconds. Some lines are held back until a run has
  been going three minutes, and a few more until fifteen. Every line is
  Shakespeare and so public domain, and the **hide** link beside one turns them
  off for good.
- A PyPI release workflow using Trusted Publishing, release instructions,
  citation metadata, a security policy, and structured issue and pull-request
  templates.
- Installation tests now open the published-model catalog from the built
  wheel, catching source-only resources that an import smoke test misses.
- The chain is drawn, and you click the sites you mean. Impurity positions are
  zero-based, must be distinct, and whether an arrangement can expose DMI at
  all depends on where they sit relative to the ends -- a row of numbers hides
  every bit of that. Both pages that choose sites now show a ball-and-stick
  chain: click a site to place an impurity or take one off, keyboard-reachable,
  with the impurity's spin labelled above it. The DMI page restates the free
  symmetry rule under each candidate as the sites are chosen, so "one impurity
  cannot break the symmetry" arrives while the design is being drawn rather
  than after a screening run. A site left off the end by a shortened chain is
  reported rather than silently dropped.
- The HamLeT mark in the page header and as the browser-tab icon, shipped
  inside the package so a wheel install serves them, and inverted under a dark
  theme where black line art on transparency would otherwise vanish.
- Files are chosen, not typed. Every page that needs a measurement accepts a
  drag-and-drop or a file picker -- which uploads to the local server and keeps
  the copy under `results/gui-uploads/` -- and a **Browse this machine** dialog
  over the server's own filesystem, which is the one to use over an SSH tunnel
  where the data never needs to travel. A file chosen on one page carries over
  to the others. `POST /api/upload`, `GET /api/browse` and `GET /api/file` back
  this; the last serves only files the interface itself produced.
- A **Get my couplings** page, which applies a trained model to a measurement
  and writes `report.html`, `summary.png`, `couplings.csv` and `report.json`,
  all openable from the page. This was the one stage the interface could not
  do: you could train a model in the browser and then had to leave for the
  command line to use it, which is exactly where a configuration file
  reappeared. It asks for no cutoff, because the cutoff is part of the model's
  contract and copying it from the manifest removes the chance to disagree with
  it and be refused later.
- DMI sample design is a form rather than a screening YAML: the chain you can
  build, the measurement you can take, and one row per candidate arrangement.
  The free symmetry verdict comes back as soon as the design is written down,
  with the calibration table that makes an imprint mean something. The design
  is still saved as a configuration, so `hamlet screen-dmi <path>` repeats it.
- The neural networks are designed in the form: one row per hidden layer with
  its width, added and removed like any other list, plus activation, dropout,
  weight decay, learning rate, batch normalisation and Huber delta; the CNN
  additionally exposes its convolution blocks, kernel width and dense layers.
  Every field starts at the library default and clearing one returns that
  setting to it. Values the library would reject are refused at the form rather
  than after generation. The `research` preset is offered alongside `standard`
  and `quick`.
- `hamlet.training.tuning`: hyperparameter search over any of the four models,
  reachable from the form as **search before training** and from a
  configuration as `training.tuning`. Two rules make it safe to leave on: the
  library defaults are evaluated first and kept if nothing beats them, so a
  search cannot produce a worse model than not searching; and selection reads
  the validation split only, so the artifact's held-out MAE stays an estimate
  of a model whose hyperparameters it did not choose. Every trial is written to
  `tuning.json` beside the artifact. Optuna's TPE sampler is used when the new
  `tune` extra is installed, and the same space is sampled at random when it is
  not, so the feature works either way.
- `HamiltonianLearningProject.tune()`, `ProjectConfig.tuning`, and a
  `TuningConfig` parsed from `training.tuning`.
- The impurity-assisted DMI family is available through project YAML. Users
  may list any number of impurities at arbitrary distinct sites and provide the
  measured spin, axial anisotropy, transverse anisotropy, and in-plane angle
  for each.
- A validated field-free three-impurity L=8 route, measured on 3000 simulated
  chains: `D_z_magnitude` reaches 0.258 +/- 0.004 meV test MAE and
  0.539 +/- 0.005 skill across five held-out splits with ridge regression.
- Models trained through the interface now appear on the models page, tagged
  as yours rather than published, and the reuse advisor searches them too.
  Previously both looked only in the published resource bank, so the model you had just
  trained was invisible exactly when you would ask about it.
- `HamiltonianLearningProject.prepare_training_data_without_experiment()`, so a
  model can be trained before any measurement exists. The calibrated path
  cannot serve that case -- augmentation is tuned to match a specific
  measurement's noise -- and the interface's own runs have no experiment
  attached, so its Run button previously failed on every project the form
  built.
- Sample previews are faster and honest about what they show. Cost is linear in
  the number of correlators evaluated, which is per site, so only a few
  representative sites are simulated -- ends of the chain plus any impurity
  sites -- and DMRG replaces exact diagonalisation once the basis exceeds 2048
  states. Bias resolution turned out to be nearly free: 81 points cost the same
  as 21, so the knob that looked like the obvious economy was the wrong one.
  Running chains in separate processes was measured at 0.57x, i.e. slower than
  sequential, because one simulation already occupies several cores; it is
  available behind a `workers` argument but no longer the default.
- `spin_multiplicity`, `hilbert_dimension` and `recommended_dynamics_mode`, so
  the ED-versus-DMRG choice can be made on basis size rather than site count.
  `DmrgpySimulator` gained `evaluate_sites` for evaluating a subset of sites.
- The interface's training page is a guided form instead of a YAML editor,
  since editing a configuration file is still coding. It walks through the
  system, the chain and coupling ranges, the impurity arrangement where one
  applies, the measurement to simulate, the model and its hyperparameters, and
  ends at a plan. The options are served from the library, so the form cannot
  offer a choice the library would reject, and combinations that would fail
  during training -- a cutoff outside the simulated window, more input points
  than simulated points, impurity sites off the end of the chain, too few sites
  for a local window -- are refused before any compute is spent. Settings are
  saved as a real configuration so a run can be repeated from the command line
  or a cluster, but nobody has to read it.
- The form can simulate two or three sample chains with the chosen settings and
  plot them, which is the cheap way to notice that a bias window misses the
  excitations or a broadening washes them out before committing to hours of
  generation.
- `hamlet gui` no longer fails when a port is busy: the default falls back to
  the next free port, an explicit `--port` is refused with actionable guidance,
  and a browser that hangs on launch can no longer leave the socket listening
  while nothing is accepted. With no display it prints an SSH port-forward
  command instead of opening nothing.
- The interface can be stopped from the page. Closing a tab leaves the server
  running, which is an easy way to end up with an interface nobody can see
  holding a port nobody can reuse; there is now a Stop button that names any
  run it would abandon, the terminal banner says a tab close is not enough, and
  the page detects a server stopped from the terminal instead of appearing hung.
- Published the bond-inhomogeneous L=12 reference model: the trained Keras
  ensemble from *Learning Inhomogeneous Heisenberg Hamiltonians in Nanographene
  Spin Chains* (Lupi et al., 2026), reused rather than retrained so the
  published artifact is the one the work was done with. Per-bond held-out
  accuracy is 1.400 and 1.407 meV MAE at ~0.55 skill and ~0.88 correlation.
  The model card records the attribution and citation, the 30-45 meV validity
  range, the TensorFlow requirement, and the fact that it is evaluated on a
  single split rather than the five resampled splits used for the homogeneous
  reference.
- The model browser now reads both published-manifest shapes, so artifacts
  imported from earlier work no longer render their model, observable and
  validity range as blanks. It also reports the training-set size from the
  split totals rather than the recipe's `n_samples`, which for a dataset merged
  from shards is the per-shard count -- that would have advertised a
  3000-chain model as trained on 20.
- A local browser interface, `hamlet gui`, built on the standard library so a
  GUI cannot break the science path with a new dependency. It is organised
  around the user's situation rather than the command names, since choosing the
  right workflow is the package's real difficulty: inspect a measurement and
  see whether it is structurally usable, ask the advisor which published models
  fit and why each was rejected, browse model contracts and cards, plan a
  training run with its compute cost before running it, and screen DMI sample
  designs. Long runs execute in the background with their output streamed to
  the page. Binds to localhost; refuses path traversal; static assets are
  covered by packaging guards so an installed wheel is not served a blank page.
- Sample-design screening for DMI: `DmiDesign`, `screen_dmi_designs`,
  `measure_dmi_imprint` and `transverse_impurities`. Given a chain length,
  exchange scale and candidate impurity arrangements, this simulates a gauge
  pair per candidate and ranks them by how well each exposes `D_z`, with
  verdicts calibrated against the `D_z` skill models trained on such designs
  actually reached. Designs with no U(1)-breaking mechanism are identified from
  the symmetry rule and reported without being simulated, so sweeping counts
  and positions costs nothing. Two simulations per candidate replaces a
  training set per candidate.
- Measured condition sensitivity for the published DMI model, on 1600 fresh
  chains across eight perturbed configurations. A 10% error in the impurity
  transverse anisotropy removes all `D_z` skill (0.50 -> 0.02); 20% is worse
  than predicting the training mean; moving one impurity by one lattice site
  costs 11 meV on a parameter ranging 0.3-2.5 meV; and an unmodelled axial
  anisotropy of 1 meV also reaches zero skill. Novelty detection does not
  catch any of it -- the mismatched spectra score as in-distribution, so the
  declared-condition check is the only defence. Recorded in the model card and
  in `condition_sensitivity.json`.
- `hamlet advise` now compares the fixed impurity and transverse-field
  conditions an artifact was trained under, via a new `experiment_conditions`
  argument. `system_type` is identical for every impurity chain regardless of
  impurity count, sites, species or anisotropies, so an undeclared or differing
  configuration blocks model reuse instead of being assumed to match -- the
  same rule the package already applies to the bias cutoff.
- Continuous integration: the test suite on Python 3.10, 3.11 and 3.12, a job
  that installs the `ml` extra so the Keras paths actually execute, a lint gate
  configured to correctness rules only, and packaging guards that compare the
  built wheel against the source tree, install it into a clean environment, and
  run every console script.
- `tests/test_packaging.py`, which compares the working tree against
  `git ls-files`. This is the only kind of check that can catch a module which
  is present locally but missing from a clone.
- `hamlet run --dry-run` and `hamlet generate --dry-run`, reporting the stages,
  the number of chains implied with a cost estimate, every output path, and
  which existing files would make the run refuse — without executing or
  writing anything. `--seconds-per-chain` overrides the built-in estimate and
  `--plan-json` writes the plan as versioned machine-readable output.
- `HamiltonianLearningProject.plan()` with the `ProjectPlan` and
  `PlannedOutput` types behind it.
- `notebooks/04_l8_three_mode_workflow.ipynb`: one L=8 chain length run through
  all three Heisenberg interpretations, scored against known ground truth, with
  sliding-window inference on a held-out chain.
- `examples/l8_demo/`: three small exact-diagonalisation datasets, with recipe
  fingerprints, so that notebook runs immediately after cloning.
- `src/hamlet/resources/models/homogeneous_heisenberg_l8_random_forest_standard_v1`: the
  first reference artifact for a homogeneous system, trained on 3000 L=8
  exact-diagonalisation chains. 0.114 meV held-out test MAE, accepted by the
  advisor, and documented by a model card recording provenance, valid
  parameter ranges and the conditions under which it must not be reused.
- `scripts/benchmark_homogeneous.py`, which repeats training over several
  split seeds and reports the spread, because single-split per-parameter
  scores at these dataset sizes are noisy enough to invert a conclusion.

### Fixed

- The waiting quotes were unstable and could not be turned back on. Three
  faults: the line was derived from the elapsed time sampled at render, and
  the jobs list polls every three seconds against a job page's two, so the
  same run quoted two different lines on two pages at once -- the choice is
  now held per job and rotated on a stored timestamp, so every surface reads
  one answer. **hide** wrote a permanent preference with no control to undo it,
  leaving clearing site data as the only way back; the *Running* page now
  carries a checkbox that reflects and sets it. And a missing `quotes.js` --
  an older install, a stale cache -- threw from inside the job renderer, which
  `refreshJobs` caught and reported as *the server has stopped*, blacking out
  the jobs page over a decoration; the call is now guarded, so the worst case
  is no quote.
- The interface's own HTML, CSS and JavaScript are served `no-store`. They
  ship inside the package and change when it is upgraded, so a cached
  `index.html` against a new bundle produced a page assembled from two
  versions -- which is how the quotes came to be missing after an update.
  There is no bandwidth argument against it on localhost.

### Removed

- The XXZ + J2 + J3 + DMI system with no symmetry breaking is no longer offered
  in the training form. `D_z` is exactly unidentifiable in a chain conserving
  total `S^z`, and models trained on it were measured at about zero `D_z` skill
  at every dataset size tried; a warning is the wrong instrument for a choice
  that is never right. The family remains in the library, because reproducing
  that measurement needs it, and the impurity system -- which does work -- is
  unaffected.

### Fixed

- Figures are drawn on a non-interactive backend. Every job runs on a worker
  thread, and matplotlib documents creating a figure on an interactive backend
  from one as likely to fail -- so an analysis started from the page could hang
  or crash on any machine with a display, which is most of them.
- `train()` and `tune()` prepare training data by whichever path the project
  actually has. Both took the calibrated one unconditionally and so demanded a
  calibration step that cannot exist without a measurement, which meant calling
  either directly on a model-only project failed.
- The test suite no longer writes into the user's real `results/` directory.
  Tests that go through the HTTP routes cannot be handed a workspace, so they
  left projects and screenings behind in the tree, and a stale one would then
  appear in the interface's own list of models the user had trained.
- An uploaded file keeps a readable name and its extension. Every disallowed
  character became an underscore, including the ones in the extension, so a
  dropped `.npz` could be stored as `_npz` and then read as a CSV.

### Changed

- Project positioning now describes HamLeT as a general experimental
  Hamiltonian-inference toolkit for STM/STS, with spin chains identified as its
  first implemented physics mode. The release maturity is beta/pre-1.0: the
  supported workflows are tested and usable while the extension API continues
  to evolve. The README now cites the paper associated with the pretrained
  inhomogeneous-Heisenberg model.
- Published model artifacts now live under `src/hamlet/resources/models/` and
  are included as package data, so the GUI catalog contains the same reference
  models after `pip install` as it does in a source checkout.
- Experimental I/O and plotting moved into the base installation because the
  browser interface depends on them. DMRGPy is now an ordinary PyPI
  requirement in the `simulation` extra rather than a direct Git URL, making
  the distribution acceptable to public package indexes.
- The README is a concise interface-first entry point. The browser-superseded
  experimental notebook and the internal validation-study notebook were
  removed; notebooks are examples, while `pytest` is the executable test
  suite.
- `HamiltonianLearningProject.run()` handles a project with no experiment: it
  generates if needed, trains, and stops, rather than failing on the inspection
  stage. Building a model before the measurement exists is an ordinary thing to
  want, and the configuration the guided form writes could not previously be
  re-run by the `hamlet run` command the form printed -- so the interface
  reimplemented the stages itself and the two could drift. `ProjectOutcome`
  gained optional `analysis_dir` and `report_path`, which are `None` for such a
  run.

- `scikit-learn` moved from the optional `ml` extra into the core
  dependencies. The ridge and random-forest models are documented first-class
  choices, the workflow advisor and end-to-end recovery tests exercise them,
  and joblib (which scikit-learn provides) is how every non-Keras artifact is
  saved. Without it the suite reported 6 failures and 21 errors, so the
  declared core install described something that could not train at all. The
  `ml` extra is now TensorFlow only, which is genuinely optional: the suite is
  135 passed and 2 skipped without it.
- Non-Keras artifacts are saved with joblib compression. Tree ensembles pickle
  redundantly — a 600-tree forest on 3000 chains measured 55 MB per seed
  uncompressed against 21 MB compressed, for identical predictions.
  `joblib.load` detects compression, so existing artifacts keep loading.

### Fixed

- **The published package could not be imported.** `.gitignore` entries naming
  a directory without a leading slash match that name at every depth, so the
  root-level `data/` and `models/` rules also excluded `src/hamlet/data/` and
  `src/hamlet/models/` from version control. `hamlet/__init__.py` re-exports
  from `.data` on its third line, so a fresh clone failed at import and the
  model factory was missing entirely. Every root-intended rule is now anchored.
- Bond coupling exports named hardcoded 0-based indices instead of the
  measurement's real site labels, so a chain labelled 1..8 reported its first
  bond as sites 0-1.
- `ExperimentalGlobalResult.save_couplings_csv` no longer derives an
  `interaction_distance` column from a target's position in the vector. That is
  wrong for the anisotropic families, where `J1_xy` and `Jz` are both
  nearest-neighbour terms and `D_z` is not a distance at all.
- `TextImportRecipe` kept its explicit file list as dynamic attributes on a
  frozen dataclass, so `dataclasses.replace()` silently dropped them and
  recipes using `files:` broke when experiment inspection redirected the
  output directory.

### Changed

- The ensemble-disagreement warning threshold moved from 0.1 to 0.25 of the
  trained parameter range. At 0.1 it fired on routine seed-to-seed scatter — a
  2.26 meV spread on couplings around 35 meV — and a warning that fires on
  ordinary runs carries no information. Predictions are unaffected.
- Documentation consolidated: one `docs/user-guide.md` covering the whole
  workflow replaces six overlapping documents that explained cutoff selection
  three separate times.
- The `hamiltonian_learning` import namespace and `hamlearn` commands were
  removed. `hamlet` is the only import and command. No saved joblib or Keras
  artifact embeds the old module path, so the alias carried no compatibility
  value.

## [0.1.0]

First internal release: the bond-inhomogeneous Heisenberg workflow end to end,
from recipe-driven import of one text file per site through manual cutoff
selection, the reuse/retrain/regenerate advisor, guided training, and a
self-contained HTML report.
