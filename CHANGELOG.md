# Changelog

Notable changes to HamLeT. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
the policy in [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

### Added

- The impurity-assisted DMI family is available through project YAML. Users
  may list any number of impurities at arbitrary distinct sites and provide the
  measured spin, axial anisotropy, transverse anisotropy, and in-plane angle
  for each.
- A validated field-free three-impurity L=8 route, measured on 3000 simulated
  chains: `D_z_magnitude` reaches 0.258 +/- 0.004 meV test MAE and
  0.539 +/- 0.005 skill across five held-out splits with ridge regression.
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
- `models/published/homogeneous_heisenberg_l8_random_forest_standard_v1`: the
  first reference artifact for a homogeneous system, trained on 3000 L=8
  exact-diagonalisation chains. 0.114 meV held-out test MAE, accepted by the
  advisor, and documented by a model card recording provenance, valid
  parameter ranges and the conditions under which it must not be reused.
- `scripts/benchmark_homogeneous.py`, which repeats training over several
  split seeds and reports the spread, because single-split per-parameter
  scores at these dataset sizes are noisy enough to invert a conclusion.

### Changed

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
