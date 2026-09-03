# Changelog

Notable changes to HamLeT. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
the policy in [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

### Added

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
