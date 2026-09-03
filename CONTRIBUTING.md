# Contributing to HamLeT

## Getting set up

```bash
python -m pip install -e '.[dev,io]'      # core, plotting, pytest
python -m pip install -e '.[dev,io,ml]'   # add Keras/scikit-learn models
pytest
```

The `simulation` extra pulls DMRGPy, which compiles a C++ backend from a git
checkout. It is only needed to *generate* datasets; everything else, including
the whole test suite, runs without it. Tests that need an optional backend skip
themselves rather than fail.

## Before opening a pull request

```bash
pytest                     # the full suite
ruff check src tests       # correctness rules; must be clean
python -m build            # the wheel must build
```

CI runs the suite on 3.10/3.11/3.12, exercises the Keras paths separately, and
verifies that the built wheel contains every module, installs into a clean
environment, and exposes all console scripts.

## Conventions that are not negotiable

These encode decisions that protect scientific results. Please do not
work around them.

- **Physical units are meV** everywhere a value is exposed. Conversion happens
  only at the simulator or importer boundary. The DMRGPy adapter applies
  `1 DMRGPy energy unit = 10 meV`.
- **The bias cutoff belongs to the artifact.** Weights trained at one cutoff
  are never substituted for another, even when the architecture and input shape
  are identical, because the cutoff changes what the input numbers mean.
- **Splits are grouped by simulated chain.** Every window from one chain lands
  in exactly one partition. Local-window learning leaks badly otherwise.
- **The test partition decides nothing.** Aggregation rules, architectures and
  hyperparameters are selected on validation data only.
- **Uncertainty is described honestly.** Ensemble spread is member
  disagreement, not a calibrated confidence interval, and must not be presented
  as one. Raw spread stays visible even under robust aggregation.
- **Nothing is silently overwritten.** Artifacts, analyses, decision reports
  and generated datasets refuse to clobber existing output; a changed
  generation recipe at an existing path is an error, not permission.
- **Physics stays out of the generic layers.** `systems` describes
  Hamiltonians and imports no ML or backend code; simulators implement a
  protocol. Adding a system should not add `if system == ...` to training or
  reporting.

## Style

`ruff` is configured (in `pyproject.toml`) to correctness rules only — pyflakes
plus pycodestyle's syntax and runtime errors. The tree is clean under those, so
a lint failure means a real defect rather than a formatting preference. Style
families such as import order and line length are deliberately not enforced;
run `ruff check src tests --select I,E501` if you want to see them.

Match the surrounding code. Comments should explain *why* something is done,
particularly where a choice protects a scientific contract — those are the
comments that stop someone "simplifying" a guard away later.

## Tests

New behaviour needs a test. For anything that touches a contract — units,
cutoffs, splits, overwrite protection, schema versions — add a test that fails
without the fix. Two lessons worth repeating:

- A test that passes for the wrong reason is worse than no test. When changing
  a shipped default, pin the old value explicitly in tests that depended on it,
  so they keep testing the behaviour rather than the constant.
- The suite runs against a working tree, so it cannot see what is missing from
  version control. `tests/test_packaging.py` exists because two whole
  subpackages were absent from the repository while every test passed.

## Versioning

Semantic versioning, with the pre-1.0 caveat that the public API is still
settling.

- **Patch** — fixes that do not change any documented contract.
- **Minor** — new capability, or a changed default that does not invalidate a
  saved artifact. A retuned warning threshold is minor; predictions are
  unchanged.
- **Major** — anything that invalidates saved artifacts, experiment manifests
  or project configurations, or that changes a physical convention.

Schema versions (`artifact_schema_version`, `config_schema_version`,
`workflow_decision_schema_version`, `plan_schema_version`) are independent and
bump only when a consumer must change. Record every notable change in
[CHANGELOG.md](CHANGELOG.md).

## Scientific claims

Please keep claims proportionate to the evidence. A per-parameter score from a
single split at these dataset sizes is a noisy point estimate, not a
demonstration — resample the split before saying a parameter is recovered. If a
number cannot clear the pessimistic end of its own spread, say so.
