"""Hyperparameter search for the supervised regressors.

Choosing an architecture by hand is the part of a training run with the least
physics in it and the most guesswork, so it is worth automating -- but only in
a way that cannot quietly make a model worse or invalidate its held-out score.
Three rules follow from that, and the whole module is built around them:

* The library defaults are always evaluated first, as trial zero. A search that
  finds nothing better returns the defaults, so tuning can never regress a
  model relative to not tuning it.
* Selection uses the **validation** split only. The test split is never read
  here, so the artifact's held-out MAE remains an honest estimate of a model
  whose hyperparameters it did not choose.
* Trials are trained with the same fixed split as the real run, so their scores
  are comparable to each other and to the untuned baseline.

Optuna drives the search when it is installed (``pip install
"hamlet-toolkit[tune]"``), because its TPE sampler spends later trials near the
good region. Without it the same search space is sampled at random, which is
worse but not much worse at these trial counts, and keeps the feature working
in an environment that cannot take another dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping

import numpy as np

from ..cancellation import OperationCancelled, check_cancelled
from .guided import TrainingRun, train_supervised
from .preprocessing import PreparedTrainingDataset


@dataclass(frozen=True)
class SearchDimension:
    """One tunable quantity, described once for both search backends.

    ``kind`` is ``"int"``, ``"float"``, ``"log_float"`` or ``"categorical"``.
    Log scale matters for the quantities that are chosen by order of magnitude
    -- learning rate, weight decay, ridge strength -- where a uniform draw would
    spend nearly every trial in the top decade.
    """

    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] = ()
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"{self.name}: a categorical dimension needs choices")
            return
        if self.kind not in {"int", "float", "log_float"}:
            raise ValueError(f"{self.name}: unknown dimension kind {self.kind!r}")
        if self.low is None or self.high is None:
            raise ValueError(f"{self.name}: a numeric dimension needs low and high")
        if not self.low < self.high:
            raise ValueError(f"{self.name}: low must be below high")
        if self.kind == "log_float" and self.low <= 0:
            raise ValueError(f"{self.name}: a log-scaled dimension needs a positive low")

    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "categorical":
            # `choices` may hold tuples, which numpy would try to broadcast.
            return self.choices[int(rng.integers(len(self.choices)))]
        if self.kind == "int":
            return int(rng.integers(int(self.low), int(self.high) + 1))
        if self.kind == "log_float":
            return float(
                math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
            )
        return float(rng.uniform(self.low, self.high))

    def ask(self, trial: Any) -> Any:
        """Draw this dimension from an Optuna trial."""
        if self.kind == "categorical":
            index = trial.suggest_categorical(
                self.name, list(range(len(self.choices)))
            )
            return self.choices[int(index)]
        if self.kind == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high))
        if self.kind == "log_float":
            return trial.suggest_float(self.name, self.low, self.high, log=True)
        return trial.suggest_float(self.name, self.low, self.high)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "label": self.label or self.name.replace("_", " "),
        }
        if self.kind == "categorical":
            payload["choices"] = [list(c) if isinstance(c, tuple) else c for c in self.choices]
        else:
            payload["low"] = self.low
            payload["high"] = self.high
        return payload


def _mlp_options(values: Mapping[str, Any]) -> dict[str, Any]:
    """Turn sampled values into MLP hyperparameters.

    Layer widths are searched as a depth, a first width and a taper rather than
    as independent per-layer widths: the funnel shape is what these networks
    actually want, and searching each width separately multiplies the space
    without adding architectures that win.
    """
    width = int(values["first_layer_units"])
    taper = float(values["layer_taper"])
    depth = int(values["n_hidden_layers"])
    units = []
    for _ in range(depth):
        units.append(max(16, int(round(width))))
        width *= taper
    return {
        "hidden_units": units,
        "activation": str(values["activation"]),
        "dropout": round(float(values["dropout"]), 4),
        "l2": float(values["l2"]),
        "learning_rate": float(values["learning_rate"]),
        "batch_normalization": bool(values["batch_normalization"]),
    }


def _cnn_options(values: Mapping[str, Any]) -> dict[str, Any]:
    channels = int(values["first_filters"])
    blocks = int(values["n_conv_blocks"])
    filters = []
    for _ in range(blocks):
        filters.append(max(8, int(round(channels))))
        channels *= 2
    return {
        "filters": filters,
        "kernel_size": int(values["kernel_size"]),
        "dense_units": [int(values["dense_units"])],
        "dropout": round(float(values["dropout"]), 4),
        "l2": float(values["l2"]),
        "learning_rate": float(values["learning_rate"]),
        "batch_normalization": bool(values["batch_normalization"]),
    }


def _ridge_options(values: Mapping[str, Any]) -> dict[str, Any]:
    return {"alpha": float(values["alpha"])}


def _forest_options(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_estimators": int(values["n_estimators"]),
        "min_samples_leaf": int(values["min_samples_leaf"]),
        # None means "every feature", which is scikit-learn's own spelling and
        # the right default for regression on smooth spectra.
        "max_features": values["max_features"],
    }


@dataclass(frozen=True)
class SearchSpace:
    """The tunable dimensions of one model, and how they become options."""

    model: str
    dimensions: tuple[SearchDimension, ...]
    build: Callable[[Mapping[str, Any]], dict[str, Any]]

    def sample_options(self, rng: np.random.Generator) -> dict[str, Any]:
        return self.build({dim.name: dim.sample(rng) for dim in self.dimensions})

    def ask_options(self, trial: Any) -> dict[str, Any]:
        return self.build({dim.name: dim.ask(trial) for dim in self.dimensions})

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dimensions": [dim.to_dict() for dim in self.dimensions],
        }


SEARCH_SPACES: dict[str, SearchSpace] = {
    "keras_mlp": SearchSpace(
        "keras_mlp",
        (
            SearchDimension("n_hidden_layers", "int", 1, 4, label="hidden layers"),
            SearchDimension(
                "first_layer_units",
                "categorical",
                choices=(64, 128, 256, 512, 1024),
                label="width of the first layer",
            ),
            SearchDimension(
                "layer_taper",
                "categorical",
                choices=(1.0, 0.5),
                label="how each layer narrows",
            ),
            SearchDimension(
                "activation", "categorical", choices=("relu", "gelu", "tanh")
            ),
            SearchDimension("dropout", "float", 0.0, 0.5),
            SearchDimension("l2", "log_float", 1e-6, 1e-2, label="weight decay"),
            SearchDimension("learning_rate", "log_float", 1e-4, 5e-3),
            SearchDimension(
                "batch_normalization", "categorical", choices=(True, False)
            ),
        ),
        _mlp_options,
    ),
    "keras_cnn": SearchSpace(
        "keras_cnn",
        (
            SearchDimension("n_conv_blocks", "int", 1, 3, label="convolution blocks"),
            SearchDimension(
                "first_filters", "categorical", choices=(16, 32, 64), label="first filters"
            ),
            SearchDimension("kernel_size", "int", 3, 15, label="kernel width in bias points"),
            SearchDimension(
                "dense_units", "categorical", choices=(64, 128, 256), label="dense width"
            ),
            SearchDimension("dropout", "float", 0.0, 0.5),
            SearchDimension("l2", "log_float", 1e-6, 1e-2, label="weight decay"),
            SearchDimension("learning_rate", "log_float", 1e-4, 5e-3),
            SearchDimension(
                "batch_normalization", "categorical", choices=(True, False)
            ),
        ),
        _cnn_options,
    ),
    "ridge": SearchSpace(
        "ridge",
        (
            SearchDimension(
                "alpha", "log_float", 1e-6, 1e2, label="regularisation strength"
            ),
        ),
        _ridge_options,
    ),
    "random_forest": SearchSpace(
        "random_forest",
        (
            SearchDimension("n_estimators", "int", 100, 800, label="number of trees"),
            SearchDimension(
                "min_samples_leaf", "int", 1, 8, label="minimum samples per leaf"
            ),
            SearchDimension(
                "max_features",
                "categorical",
                choices=("sqrt", "log2", None),
                label="features considered per split",
            ),
        ),
        _forest_options,
    ),
}


def optuna_available() -> bool:
    """Whether the TPE backend can be used, without importing it for real."""
    from importlib.util import find_spec

    return find_spec("optuna") is not None


@dataclass(frozen=True)
class TuningTrial:
    """One evaluated configuration."""

    number: int
    options: dict[str, Any]
    validation_mae_mev: float | None
    seconds: float
    status: str = "finished"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "options": dict(self.options),
            "validation_mae_mev": self.validation_mae_mev,
            "seconds": round(self.seconds, 2),
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class TuningReport:
    """What a search evaluated, what it chose, and what that is worth.

    ``improvement_mev`` is the honest headline: how much better than the
    library defaults the chosen configuration scored on validation. It is zero
    when the defaults won, which is a real and common outcome.
    """

    model: str
    backend: str
    preset: str
    n_trials_requested: int
    trials: tuple[TuningTrial, ...]
    best_options: dict[str, Any]
    best_validation_mae_mev: float
    baseline_options: dict[str, Any]
    baseline_validation_mae_mev: float
    kept_defaults: bool
    seconds: float

    @property
    def improvement_mev(self) -> float:
        return max(0.0, self.baseline_validation_mae_mev - self.best_validation_mae_mev)

    @property
    def improvement_fraction(self) -> float:
        if not self.baseline_validation_mae_mev > 0:
            return 0.0
        return self.improvement_mev / self.baseline_validation_mae_mev

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "preset": self.preset,
            "selection_split": "validation",
            "n_trials_requested": self.n_trials_requested,
            "n_trials_evaluated": len(self.trials),
            "n_trials_failed": sum(1 for t in self.trials if t.status == "failed"),
            "best_options": dict(self.best_options),
            "best_validation_mae_mev": self.best_validation_mae_mev,
            "baseline_options": dict(self.baseline_options),
            "baseline_validation_mae_mev": self.baseline_validation_mae_mev,
            "kept_defaults": self.kept_defaults,
            "improvement_mev": self.improvement_mev,
            "improvement_fraction": self.improvement_fraction,
            "seconds": round(self.seconds, 1),
            "trials": [trial.to_dict() for trial in self.trials],
            "note": (
                "Hyperparameters were selected on the validation split only; the "
                "test split was never read during the search, so the artifact's "
                "held-out score remains an estimate of a model whose settings it "
                "did not choose."
            ),
        }


def tune_supervised(
    prepared: PreparedTrainingDataset,
    *,
    view: str,
    model: str = "keras_mlp",
    n_trials: int = 20,
    preset: str = "quick",
    base_options: Mapping[str, Any] | None = None,
    seed: int = 0,
    timeout_seconds: float | None = None,
    progress: Callable[[TuningTrial], None] | None = None,
    backend: str | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> TuningReport:
    """Search hyperparameters for ``model`` and report the best configuration.

    ``preset`` is the compute budget of each *trial*, not of the final model:
    ``quick`` (one seed, few epochs) keeps a search affordable, and the winning
    settings are then trained properly by the caller. Trials therefore rank
    configurations rather than measure them, which is what a search needs.

    ``base_options`` are held fixed across every trial, including the baseline.
    Use it for anything the search must not touch.

    Raises ``ValueError`` for a model with no search space, and only when
    *every* trial fails -- an individual failure is recorded and skipped, since
    a combination the library rejects is information, not a crash.
    """
    if model not in SEARCH_SPACES:
        raise ValueError(
            f"no search space for model {model!r}; "
            f"tunable models are {sorted(SEARCH_SPACES)}"
        )
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    space = SEARCH_SPACES[model]
    fixed = dict(base_options or {})
    started = time.time()
    trials: list[TuningTrial] = []

    def evaluate(options: Mapping[str, Any], number: int) -> TuningTrial:
        # Asked before each trial rather than inside one: a trial is short by
        # construction, and stopping between them leaves a complete record of
        # every configuration that was actually scored.
        check_cancelled(should_stop, f"the {model} hyperparameter search")
        merged = {**fixed, **options}
        trial_started = time.time()
        try:
            run = _train_once(
                prepared, view=view, model=model, preset=preset, options=merged,
                should_stop=should_stop,
            )
        except OperationCancelled:
            # Not a failed trial: the caller stopped the search, and burying
            # that as "this combination did not work" would be a lie.
            raise
        except Exception as exc:  # noqa: BLE001 - a rejected combination is data
            trial = TuningTrial(
                number=number,
                options=dict(options),
                validation_mae_mev=None,
                seconds=time.time() - trial_started,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            trial = TuningTrial(
                number=number,
                options=dict(options),
                validation_mae_mev=float(run.metrics["validation"]["ensemble"]["mae"]),
                seconds=time.time() - trial_started,
            )
        trials.append(trial)
        if progress is not None:
            progress(trial)
        return trial

    # Trial zero is the library defaults, so the search has something to beat
    # and the caller can always fall back to a configuration known to work.
    baseline = evaluate({}, 0)
    if baseline.validation_mae_mev is None:
        raise ValueError(
            f"training {model} with its default settings failed, so there is "
            f"nothing to tune against: {baseline.error}"
        )

    chosen = backend or ("optuna" if optuna_available() else "random")
    if chosen == "optuna":
        _search_with_optuna(
            space, evaluate, n_trials=n_trials, seed=seed, timeout_seconds=timeout_seconds
        )
    elif chosen == "random":
        _search_at_random(
            space, evaluate, n_trials=n_trials, seed=seed, timeout_seconds=timeout_seconds
        )
    else:
        raise ValueError(f"unknown search backend {chosen!r}")

    scored = [t for t in trials if t.validation_mae_mev is not None]
    best = min(scored, key=lambda t: t.validation_mae_mev)
    # Ties go to the defaults: an equal score is not a reason to ship a
    # configuration nobody has run before.
    kept_defaults = best.number == 0 or not (
        best.validation_mae_mev < baseline.validation_mae_mev
    )
    return TuningReport(
        model=model,
        backend=chosen,
        preset=preset,
        n_trials_requested=n_trials,
        trials=tuple(trials),
        best_options=dict(fixed if kept_defaults else {**fixed, **best.options}),
        best_validation_mae_mev=float(
            baseline.validation_mae_mev if kept_defaults else best.validation_mae_mev
        ),
        baseline_options=dict(fixed),
        baseline_validation_mae_mev=float(baseline.validation_mae_mev),
        kept_defaults=kept_defaults,
        seconds=time.time() - started,
    )


def _train_once(
    prepared: PreparedTrainingDataset,
    *,
    view: str,
    model: str,
    preset: str,
    options: Mapping[str, Any],
    should_stop: Callable[[], bool] | None = None,
) -> TrainingRun:
    return train_supervised(
        prepared,
        view=view,
        model=model,
        preset=preset,
        model_options=dict(options),
        verbose=0,
        should_stop=should_stop,
    )


def _search_at_random(
    space: SearchSpace,
    evaluate: Callable[[Mapping[str, Any], int], TuningTrial],
    *,
    n_trials: int,
    seed: int,
    timeout_seconds: float | None,
) -> None:
    rng = np.random.default_rng(seed)
    deadline = None if timeout_seconds is None else time.time() + timeout_seconds
    for number in range(1, n_trials + 1):
        if deadline is not None and time.time() >= deadline:
            break
        evaluate(space.sample_options(rng), number)


def _search_with_optuna(
    space: SearchSpace,
    evaluate: Callable[[Mapping[str, Any], int], TuningTrial],
    *,
    n_trials: int,
    seed: int,
    timeout_seconds: float | None,
) -> None:
    import logging

    import optuna

    # Optuna logs a paragraph per trial at INFO, which would drown the job
    # output the interface streams back to the user.
    optuna.logging.set_verbosity(logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )

    def objective(trial: Any) -> float:
        result = evaluate(space.ask_options(trial), trial.number + 1)
        if result.validation_mae_mev is None:
            # Pruned rather than failed: a rejected combination should steer the
            # sampler away, not abort the study.
            raise optuna.TrialPruned()
        return result.validation_mae_mev

    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)


def format_tuning_table(report: TuningReport, *, limit: int = 10) -> str:
    """Render the best trials as a fixed-width table, best first."""
    scored = sorted(
        (t for t in report.trials if t.validation_mae_mev is not None),
        key=lambda t: t.validation_mae_mev,
    )
    lines = [
        f"{report.backend} search over {report.model}: "
        f"{len(report.trials)} trial(s) at the {report.preset} preset",
        f"{'trial':>6s}{'val MAE [meV]':>15s}  settings",
        "-" * 72,
    ]
    for trial in scored[:limit]:
        label = "defaults" if trial.number == 0 else _compact(trial.options)
        lines.append(f"{trial.number:6d}{trial.validation_mae_mev:15.4f}  {label}")
    failed = [t for t in report.trials if t.status == "failed"]
    if failed:
        lines.append(f"{len(failed)} trial(s) failed and were skipped")
    lines.append("")
    if report.kept_defaults:
        lines.append(
            "No configuration beat the defaults on validation, so the defaults "
            "were kept."
        )
    else:
        lines.append(
            f"Best beats the defaults by {report.improvement_mev:.4f} meV "
            f"({100 * report.improvement_fraction:.1f}%) on validation."
        )
    return "\n".join(lines)


def _compact(options: Mapping[str, Any]) -> str:
    parts = []
    for key, value in options.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        elif isinstance(value, (list, tuple)):
            parts.append(f"{key}=[{','.join(str(v) for v in value)}]")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def describe_search_spaces() -> dict[str, Any]:
    """Every tunable model and what a search would vary, for the interface."""
    return {
        "optuna_available": optuna_available(),
        "spaces": {name: space.to_dict() for name, space in SEARCH_SPACES.items()},
    }


__all__ = [
    "SEARCH_SPACES",
    "SearchDimension",
    "SearchSpace",
    "TuningReport",
    "TuningTrial",
    "describe_search_spaces",
    "format_tuning_table",
    "optuna_available",
    "tune_supervised",
]
