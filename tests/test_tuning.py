"""Hyperparameter search.

Automating a choice that has no physics in it is worth doing, but a search that
can quietly ship a worse model, or that reads the held-out split, would cost
more than the tuning is worth. These tests are mostly about those two rules.
"""

import json

import numpy as np
import pytest

from hamlet.data import SpectroscopyDataset
from hamlet.training import (
    SEARCH_SPACES,
    TrainingPreprocessingConfig,
    prepare_training_dataset,
    tune_supervised,
)
from hamlet.training.tuning import (
    SearchDimension,
    describe_search_spaces,
    format_tuning_table,
)


def make_training_dataset(n_samples=16, n_sites=5):
    """A learnable toy: peak positions track the couplings."""
    bias = np.linspace(0.0, 80.0, 121)
    targets = np.stack(
        [
            np.linspace(30.0 + sample / 4, 38.0 + sample / 4, n_sites - 1)
            for sample in range(n_samples)
        ]
    ).astype(np.float32)
    spectra = np.empty((n_samples, n_sites, bias.size), dtype=np.float32)
    for sample in range(n_samples):
        for site in range(n_sites):
            left = targets[sample, max(site - 1, 0)]
            right = targets[sample, min(site, n_sites - 2)]
            spectra[sample, site] = (
                0.2
                + np.exp(-0.5 * ((bias - left) / 3.0) ** 2)
                + 0.7 * np.exp(-0.5 * ((bias - right) / 4.0) ** 2)
            )
    return SpectroscopyDataset(
        spectra=spectra,
        targets_mev=targets,
        bias_mev=bias,
        target_names=tuple(f"J{i + 1}" for i in range(n_sites - 1)),
        system_type="inhomogeneous_heisenberg",
        metadata={"generator": "unit-test"},
    )


@pytest.fixture(scope="module")
def prepared():
    return prepare_training_dataset(
        make_training_dataset(),
        TrainingPreprocessingConfig(bias_cutoff_mev=50.0, output_points=30),
    )


# --- the two rules the search exists under -----------------------------------

def test_the_defaults_are_evaluated_first_and_can_win(prepared):
    """A search must never be able to make a model worse than not searching."""
    report = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=5, backend="random"
    )
    assert report.trials[0].number == 0, "trial zero is not the baseline"
    assert report.trials[0].options == {}, "the baseline is not the library defaults"
    assert report.best_validation_mae_mev <= report.baseline_validation_mae_mev
    assert report.improvement_mev >= 0.0
    if report.kept_defaults:
        assert report.best_options == report.baseline_options


def test_a_tie_goes_to_the_defaults(prepared, monkeypatch):
    """An equal score is not a reason to ship settings nobody has run."""
    from hamlet.training import tuning

    def fake_train(prepared_dataset, *, view, model, preset, options, should_stop=None):
        class Run:
            metrics = {"validation": {"ensemble": {"mae": 1.0}}}

        return Run()

    monkeypatch.setattr(tuning, "_train_once", fake_train)
    report = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=4, backend="random"
    )
    assert report.kept_defaults
    assert report.best_options == {}
    assert report.improvement_mev == 0.0


def test_the_search_never_reads_the_test_split(prepared, monkeypatch):
    """Otherwise the artifact's held-out score stops being held out."""
    from hamlet.training import tuning

    seen = []

    class Tripwire(dict):
        """Metrics that fail loudly if the held-out split is ever looked at."""

        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("the search read the test split")
            return super().__getitem__(key)

    def fake_train(prepared_dataset, *, view, model, preset, options, should_stop=None):
        class Run:
            metrics = Tripwire(
                validation={"ensemble": {"mae": 1.0 - 0.01 * len(seen)}},
                test={"ensemble": {"mae": 0.0}},
            )

        seen.append(options)
        return Run()

    monkeypatch.setattr(tuning, "_train_once", fake_train)
    report = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=3, backend="random"
    )
    assert report.to_dict()["selection_split"] == "validation"
    assert len(seen) == 4  # the baseline plus three trials


def test_settings_the_search_does_not_vary_are_held_fixed(prepared, monkeypatch):
    from hamlet.training import tuning

    seen = []

    def fake_train(prepared_dataset, *, view, model, preset, options, should_stop=None):
        seen.append(dict(options))

        class Run:
            metrics = {"validation": {"ensemble": {"mae": float(len(seen))}}}

        return Run()

    monkeypatch.setattr(tuning, "_train_once", fake_train)
    tune_supervised(
        prepared,
        view="local_bonds",
        model="keras_mlp",
        n_trials=3,
        base_options={"huber_delta": 0.05},
        backend="random",
    )
    assert all(options["huber_delta"] == 0.05 for options in seen)
    # And the baseline is the defaults *plus* the fixed settings, not something
    # the fixed settings were left out of.
    assert seen[0] == {"huber_delta": 0.05}


# --- behaviour under failure --------------------------------------------------

def test_a_rejected_combination_is_recorded_and_skipped(prepared, monkeypatch):
    """A combination the library refuses is information, not a crash."""
    from hamlet.training import tuning

    calls = {"n": 0}

    def fake_train(prepared_dataset, *, view, model, preset, options, should_stop=None):
        calls["n"] += 1
        if calls["n"] in (2, 3):
            raise ValueError("no")

        class Run:
            metrics = {"validation": {"ensemble": {"mae": 1.0 / calls["n"]}}}

        return Run()

    monkeypatch.setattr(tuning, "_train_once", fake_train)
    report = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=4, backend="random"
    )
    failed = [trial for trial in report.trials if trial.status == "failed"]
    assert len(failed) == 2
    assert all("no" in trial.error for trial in failed)
    assert report.to_dict()["n_trials_failed"] == 2
    assert "2 trial(s) failed" in format_tuning_table(report)


def test_defaults_that_do_not_train_stop_the_search(prepared, monkeypatch):
    """There is nothing to tune against, and saying so beats searching blind."""
    from hamlet.training import tuning

    def fake_train(prepared_dataset, *, view, model, preset, options, should_stop=None):
        raise ValueError("keras is not installed")

    monkeypatch.setattr(tuning, "_train_once", fake_train)
    with pytest.raises(ValueError, match="nothing to tune against"):
        tune_supervised(
            prepared, view="local_bonds", model="ridge", n_trials=3, backend="random"
        )


def test_an_untunable_model_says_which_are(prepared):
    with pytest.raises(ValueError, match="no search space"):
        tune_supervised(prepared, view="local_bonds", model="nonsense", n_trials=2)


# --- the search space ---------------------------------------------------------

def test_every_offered_model_can_be_searched():
    from hamlet.models import available_supervised_models

    assert set(SEARCH_SPACES) == set(available_supervised_models())


def test_sampled_mlp_architectures_are_buildable():
    """A dimension that produces an unbuildable network wastes a trial."""
    from hamlet.models.supervised import MLPConfig

    rng = np.random.default_rng(0)
    space = SEARCH_SPACES["keras_mlp"]
    for _ in range(50):
        options = space.sample_options(rng)
        config = MLPConfig(**options)
        assert len(config.hidden_units) >= 1
        assert all(units >= 16 for units in config.hidden_units)


def test_sampled_cnn_and_forest_settings_are_valid():
    from hamlet.models.supervised import CNNConfig

    rng = np.random.default_rng(1)
    for _ in range(30):
        CNNConfig(**SEARCH_SPACES["keras_cnn"].sample_options(rng))
    for _ in range(30):
        options = SEARCH_SPACES["random_forest"].sample_options(rng)
        assert options["n_estimators"] >= 1
        assert options["max_features"] in {"sqrt", "log2", None}


def test_log_scaled_dimensions_spread_across_decades():
    """A uniform draw would put nearly every trial in the top decade."""
    dimension = SearchDimension("learning_rate", "log_float", 1e-5, 1e-1)
    rng = np.random.default_rng(0)
    values = np.array([dimension.sample(rng) for _ in range(400)])
    assert values.min() < 1e-4
    assert values.max() > 1e-2
    # Roughly a quarter of draws per decade, which is the point of log scale.
    below_middle = float((values < 1e-3).mean())
    assert 0.35 < below_middle < 0.65


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"name": "x", "kind": "categorical"}, "needs choices"),
        ({"name": "x", "kind": "made_up", "low": 0, "high": 1}, "unknown dimension kind"),
        ({"name": "x", "kind": "float"}, "needs low and high"),
        ({"name": "x", "kind": "float", "low": 1, "high": 0}, "low must be below high"),
        ({"name": "x", "kind": "log_float", "low": 0, "high": 1}, "positive low"),
    ],
)
def test_a_malformed_dimension_is_refused(kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        SearchDimension(**kwargs)


def test_the_report_is_json_serialisable_and_says_how_it_chose(prepared):
    report = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=3, backend="random"
    )
    payload = report.to_dict()
    json.dumps(payload)  # written to tuning.json beside the artifact
    assert payload["backend"] == "random"
    assert payload["n_trials_evaluated"] == 4
    assert "validation split only" in payload["note"]
    assert len(payload["trials"]) == 4


def test_the_backend_is_reported_honestly():
    described = describe_search_spaces()
    assert described["optuna_available"] in {True, False}
    assert set(described["spaces"]) == set(SEARCH_SPACES)
    for space in described["spaces"].values():
        for dimension in space["dimensions"]:
            assert dimension["label"], dimension["name"]


def test_a_real_ridge_search_beats_or_matches_the_default(prepared):
    """End to end against the real trainer, on a dataset small enough to be quick."""
    report = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=6, seed=3, backend="random"
    )
    assert report.best_validation_mae_mev <= report.baseline_validation_mae_mev
    assert report.best_options == {} or "alpha" in report.best_options
    assert "search over ridge" in format_tuning_table(report)


@pytest.mark.skipif(
    not __import__("hamlet.training.tuning", fromlist=["optuna_available"]).optuna_available(),
    reason="optuna is not installed",
)
def test_the_optuna_backend_runs_the_same_space(prepared):
    report = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=4, backend="optuna"
    )
    assert report.backend == "optuna"
    assert len(report.trials) == 5


# --- the report ---------------------------------------------------------------

def test_the_report_quantifies_what_the_search_bought(prepared, monkeypatch):
    """"It found something" is not a result; how much better is."""
    from hamlet.training import tuning

    scores = iter([2.0, 1.5, 1.9, 1.8])

    def fake_train(prepared_dataset, *, view, model, preset, options, should_stop=None):
        value = next(scores)

        class Run:
            metrics = {"validation": {"ensemble": {"mae": value}}}

        return Run()

    monkeypatch.setattr(tuning, "_train_once", fake_train)
    report = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=3, backend="random"
    )
    assert report.baseline_validation_mae_mev == 2.0
    assert report.best_validation_mae_mev == 1.5
    assert report.improvement_mev == pytest.approx(0.5)
    assert report.improvement_fraction == pytest.approx(0.25)
    assert not report.kept_defaults


def test_improvement_is_never_negative(prepared, monkeypatch):
    """Even if every trial is worse, the defaults are what gets returned."""
    from hamlet.training import tuning

    scores = iter([1.0, 5.0, 6.0])

    def fake_train(prepared_dataset, *, view, model, preset, options, should_stop=None):
        value = next(scores)

        class Run:
            metrics = {"validation": {"ensemble": {"mae": value}}}

        return Run()

    monkeypatch.setattr(tuning, "_train_once", fake_train)
    report = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=2, backend="random"
    )
    assert report.kept_defaults
    assert report.improvement_mev == 0.0
    assert report.improvement_fraction == 0.0
    assert report.best_validation_mae_mev == 1.0


def test_a_baseline_of_zero_error_does_not_divide_by_zero(prepared, monkeypatch):
    from hamlet.training import tuning

    def fake_train(prepared_dataset, *, view, model, preset, options, should_stop=None):
        class Run:
            metrics = {"validation": {"ensemble": {"mae": 0.0}}}

        return Run()

    monkeypatch.setattr(tuning, "_train_once", fake_train)
    report = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=2, backend="random"
    )
    assert report.improvement_fraction == 0.0


def test_the_table_reads_best_first_and_names_the_defaults(prepared, monkeypatch):
    from hamlet.training import tuning

    scores = iter([3.0, 1.0, 2.0])

    def fake_train(prepared_dataset, *, view, model, preset, options, should_stop=None):
        value = next(scores)

        class Run:
            metrics = {"validation": {"ensemble": {"mae": value}}}

        return Run()

    monkeypatch.setattr(tuning, "_train_once", fake_train)
    report = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=2, backend="random"
    )
    table = format_tuning_table(report)
    lines = [line for line in table.splitlines() if line.strip()]
    ordered = [line for line in lines if line.strip()[0].isdigit()]
    values = [float(line.split()[1]) for line in ordered]
    assert values == sorted(values), "the table is not best first"
    assert "defaults" in table
    assert "beats the defaults" in table


def test_the_table_says_so_when_nothing_won(prepared, monkeypatch):
    from hamlet.training import tuning

    def fake_train(prepared_dataset, *, view, model, preset, options, should_stop=None):
        class Run:
            metrics = {"validation": {"ensemble": {"mae": 1.0}}}

        return Run()

    monkeypatch.setattr(tuning, "_train_once", fake_train)
    report = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=2, backend="random"
    )
    assert "No configuration beat the defaults" in format_tuning_table(report)


def test_settings_are_rendered_compactly_in_the_table():
    from hamlet.training.tuning import _compact

    rendered = _compact(
        {"hidden_units": [512, 256], "dropout": 0.123456, "activation": "relu",
         "batch_normalization": True}
    )
    assert "hidden_units=[512,256]" in rendered
    assert "dropout=0.1235" in rendered, "floats are not trimmed for the table"
    assert "activation=relu" in rendered
    assert "batch_normalization=True" in rendered


# --- budget and backend -------------------------------------------------------

def test_a_time_limit_stops_the_random_search_early(prepared, monkeypatch):
    """A search that outran its budget would be worse than not offering one."""
    import time

    from hamlet.training import tuning

    def slow_train(prepared_dataset, *, view, model, preset, options, should_stop=None):
        time.sleep(0.05)

        class Run:
            metrics = {"validation": {"ensemble": {"mae": 1.0}}}

        return Run()

    monkeypatch.setattr(tuning, "_train_once", slow_train)
    report = tune_supervised(
        prepared,
        view="local_bonds",
        model="ridge",
        n_trials=200,
        timeout_seconds=0.2,
        backend="random",
    )
    assert len(report.trials) < 200, "the time limit was ignored"
    assert report.n_trials_requested == 200


def test_at_least_one_trial_is_required(prepared):
    with pytest.raises(ValueError, match="at least 1"):
        tune_supervised(prepared, view="local_bonds", model="ridge", n_trials=0)


def test_an_unknown_backend_is_refused(prepared):
    with pytest.raises(ValueError, match="unknown search backend"):
        tune_supervised(
            prepared, view="local_bonds", model="ridge", n_trials=2, backend="magic"
        )


def test_the_backend_falls_back_when_optuna_is_absent(prepared, monkeypatch):
    """The feature has to work without the optional dependency."""
    from hamlet.training import tuning

    monkeypatch.setattr(tuning, "optuna_available", lambda: False)
    report = tune_supervised(prepared, view="local_bonds", model="ridge", n_trials=2)
    assert report.backend == "random"


def test_the_search_is_repeatable_for_a_given_seed(prepared):
    """Two runs of the same search must evaluate the same configurations."""
    first = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=4, seed=7, backend="random"
    )
    second = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=4, seed=7, backend="random"
    )
    assert [t.options for t in first.trials] == [t.options for t in second.trials]
    assert first.best_options == second.best_options


def test_different_seeds_explore_differently(prepared):
    a = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=4, seed=1, backend="random"
    )
    b = tune_supervised(
        prepared, view="local_bonds", model="ridge", n_trials=4, seed=2, backend="random"
    )
    assert [t.options for t in a.trials] != [t.options for t in b.trials]


# --- what each space actually varies ------------------------------------------

def test_the_mlp_search_builds_a_funnel_not_random_widths():
    """Searching each width independently multiplies the space without adding
    architectures that win."""
    from hamlet.training.tuning import _mlp_options

    tapered = _mlp_options({
        "n_hidden_layers": 3, "first_layer_units": 512, "layer_taper": 0.5,
        "activation": "relu", "dropout": 0.2, "l2": 1e-4,
        "learning_rate": 1e-3, "batch_normalization": True,
    })
    assert tapered["hidden_units"] == [512, 256, 128]
    flat = _mlp_options({
        "n_hidden_layers": 2, "first_layer_units": 128, "layer_taper": 1.0,
        "activation": "gelu", "dropout": 0.0, "l2": 0.0,
        "learning_rate": 1e-3, "batch_normalization": False,
    })
    assert flat["hidden_units"] == [128, 128]
    # A taper cannot shrink a layer below something trainable.
    narrow = _mlp_options({
        "n_hidden_layers": 4, "first_layer_units": 64, "layer_taper": 0.5,
        "activation": "relu", "dropout": 0.1, "l2": 0.0,
        "learning_rate": 1e-3, "batch_normalization": True,
    })
    assert min(narrow["hidden_units"]) >= 16


def test_the_cnn_search_widens_its_channels_by_block():
    from hamlet.training.tuning import _cnn_options

    options = _cnn_options({
        "n_conv_blocks": 3, "first_filters": 16, "kernel_size": 5,
        "dense_units": 128, "dropout": 0.1, "l2": 1e-4,
        "learning_rate": 1e-3, "batch_normalization": True,
    })
    assert options["filters"] == [16, 32, 64]
    assert options["dense_units"] == [128]


def test_every_dimension_of_every_space_is_describable():
    """The form renders what a search would vary, so it has to be sayable."""
    described = describe_search_spaces()
    for name, space in described["spaces"].items():
        assert space["dimensions"], name
        for dimension in space["dimensions"]:
            assert dimension["label"] and "_" not in dimension["label"], dimension
            if dimension["kind"] == "categorical":
                assert dimension["choices"]
            else:
                assert dimension["low"] < dimension["high"]


def test_categorical_choices_survive_a_round_trip_through_json():
    """They are written into tuning.json, and None is a real choice there."""
    import json as _json

    described = describe_search_spaces()
    restored = _json.loads(_json.dumps(described))
    forest = {d["name"]: d for d in restored["spaces"]["random_forest"]["dimensions"]}
    assert None in forest["max_features"]["choices"]
