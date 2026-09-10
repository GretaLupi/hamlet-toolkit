#!/usr/bin/env python3
"""Controlled model comparison for the homogeneous (global-view) chain models.

Why this exists
---------------
A single grouped split gives per-parameter numbers that swing wildly at these
dataset sizes: in the L=8 pilots ``J3`` came out as the best-recovered
parameter on one dataset and worse than the training mean on another, purely
from which chains landed in the 8-chain test set. Quoting one split is
therefore not evidence about which couplings are identifiable.

This script repeats the whole train/evaluate cycle over several *split* seeds
and reports the mean and spread of each per-parameter score, so a claim like
"this mode recovers J3" has to survive resampling before it is made. Model
selection uses validation scores only; the test partition is read once per
split for reporting and never used to choose anything.

Usage
-----
    python scripts/benchmark_homogeneous.py \
        --dataset /path/to/homogeneous_heisenberg_l8_ed.npz \
        --models ridge random_forest keras_mlp \
        --split-seeds 42 43 44 45 46 \
        --output results/benchmark_homogeneous

Unavailable model backends (no TensorFlow, say) are skipped with a note rather
than aborting the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from hamlet.data import SpectroscopyDataset, as_supervised
from hamlet.training import (
    TrainingPreprocessingConfig,
    grouped_split,
    prepare_training_dataset,
    train_supervised,
)
from hamlet.training.guided import TrainingPreset

DEFAULT_MODELS = ("ridge", "random_forest", "keras_mlp", "keras_cnn")
DEFAULT_SPLIT_SEEDS = (42, 43, 44, 45, 46)

MODEL_OPTIONS = {
    "ridge": {"alpha": 1e-3},
    "random_forest": {"n_estimators": 600, "min_samples_leaf": 2, "n_jobs": -1},
    "keras_mlp": {},
    "keras_cnn": {},
}

# Verdict thresholds applied to (mean skill - std skill) across splits, i.e. a
# parameter must clear the bar even at the pessimistic end of its spread.
RESOLVED_SKILL = 0.5
MARGINAL_SKILL = 0.2


def build_preset(name: str, model_seeds: tuple[int, ...], split_seed: int) -> TrainingPreset:
    return TrainingPreset(
        name,
        seeds=model_seeds,
        epochs=200,
        batch_size=32,
        patience=20,
        validation_fraction=0.2,
        test_fraction=0.2,
        split_seed=split_seed,
    )


def score_split(run, split, names: tuple[str, ...]) -> list[dict]:
    """Per-parameter test MAE and skill against a training-mean baseline."""
    predicted = run.predict(split.test.inputs)
    baseline = split.train.targets.mean(axis=0)
    rows = []
    for index, name in enumerate(names):
        model_mae = float(np.abs(predicted[:, index] - split.test.targets[:, index]).mean())
        base_mae = float(np.abs(baseline[index] - split.test.targets[:, index]).mean())
        rows.append(
            {
                "parameter": name,
                "test_mae_mev": model_mae,
                "baseline_mae_mev": base_mae,
                "skill": (1.0 - model_mae / base_mae) if base_mae else float("nan"),
            }
        )
    return rows


def parameter_count(run) -> int | None:
    """Trainable parameters for neural members; None for the classical models."""
    total = 0
    for model in run.models:
        counter = getattr(model, "count_params", None)
        if counter is None:
            return None
        total += int(counter())
    return total


def verdict(low_skill: float) -> str:
    if low_skill >= RESOLVED_SKILL:
        return "resolved"
    if low_skill >= MARGINAL_SKILL:
        return "marginal"
    return "not resolved"


def benchmark_model(
    prepared,
    view: str,
    model: str,
    split_seeds: tuple[int, ...],
    model_seeds: tuple[int, ...],
) -> dict:
    per_split: list[dict] = []
    for split_seed in split_seeds:
        preset = build_preset("benchmark", model_seeds, split_seed)
        started = time.time()
        run = train_supervised(
            prepared,
            view=view,
            model=model,
            preset=preset,
            model_options=MODEL_OPTIONS.get(model, {}),
        )
        train_seconds = time.time() - started

        supervised = as_supervised(prepared.dataset, view)
        split = grouped_split(
            supervised,
            validation_fraction=preset.validation_fraction,
            test_fraction=preset.test_fraction,
            seed=split_seed,
        )

        inference_started = time.time()
        run.predict(split.test.inputs)
        inference_seconds = time.time() - inference_started

        per_split.append(
            {
                "split_seed": split_seed,
                "validation_mae_mev": run.metrics["validation"]["ensemble"]["mae"],
                "test_mae_mev": run.metrics["test"]["ensemble"]["mae"],
                "test_rmse_mev": run.metrics["test"]["ensemble"]["rmse"],
                "test_correlation_fidelity": run.metrics["test"]["ensemble"][
                    "correlation_fidelity"
                ],
                "aggregation": run.aggregation.method,
                "train_seconds": round(train_seconds, 2),
                "inference_seconds": round(inference_seconds, 4),
                "per_parameter": score_split(run, split, run.target_names),
                "n_test_chains": int(np.unique(split.test.group_ids).size),
            }
        )
        print(
            f"    split {split_seed}: validation {per_split[-1]['validation_mae_mev']:.3f}"
            f" | test {per_split[-1]['test_mae_mev']:.3f} meV"
            f" ({train_seconds:.0f}s)",
            flush=True,
        )
        last_run = run

    names = list(last_run.target_names)
    aggregated = []
    for index, name in enumerate(names):
        maes = np.array([s["per_parameter"][index]["test_mae_mev"] for s in per_split])
        skills = np.array([s["per_parameter"][index]["skill"] for s in per_split])
        low = float(skills.mean() - skills.std(ddof=0))
        aggregated.append(
            {
                "parameter": name,
                "test_mae_mev_mean": float(maes.mean()),
                "test_mae_mev_std": float(maes.std(ddof=0)),
                "skill_mean": float(skills.mean()),
                "skill_std": float(skills.std(ddof=0)),
                "skill_lower": low,
                "verdict": verdict(low),
            }
        )

    validation = np.array([s["validation_mae_mev"] for s in per_split])
    test = np.array([s["test_mae_mev"] for s in per_split])
    return {
        "model": model,
        "n_splits": len(per_split),
        "validation_mae_mev_mean": float(validation.mean()),
        "validation_mae_mev_std": float(validation.std(ddof=0)),
        "test_mae_mev_mean": float(test.mean()),
        "test_mae_mev_std": float(test.std(ddof=0)),
        "train_seconds_mean": float(np.mean([s["train_seconds"] for s in per_split])),
        "inference_seconds_mean": float(np.mean([s["inference_seconds"] for s in per_split])),
        "ensemble_parameters": parameter_count(last_run),
        "per_parameter": aggregated,
        "splits": per_split,
    }


def render_report(summary: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# Homogeneous global-view benchmark — {summary['dataset_name']}")
    add("")
    add(f"- system: `{summary['system_type']}`  view: `{summary['view']}`")
    add(f"- chains: {summary['n_chains']}  sites: {summary['n_sites']}")
    add(f"- targets: {', '.join(summary['target_names'])}")
    add(f"- bias cutoff: {summary['bias_cutoff_mev']:g} meV over {summary['output_points']} points")
    add(f"- split seeds: {summary['split_seeds']}  model seeds: {summary['model_seeds']}")
    add("")
    add("Scores are the mean +/- spread over the split seeds. Selection uses the")
    add("validation column only; the test column is reported, never optimised against.")
    add("")
    add("## Overall")
    add("")
    add("| model | validation MAE | test MAE | train s | infer s | ens. params |")
    add("|---|---|---|---|---|---|")
    for entry in summary["models"]:
        params = entry["ensemble_parameters"]
        add(
            f"| `{entry['model']}` "
            f"| {entry['validation_mae_mev_mean']:.3f} +/- {entry['validation_mae_mev_std']:.3f} "
            f"| {entry['test_mae_mev_mean']:.3f} +/- {entry['test_mae_mev_std']:.3f} "
            f"| {entry['train_seconds_mean']:.0f} "
            f"| {entry['inference_seconds_mean']:.3f} "
            f"| {params if params is not None else 'n/a'} |"
        )
    add("")
    if summary["selected_model"]:
        add(f"**Selected on validation MAE: `{summary['selected_model']}`**")
        add("")
    add("## Per parameter")
    add("")
    add("`skill` is 1 - MAE_model / MAE_training_mean. `lower` is mean - spread, and")
    add(f"the verdict needs lower >= {RESOLVED_SKILL} to read `resolved`, >= {MARGINAL_SKILL}")
    add("to read `marginal`. A parameter that cannot clear the pessimistic end of its")
    add("own spread has not been shown to be identifiable.")
    add("")
    for entry in summary["models"]:
        add(f"### `{entry['model']}`")
        add("")
        add("| parameter | test MAE [meV] | skill | lower | verdict |")
        add("|---|---|---|---|---|")
        for row in entry["per_parameter"]:
            add(
                f"| `{row['parameter']}` "
                f"| {row['test_mae_mev_mean']:.3f} +/- {row['test_mae_mev_std']:.3f} "
                f"| {row['skill_mean']:.2f} +/- {row['skill_std']:.2f} "
                f"| {row['skill_lower']:.2f} "
                f"| {row['verdict']} |"
            )
        add("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="merged homogeneous dataset .npz")
    parser.add_argument("--view", default="global")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--split-seeds", nargs="+", type=int, default=list(DEFAULT_SPLIT_SEEDS)
    )
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--cutoff-mev", type=float, default=60.0)
    parser.add_argument("--output-points", type=int, default=61)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset = SpectroscopyDataset.load(dataset_path)
    print(
        f"{dataset_path.name}: {dataset.n_samples} chains, {dataset.n_sites} sites, "
        f"system {dataset.system_type}, targets {', '.join(dataset.target_names)}",
        flush=True,
    )

    prepared = prepare_training_dataset(
        dataset,
        TrainingPreprocessingConfig(
            bias_cutoff_mev=args.cutoff_mev, output_points=args.output_points
        ),
    )

    entries: list[dict] = []
    skipped: dict[str, str] = {}
    for model in args.models:
        print(f"\n{model}:", flush=True)
        try:
            entries.append(
                benchmark_model(
                    prepared,
                    args.view,
                    model,
                    tuple(args.split_seeds),
                    tuple(args.model_seeds),
                )
            )
        except ImportError as exc:
            skipped[model] = str(exc)
            print(f"    skipped: {exc}", flush=True)

    entries.sort(key=lambda item: item["validation_mae_mev_mean"])

    summary = {
        "dataset": str(dataset_path),
        "dataset_name": dataset_path.name,
        "system_type": dataset.system_type,
        "view": args.view,
        "n_chains": int(dataset.n_samples),
        "n_sites": int(dataset.n_sites),
        "target_names": list(dataset.target_names),
        "bias_cutoff_mev": args.cutoff_mev,
        "output_points": args.output_points,
        "split_seeds": list(args.split_seeds),
        "model_seeds": list(args.model_seeds),
        "selected_model": entries[0]["model"] if entries else None,
        "selection_metric": "mean validation MAE over split seeds",
        "skipped_models": skipped,
        "models": entries,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset_path.stem
    json_path = output_dir / f"{stem}_benchmark.json"
    md_path = output_dir / f"{stem}_benchmark.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(render_report(summary), encoding="utf-8")

    print("\n" + render_report(summary))
    print(f"\nwrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
