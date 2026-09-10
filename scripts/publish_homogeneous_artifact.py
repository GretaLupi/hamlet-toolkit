#!/usr/bin/env python3
"""Train and document a distributable homogeneous-chain artifact.

Produces the pieces a reference model needs before anyone else can responsibly
reuse it:

* the artifact itself, trained with a canonical preset (``standard`` or
  ``research``, never ``quick`` -- the advisor treats quick artifacts as
  development-only and refuses them for real inference);
* a MODEL_CARD.md recording the training dataset, the sampled parameter
  ranges, the observable and preprocessing contract, the measured accuracy,
  and the conditions under which reuse is *not* valid;
* a round-trip check that reloads the saved artifact and reproduces its
  predictions, so a broken save cannot be published.

Usage
-----
    python scripts/publish_homogeneous_artifact.py \
        --dataset results/triton_n3000/homogeneous_heisenberg_l8_ed.npz \
        --model ridge --preset standard \
        --benchmark results/triton_n3000/benchmark/..._benchmark.json \
        --output-dir models/homogeneous_heisenberg_l8_ridge_standard_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from hamlet.data import SpectroscopyDataset, as_supervised
from hamlet.training import (
    TrainingPreprocessingConfig,
    grouped_split,
    prepare_training_dataset,
    train_supervised,
)
from hamlet.training.guided import TrainingRun, get_training_preset

MODEL_OPTIONS = {
    "ridge": {"alpha": 1e-3},
    # 200 trees rather than the benchmark's 600: measured at 3000 chains the
    # validation MAE is the same (0.114 vs 0.113 meV) while the saved artifact
    # is a third of the size, and a distributable reference model has to be
    # small enough to ship.
    "random_forest": {"n_estimators": 200, "min_samples_leaf": 2, "n_jobs": -1},
    "keras_mlp": {},
    "keras_cnn": {},
}


def parameter_ranges(dataset: SpectroscopyDataset) -> list[dict]:
    lo = dataset.targets_mev.min(axis=0)
    hi = dataset.targets_mev.max(axis=0)
    return [
        {"parameter": name, "min_mev": float(a), "max_mev": float(b)}
        for name, a, b in zip(dataset.target_names, lo, hi)
    ]


def render_model_card(info: dict) -> str:
    d = info
    lines: list[str] = []
    add = lines.append
    add(f"# Model card — `{d['artifact_name']}`")
    add("")
    add(d["headline"])
    add("")
    add("## What it does")
    add("")
    add(
        f"Takes one complete `({d['n_sites']} sites x {d['output_points']} bias points)` "
        f"dI/dV map of {d['system_description']} and returns the global exchange "
        f"parameters `{', '.join(d['target_names'])}` in meV."
    )
    add("")
    add("This is a **global-view** model: the whole site-by-bias map is a single input.")
    add(
        f"It is therefore valid only for chains of exactly **L = {d['n_sites']}**. A "
        "different chain length needs a model trained for that length, and the package "
        "refuses the mismatch rather than padding or cropping."
    )
    add("")
    add("## Training data")
    add("")
    add(f"- source dataset: `{d['dataset_name']}` ({d['n_chains']} simulated chains)")
    add(f"- simulator: DMRGPy, `dynamics_mode=ED` (exact diagonalisation, exact at L={d['n_sites']})")
    add(f"- observable contract: `{d['observable']}`")
    add(f"- output quantity: `{d['output_quantity']}`")
    add(f"- bias window: 0 to {d['bias_cutoff_mev']:g} meV over {d['output_points']} points")
    add(f"- broadening: {d['broadening_mev']} meV")
    for condition in d.get("known_conditions", []):
        add(f"- {condition}")
    add("")
    add("Sampled parameter ranges — **predictions outside these ranges are extrapolation**:")
    add("")
    add("| parameter | min [meV] | max [meV] |")
    add("|---|---|---|")
    for row in d["parameter_ranges"]:
        add(f"| `{row['parameter']}` | {row['min_mev']:.2f} | {row['max_mev']:.2f} |")
    add("")
    add("## Accuracy")
    add("")
    add(
        f"Model `{d['model']}`, preset `{d['preset']}` "
        f"({d['n_seeds']} seeds), aggregation `{d['aggregation']}`. "
        f"Grouped split by simulated chain, so no chain appears in more than one "
        f"partition."
    )
    add("")
    add(f"- validation MAE: **{d['validation_mae_mev']:.3f} meV**")
    add(f"- held-out test MAE: **{d['test_mae_mev']:.3f} meV**")
    add(f"- held-out test RMSE: {d['test_rmse_mev']:.3f} meV")
    add(f"- correlation fidelity: {d['test_correlation_fidelity']:.3f}")
    add(f"- split sizes: {d['split_sizes']}")
    add("")
    if d.get("per_parameter"):
        add("Per parameter on the held-out split, with skill against a training-mean")
        add("baseline (`1 - MAE_model / MAE_mean`):")
        add("")
        add("| parameter | test MAE [meV] | skill |")
        add("|---|---|---|")
        for row in d["per_parameter"]:
            add(f"| `{row['parameter']}` | {row['test_mae_mev']:.3f} | {row['skill']:.2f} |")
        add(f"Model options: `{d['model_options']}`")
    add("")
    if d.get("benchmark_note"):
        add(d["benchmark_note"])
        add("")
    add("## When NOT to reuse this model")
    add("")
    add(f"- the chain is not L = {d['n_sites']}")
    add(f"- the measurement does not cover 0 to {d['bias_cutoff_mev']:g} meV")
    add(f"- the experiment is not the `{d['system_type']}` system, or not the `global` view")
    add(f"- the simulated observable expected by the experiment is not `{d['observable']}`")
    add("- the couplings are expected outside the sampled ranges above")
    if d.get("known_conditions"):
        add("- the impurity count, sites, species, anisotropies, or field differ from the fixed training conditions above")
    add("")
    if d.get("known_conditions"):
        add(
            "`hamlet advise` checks the stored system, view, chain length, cutoff, "
            "observable, and parameter-range contracts. The user must also verify "
            "that the experiment's fixed impurity and field conditions exactly "
            "match those listed above."
        )
    else:
        add(
            "`hamlet advise` checks every one of these against the stored manifest "
            "before inference, and reports `use_existing_model` only when all of "
            "them hold."
        )
    add("")
    add("## Honest limits")
    add("")
    add(
        "The quoted accuracy is against **simulated** spectra drawn from the same "
        "generator that produced the training set. It says nothing about whether the "
        "simulator describes any particular real material, and it is not a calibrated "
        "uncertainty: ensemble spread reported at inference measures agreement between "
        "seeds, not distance from truth."
    )
    if d["model"] == "ridge":
        add("")
        add(
            "Ridge regression is deterministic, so the saved seed members are "
            "identical and their ensemble spread is zero. Use the resampled-split "
            "benchmark—not ensemble disagreement—as the stability estimate for "
            "this model."
        )
    add("")
    add("## Reproducing")
    add("")
    add("```bash")
    add("python scripts/publish_homogeneous_artifact.py \\")
    add(f"  --dataset {d['dataset_name']} \\")
    add(f"  --model {d['model']} --preset {d['preset']} \\")
    add(f"  --output-dir {d['artifact_relpath']}")
    add("```")
    add("")
    add(
        f"Generated {d['created']} with hamlet {d['hamlet_version']} on "
        f"Python {d['python_version']}."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="ridge", choices=sorted(MODEL_OPTIONS))
    parser.add_argument("--preset", default="standard", choices=["standard", "research"])
    parser.add_argument("--view", default="global")
    parser.add_argument("--cutoff-mev", type=float, default=60.0)
    parser.add_argument("--output-points", type=int, default=61)
    parser.add_argument("--benchmark", help="benchmark JSON to cite in the model card")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import hamlet

    dataset_path = Path(args.dataset)
    dataset = SpectroscopyDataset.load(dataset_path)
    print(
        f"{dataset_path.name}: {dataset.n_samples} chains, L={dataset.n_sites}, "
        f"targets {', '.join(dataset.target_names)}",
        flush=True,
    )

    prepared = prepare_training_dataset(
        dataset,
        TrainingPreprocessingConfig(
            bias_cutoff_mev=args.cutoff_mev, output_points=args.output_points
        ),
    )
    preset = get_training_preset(args.preset)
    run = train_supervised(
        prepared,
        view=args.view,
        model=args.model,
        preset=args.preset,
        model_options=MODEL_OPTIONS[args.model],
    )

    output_dir = Path(args.output_dir)
    run.save(output_dir)
    print(f"saved artifact -> {output_dir}", flush=True)

    # Reload and confirm the saved artifact reproduces its own predictions.
    supervised = as_supervised(prepared.dataset, args.view)
    split = grouped_split(
        supervised,
        validation_fraction=preset.validation_fraction,
        test_fraction=preset.test_fraction,
        seed=preset.split_seed,
    )
    reloaded = TrainingRun.load(output_dir)
    before = run.predict(split.test.inputs)
    after = reloaded.predict(split.test.inputs)
    max_drift = float(np.max(np.abs(before - after)))
    if not np.allclose(before, after, rtol=0.0, atol=1e-4):
        raise SystemExit(
            f"round trip changed predictions by up to {max_drift:.2e} meV; refusing to publish"
        )
    print(f"round-trip check OK (max drift {max_drift:.2e} meV)", flush=True)

    baseline = split.train.targets.mean(axis=0)
    per_parameter = []
    for index, name in enumerate(run.target_names):
        mae = float(np.abs(after[:, index] - split.test.targets[:, index]).mean())
        base = float(np.abs(baseline[index] - split.test.targets[:, index]).mean())
        per_parameter.append(
            {
                "parameter": name,
                "test_mae_mev": mae,
                "skill": (1.0 - mae / base) if base else float("nan"),
            }
        )

    protocol = dataset.metadata.get("protocol", {})
    recipe = dataset.metadata.get("generation_recipe", {})
    known_conditions: list[str] = []
    if dataset.system_type == "homogeneous_xxz_j1j2j3_dmi_impurity":
        impurity_records = recipe.get("impurities")
        if impurity_records:
            rendered = ", ".join(
                f"site {item['site']} ({item.get('spin', 'S=1')}, "
                f"D={item.get('axial_mev', 0):g} meV, "
                f"E={item.get('transverse_mev', 0):g} meV, "
                f"phi={item.get('transverse_angle_rad', 0):g} rad)"
                for item in impurity_records
            )
        else:
            sites = recipe.get("impurity_sites", [])
            spin = recipe.get("impurity_spin", "unknown")
            transverse = recipe.get("impurity_transverse_mev", "unknown")
            rendered = ", ".join(
                f"site {site} ({spin}, D=0 meV, E={transverse} meV, phi=0 rad)"
                for site in sites
            )
        known_conditions.append(f"fixed impurity configuration: {rendered}")
        known_conditions.append(
            f"fixed transverse field: {recipe.get('transverse_field_mev', 0):g} meV"
        )
    benchmark_note = ""
    if args.benchmark:
        bench = json.loads(Path(args.benchmark).read_text())
        others = ", ".join(
            f"`{m['model']}` {m['validation_mae_mev_mean']:.3f}"
            for m in bench.get("models", [])
        )
        benchmark_note = (
            f"Chosen by `scripts/benchmark_homogeneous.py` over "
            f"{len(bench.get('split_seeds', []))} resampled splits on validation MAE "
            f"(mean validation MAE by model: {others}). See "
            f"`{Path(args.benchmark).name}`."
        )

    info = {
        "artifact_name": output_dir.name,
        "artifact_relpath": str(output_dir),
        "headline": (
            f"Reference global-view model for the `{dataset.system_type}` system at "
            f"L = {dataset.n_sites}, trained on {dataset.n_samples} exact-diagonalisation "
            f"chains."
        ),
        "dataset_name": dataset_path.name,
        "system_type": dataset.system_type,
        "system_description": (
            "a homogeneous spin-1/2 host chain with substituted site impurities"
            if dataset.system_type == "homogeneous_xxz_j1j2j3_dmi_impurity"
            else "a homogeneous spin-1/2 chain"
        ),
        "known_conditions": known_conditions,
        "n_chains": int(dataset.n_samples),
        "n_sites": int(dataset.n_sites),
        "target_names": list(dataset.target_names),
        "parameter_ranges": parameter_ranges(dataset),
        "observable": protocol.get("observable", "unknown"),
        "output_quantity": protocol.get("output_quantity", "unknown"),
        "broadening_mev": protocol.get("broadening_mev", "unknown"),
        "bias_cutoff_mev": args.cutoff_mev,
        "output_points": args.output_points,
        "model": args.model,
        "model_options": MODEL_OPTIONS[args.model],
        "preset": args.preset,
        "n_seeds": len(preset.seeds),
        "aggregation": run.aggregation.method,
        "validation_mae_mev": run.metrics["validation"]["ensemble"]["mae"],
        "test_mae_mev": run.metrics["test"]["ensemble"]["mae"],
        "test_rmse_mev": run.metrics["test"]["ensemble"]["rmse"],
        "test_correlation_fidelity": run.metrics["test"]["ensemble"]["correlation_fidelity"],
        "split_sizes": run.metrics["split"],
        "per_parameter": per_parameter,
        "benchmark_note": benchmark_note,
        "created": time.strftime("%Y-%m-%d"),
        "hamlet_version": hamlet.__version__,
        "python_version": platform.python_version(),
    }

    if args.benchmark:
        import shutil

        benchmark_source = Path(args.benchmark)
        for suffix in (".json", ".md"):
            candidate = benchmark_source.with_suffix(suffix)
            if candidate.exists():
                shutil.copy2(candidate, output_dir / candidate.name)
        print("copied the benchmark report next to the artifact", flush=True)

    card_path = output_dir / "MODEL_CARD.md"
    card_path.write_text(render_model_card(info), encoding="utf-8")
    (output_dir / "publish_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"wrote {card_path}", flush=True)

    size = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())
    print(f"artifact size: {size / 1e6:.2f} MB", flush=True)
    print(
        f"\nvalidation MAE {info['validation_mae_mev']:.3f} | "
        f"test MAE {info['test_mae_mev']:.3f} meV",
        flush=True,
    )
    for row in per_parameter:
        print(f"  {row['parameter']:16s} MAE {row['test_mae_mev']:7.3f}  skill {row['skill']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
