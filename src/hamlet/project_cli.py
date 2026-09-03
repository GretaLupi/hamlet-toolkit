"""Command-line entry point for complete HamLeT projects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .project import HamiltonianLearningProject, ProjectConfig, ProjectPlan


def _format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _render_plan(plan: ProjectPlan, *, command: str) -> str:
    lines = [
        f"DRY RUN — `hamlet {command}` would do the following. Nothing was written.",
        "",
        f"project : {plan.name}",
        f"system  : {plan.system_type}   view: {plan.view}",
        "",
        "Stages",
    ]
    for index, stage in enumerate(plan.stages, start=1):
        lines.append(f"  {index}. {stage}")
    if not plan.stages:
        lines.append("  (nothing to do)")

    lines += ["", f"Dataset source: {plan.dataset_source}"]
    for key, value in plan.dataset_detail.items():
        lines.append(f"  {key}: {value}")

    if plan.generation_chains is not None:
        lines += [
            "",
            "Compute budget",
            f"  chains to simulate    : {plan.generation_chains}",
            f"  assumed cost per chain: {plan.seconds_per_chain:.0f} s",
            f"  serial estimate       : "
            f"{_format_duration(plan.estimated_generation_seconds)}",
            "  This is an order-of-magnitude anchor from the project's own L=8 ED runs.",
            "  Pass --seconds-per-chain with a local measurement for a real estimate;",
            "  generation is checkpointed, so it can be resumed rather than restarted.",
        ]

    lines += ["", "Outputs"]
    for item in plan.outputs:
        if item.blocks_run:
            marker = "REFUSES"
        elif item.exists:
            marker = "exists "
        else:
            marker = "new    "
        lines.append(f"  [{marker}] {item.path}")
        lines.append(f"            {item.description}")

    if plan.notes:
        lines += ["", "Notes"]
        lines += [f"  - {note}" for note in plan.notes]

    lines.append("")
    if plan.blocking_issues:
        lines.append("WOULD NOT RUN — configuration problems:")
        lines += [f"  - {issue}" for issue in plan.blocking_issues]
    elif plan.would_refuse:
        lines.append(
            "WOULD NOT COMPLETE — existing outputs marked REFUSES above would stop the "
            "run. Choose a new output_dir to keep the previous results."
        )
    else:
        lines.append("Ready to run.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hamlet",
        description="Run HamLeT — Hamiltonian Learning Toolkit workflows.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("modes", help="list registered physical experiment modes")
    generate = commands.add_parser(
        "generate", help="generate or resume the configured simulation dataset"
    )
    generate.add_argument("config", help="YAML or JSON project configuration")
    generate.add_argument(
        "--dry-run",
        action="store_true",
        help="report the simulation budget and outputs without generating anything",
    )
    generate.add_argument(
        "--seconds-per-chain",
        type=float,
        help="measured simulation cost per chain, for the --dry-run budget estimate",
    )
    generate.add_argument(
        "--plan-json",
        help="with --dry-run, also write the machine-readable plan to this path",
    )
    run = commands.add_parser("run", help="calibrate, train, infer, and report")
    run.add_argument("config", help="YAML or JSON project configuration")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="report the stages, compute budget, and outputs without running anything",
    )
    run.add_argument(
        "--seconds-per-chain",
        type=float,
        help="measured simulation cost per chain, for the --dry-run budget estimate",
    )
    run.add_argument(
        "--plan-json",
        help="with --dry-run, also write the machine-readable plan to this path",
    )
    inspect = commands.add_parser("inspect", help="inspect an experiment without training")
    inspect.add_argument("config", help="YAML or JSON project configuration")
    import_command = commands.add_parser(
        "import-measurement",
        help="convert one text/DAT file per site into canonical CSV and NPZ",
    )
    import_command.add_argument("recipe", help="YAML or JSON measurement-import recipe")
    inspect_experiment = commands.add_parser(
        "inspect-experiment",
        help="import raw per-site files and inspect them for a selected physics mode",
    )
    inspect_experiment.add_argument("recipe", help="YAML or JSON measurement-import recipe")
    inspect_experiment.add_argument(
        "--mode", required=True, help="physical mode, currently: heisenberg"
    )
    inspect_experiment.add_argument(
        "--variant",
        required=True,
        help="mode variant, currently: inhomogeneous or homogeneous",
    )
    inspect_experiment.add_argument(
        "--output-dir",
        help="override the recipe output directory with a self-contained experiment project",
    )
    inspect_experiment.add_argument(
        "--candidate-cutoffs",
        nargs="+",
        type=float,
        default=[30.0, 40.0, 50.0, 70.0, 100.0],
        metavar="MEV",
        help="cutoffs to show as covered/not covered; these are not automatic recommendations",
    )
    inspect_experiment.add_argument("--overwrite", action="store_true")
    select_mode = commands.add_parser(
        "select-experiment-mode",
        help="choose a physics interpretation for an already inspected experiment",
    )
    select_mode.add_argument("experiment", help="existing experiment manifest")
    select_mode.add_argument("--mode", required=True)
    select_mode.add_argument("--variant", required=True)
    select_mode.add_argument("--output-dir", required=True)
    select_mode.add_argument("--overwrite", action="store_true")
    advise = commands.add_parser(
        "advise",
        help="decide whether to reuse a model, retrain, regenerate, or fix the experiment",
    )
    advise.add_argument(
        "experiment", help="experiment manifest, canonical measurement NPZ, or spectroscopy CSV"
    )
    advise.add_argument("--cutoff", type=float, required=True, help="manually chosen cutoff [meV]")
    advise.add_argument("--artifact-root", action="append", default=[], help="artifact or bank directory; repeatable")
    advise.add_argument("--dataset", action="append", default=[], help="portable dataset NPZ; repeatable")
    advise.add_argument(
        "--system", help="override for legacy CSV input; manifests already declare the system"
    )
    advise.add_argument(
        "--view", choices=["local_bonds", "global"], help="override for legacy CSV input"
    )
    advise.add_argument("--output-dir", default="workflow-decision")
    advise.add_argument("--allow-development-artifacts", action="store_true")
    advise.add_argument("--max-validation-mae", type=float, help="optional artifact validation-MAE limit [meV]")
    advise.add_argument("--max-test-mae", type=float, help="optional artifact held-out test-MAE limit [meV]")
    advise.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "modes":
        from .experiments import available_experiment_modes, resolve_experiment_mode

        for mode, variants in available_experiment_modes().items():
            for variant in variants:
                profile = resolve_experiment_mode(mode, variant)
                support = (
                    "experimental workflow available"
                    if profile.experimental_inference_supported
                    else "inspection/training only; analyzer pending"
                )
                print(
                    f"{mode}/{variant}: {profile.system_type}, {profile.view}, "
                    f"observable={profile.recommended_observable}; {support}"
                )
        return 0
    if args.command == "inspect-experiment":
        from .experiments import inspect_experiment_recipe

        result = inspect_experiment_recipe(
            args.recipe,
            mode=args.mode,
            variant=args.variant,
            output_dir=args.output_dir,
            candidate_cutoffs_mev=args.candidate_cutoffs,
            overwrite=args.overwrite,
        )
        print(f"experiment status: {result.status}")
        print(
            f"mode: {result.profile.mode}/{result.profile.variant}; "
            f"system: {result.profile.system_type}; view: {result.profile.view}"
        )
        available = result.manifest["available_candidate_cutoffs_mev"]
        print(
            "covered candidate cutoffs: "
            + (", ".join(f"{value:g} meV" for value in available) if available else "none")
        )
        print(f"manifest: {result.manifest_path}")
        print(f"inspection: {result.inspection_path}")
        print("choose the cutoff manually, then run `hamlet advise MANIFEST --cutoff VALUE`")
        return 0
    if args.command == "select-experiment-mode":
        from .experiments import select_experiment_mode

        result = select_experiment_mode(
            args.experiment,
            mode=args.mode,
            variant=args.variant,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        print(
            f"selected mode: {result.profile.mode}/{result.profile.variant}; "
            f"view: {result.profile.view}"
        )
        print("canonical measurement reused without re-importing raw files")
        print(f"manifest: {result.manifest_path}")
        print(f"inspection: {result.inspection_path}")
        return 0
    if args.command == "import-measurement":
        from .io import import_text_measurement

        result = import_text_measurement(args.recipe)
        status = (
            "structurally ready for model compatibility checks"
            if result.measurement.is_primary_complete else "review missing data"
        )
        print(
            f"imported {result.measurement.shape[0]} sites and "
            f"{result.measurement.shape[1]} bias points: {status}"
        )
        print(f"channels: {', '.join(result.measurement.channels)}")
        print(f"CSV: {result.csv_path}")
        print(f"canonical measurement: {result.measurement_path}")
        print(f"import report: {result.report_path}")
        if result.preview_path is not None:
            print(f"HTML preview: {result.preview_path}")
        for warning in result.warnings:
            print(f"warning: {warning}")
        return 0
    if args.command == "advise":
        from .workflow import advise_experiment

        output = Path(args.output_dir).resolve()
        json_path = output / "workflow_decision.json"
        html_path = output / "workflow_decision.html"
        existing = [path for path in (json_path, html_path) if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                f"decision outputs already exist: {[str(path) for path in existing]}; "
                "choose a new --output-dir or pass --overwrite"
            )
        decision = advise_experiment(
            args.experiment,
            manual_cutoff_mev=args.cutoff,
            artifact_roots=args.artifact_root,
            dataset_paths=args.dataset,
            system_type=args.system,
            view=args.view,
            allow_development_artifacts=args.allow_development_artifacts,
            max_validation_mae_mev=args.max_validation_mae,
            max_test_mae_mev=args.max_test_mae,
        )
        decision.save_json(json_path)
        decision.save_html(html_path)
        print(f"decision: {decision.action}")
        print(decision.summary)
        if decision.selected_artifact is not None:
            print(f"artifact: {decision.selected_artifact}")
        if decision.selected_dataset is not None:
            print(f"dataset: {decision.selected_dataset}")
        print(f"JSON: {json_path}")
        print(f"HTML: {html_path}")
        return 0
    project = HamiltonianLearningProject(ProjectConfig.from_file(args.config))
    if getattr(args, "dry_run", False):
        plan = project.plan(seconds_per_chain=args.seconds_per_chain)
        print(_render_plan(plan, command=args.command))
        if args.plan_json:
            destination = Path(args.plan_json)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
            print(f"\nplan JSON: {destination}")
        # A plan that the real run would refuse is reported as a failure, so a
        # dry run is usable as a precondition check in a script.
        return 1 if plan.would_refuse else 0
    if args.command == "generate":
        result = project.generate_training_dataset()
        action = "reused cached" if result.cache_hit else "generated"
        print(f"{action} dataset: {result.dataset_path}")
        print(f"samples: {result.dataset.n_samples}; system: {result.dataset.system_type}")
        print(f"recipe manifest: {result.manifest_path}")
        return 0
    if args.command == "inspect":
        summary = project.inspect_experiment()
        print(
            f"experiment: {summary['n_sites']} sites, {summary['bias_min_mev']:g}–"
            f"{summary['bias_max_mev']:g} meV, {summary['n_bias_points']} points"
        )
        print(f"inspection: {project.config.output_dir / 'experiment_inspection.json'}")
        return 0

    outcome = project.run()
    print(f"selected cutoff: {outcome.selected_cutoff_mev:g} meV")
    print(f"analysis status: {outcome.status}")
    print(f"artifact: {outcome.artifact_path}")
    print(f"HTML report: {outcome.report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
