"""Command-line interface for diagnostics-first experimental inference."""

import argparse
from pathlib import Path

from .analysis import ExperimentalChainAnalyzer
from ..experiments import load_canonical_experiment
from ..preprocessing import SpectralPreprocessor
from ..training import TrainingRun
from ..workflow import advise_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hamlet-analyze",
        description="Infer local or homogeneous Heisenberg couplings from experimental spectra.",
    )
    parser.add_argument(
        "csv", help="experiment manifest, canonical measurement NPZ, or legacy spectroscopy CSV"
    )
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--models", nargs="+", help="legacy Keras ensemble artifacts")
    model_source.add_argument("--artifact", help="one portable trained artifact directory")
    model_source.add_argument(
        "--artifact-bank",
        help="artifact-bank root; automatically choose from experimental bias coverage",
    )
    parser.add_argument("--output-dir", default="analysis-result")
    parser.add_argument("--output-range", nargs=2, type=float, metavar=("LOW", "HIGH"))
    parser.add_argument("--unit", default="model units")
    parser.add_argument("--bias-max", type=float, default=50.0)
    parser.add_argument(
        "--cutoff",
        type=float,
        help="manual cutoff in meV; with --artifact-bank, require exact cutoff-specific weights",
    )
    parser.add_argument("--max-validation-mae", type=float)
    parser.add_argument("--max-test-mae", type=float)
    parser.add_argument("--allow-development-artifacts", action="store_true")
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--baseline-max", type=float, default=3.0)
    parser.add_argument("--scale-min", type=float, default=40.0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing decision and analysis files in --output-dir",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.artifact or args.artifact_bank) and args.cutoff is None:
        raise SystemExit(
            "portable artifact inference requires an explicit manual --cutoff; "
            "run `hamlet advise` first when resource compatibility is unknown"
        )
    output_dir = Path(args.output_dir)
    expected_outputs = [
        output_dir / "workflow_decision.json",
        output_dir / "workflow_decision.html",
        output_dir / "couplings.csv",
        output_dir / "report.json",
        output_dir / "report.html",
    ]
    if not args.no_plot:
        expected_outputs.append(output_dir / "summary.png")
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"analysis outputs already exist: {[str(path) for path in existing]}; "
            "choose a new --output-dir or pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_manifest = None
    if args.artifact_bank:
        decision = advise_experiment(
            args.csv,
            manual_cutoff_mev=args.cutoff,
            artifact_roots=[args.artifact_bank],
            allow_development_artifacts=args.allow_development_artifacts,
            max_validation_mae_mev=args.max_validation_mae,
            max_test_mae_mev=args.max_test_mae,
        )
        decision.save_json(output_dir / "workflow_decision.json")
        decision.save_html(output_dir / "workflow_decision.html")
        if not decision.can_use_existing_model:
            raise SystemExit(
                f"preflight decision: {decision.action}; {decision.summary}; "
                f"inspect {output_dir / 'workflow_decision.html'}"
            )
        artifact_path = decision.selected_artifact
        assert artifact_path is not None
        print(f"model recommendation: {decision.summary}")
        print(f"artifact: {artifact_path}")
        analyzer = TrainingRun.load(artifact_path).create_analyzer()
        artifact_manifest = artifact_path / "manifest.json"
    elif args.artifact:
        artifact_path = Path(args.artifact)
        if args.cutoff is not None:
            decision = advise_experiment(
                args.csv,
                manual_cutoff_mev=args.cutoff,
                artifact_roots=[artifact_path],
                allow_development_artifacts=args.allow_development_artifacts,
                max_validation_mae_mev=args.max_validation_mae,
                max_test_mae_mev=args.max_test_mae,
            )
            decision.save_json(output_dir / "workflow_decision.json")
            decision.save_html(output_dir / "workflow_decision.html")
            if not decision.can_use_existing_model:
                raise SystemExit(
                    f"preflight decision: {decision.action}; {decision.summary}; "
                    f"inspect {output_dir / 'workflow_decision.html'}"
                )
        analyzer = TrainingRun.load(artifact_path).create_analyzer()
        artifact_manifest = artifact_path / "manifest.json"
    else:
        preprocessor = SpectralPreprocessor(
            output_points=args.points,
            bias_range_mev=(0.0, args.bias_max),
            baseline_range_mev=(0.0, args.baseline_max),
            scale_range_mev=(args.scale_min, args.bias_max),
        )
        analyzer = ExperimentalChainAnalyzer.from_keras_paths(
            args.models,
            preprocessor,
            output_range=tuple(args.output_range) if args.output_range else None,
            coupling_unit=args.unit,
        )
    measurement, source, _, _ = load_canonical_experiment(args.csv)
    result = analyzer.analyze_measurement(measurement, source=source)
    result.save_couplings_csv(output_dir / "couplings.csv")
    result.save_report_json(output_dir / "report.json")
    result.save_html_report(
        output_dir / "report.html", artifact_manifest=artifact_manifest
    )
    if not args.no_plot:
        figure = result.plot_summary()
        figure.savefig(output_dir / "summary.png", dpi=160)
        try:
            import matplotlib.pyplot as plt

            plt.close(figure)
        except ImportError:  # pragma: no cover
            pass
    print(f"status: {result.diagnostics.status}")
    count_label = "parameters" if hasattr(result, "n_parameters") else "bonds"
    print(f"sites: {result.n_sites}; {count_label}: {result.n_bonds}")
    print(f"results: {output_dir.resolve()}")
    for warning in result.diagnostics.warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
