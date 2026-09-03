"""Command-line entry point for training a controlled cutoff model bank."""

import argparse
import json
from pathlib import Path

from ..data import SpectroscopyDataset
from ..io import load_reference_heisenberg_datasets
from .calibration import AugmentationConfig
from .cutoff_bank import train_cutoff_bank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hamlet-train-cutoff-bank",
        description="Compare model architectures at 50 meV and train a cutoff bank.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="portable SpectroscopyDataset NPZ")
    source.add_argument(
        "--reference-npz",
        nargs="+",
        help="one or more original X/Z/J Heisenberg NPZ datasets",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cutoffs", nargs="+", type=float, default=[30, 40, 50, 70, 100])
    parser.add_argument("--comparison-cutoff", type=float, default=50.0)
    parser.add_argument("--candidates", nargs="+", default=["keras_mlp", "keras_cnn"])
    parser.add_argument("--preset", choices=["quick", "standard", "research"], default="research")
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument(
        "--theory",
        action="store_true",
        help="disable experimental-like broadening/noise/drift augmentation",
    )
    parser.add_argument("--augmentation-seed", type=int, default=42)
    parser.add_argument("--noise", type=float, default=0.002)
    parser.add_argument(
        "--augmentation-calibration",
        help="calibration JSON whose selected_config should be used for training",
    )
    parser.add_argument(
        "--allow-failed-calibration",
        action="store_true",
        help="use selected settings even when the calibration acceptance gate failed",
    )
    parser.add_argument("--verbose", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = (
        SpectroscopyDataset.load(args.dataset)
        if args.dataset
        else load_reference_heisenberg_datasets(args.reference_npz)
    )
    augmentation_config = None
    if args.augmentation_calibration:
        payload = json.loads(
            Path(args.augmentation_calibration).read_text(encoding="utf-8")
        )
        if not payload.get("acceptance_passed") and not args.allow_failed_calibration:
            raise SystemExit(
                "augmentation calibration did not pass; calibrate only the intended "
                "cutoff or pass --allow-failed-calibration to override"
            )
        augmentation_config = AugmentationConfig(**payload["selected_config"])
    catalog = train_cutoff_bank(
        dataset,
        args.output_dir,
        cutoffs_mev=args.cutoffs,
        comparison_cutoff_mev=args.comparison_cutoff,
        candidates=args.candidates,
        preset=args.preset,
        output_points=args.points,
        experimental_like=not args.theory,
        augmentation_seed=args.augmentation_seed,
        augmentation_noise=args.noise,
        augmentation_config=augmentation_config,
        verbose=args.verbose,
    )
    print(f"selected architecture: {catalog['selected_model']}")
    print(f"catalog: {args.output_dir}/catalog.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
