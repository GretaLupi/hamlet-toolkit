"""Command-line interface for label-free augmentation calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..data import SpectroscopyDataset
from ..io import load_reference_heisenberg_datasets, load_spectroscopy_csv
from .calibration import calibrate_augmentation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hamlet-calibrate-augmentation",
        description=(
            "Compare bounded simulation nuisance models with an experiment "
            "without using Hamiltonian labels or inverse-model predictions."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="portable SpectroscopyDataset NPZ")
    source.add_argument(
        "--reference-npz",
        nargs="+",
        help="one or more original X/Z/J Heisenberg NPZ datasets",
    )
    parser.add_argument("--experiment", required=True, help="spectroscopy CSV")
    parser.add_argument("--output", required=True, help="calibration JSON output")
    parser.add_argument("--cutoffs", nargs="+", type=float, default=[50.0, 70.0])
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--split-seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = (
        SpectroscopyDataset.load(args.dataset)
        if args.dataset
        else load_reference_heisenberg_datasets(args.reference_npz)
    )
    bias_mev, spectra = load_spectroscopy_csv(args.experiment)
    result = calibrate_augmentation(
        dataset,
        spectra,
        bias_mev,
        cutoffs_mev=args.cutoffs,
        output_points=args.points,
        split_seed=args.split_seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    print(f"selected augmentation: {result.selected_name}")
    print(f"joint acceptance passed: {result.acceptance_passed}")
    print(f"calibration: {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
