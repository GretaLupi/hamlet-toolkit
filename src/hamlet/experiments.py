"""Guided raw-measurement inspection with explicit physical-mode contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import base64
from io import BytesIO
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .branding import BRAND_NAME, FULL_NAME, brand_manifest
from .io import (
    MeasurementImportResult,
    TextImportRecipe,
    import_text_measurement,
    load_measurement_csv,
)
from .measurements import Measurement


EXPERIMENT_PROJECT_SCHEMA_VERSION = 1
DEFAULT_CUTOFF_CANDIDATES_MEV = (30.0, 40.0, 50.0, 70.0, 100.0)


@dataclass(frozen=True)
class ExperimentModeProfile:
    """Physics-aware structural contract selected before ML preprocessing."""

    mode: str
    variant: str
    system_type: str
    view: str
    required_axes: tuple[str, ...]
    primary_channel: str
    minimum_sites: int
    variable_site_count: bool
    experimental_inference_supported: bool
    recommended_observable: str
    description: str


@dataclass(frozen=True)
class ExperimentInspectionResult:
    """Files and structured result from one guided experiment inspection."""

    measurement: Measurement
    profile: ExperimentModeProfile
    import_result: MeasurementImportResult
    manifest: Mapping[str, Any]
    manifest_path: Path
    inspection_path: Path

    @property
    def status(self) -> str:
        return str(self.manifest["status"])


@dataclass(frozen=True)
class ExperimentModeSelectionResult:
    """A new physics interpretation of an already inspected measurement."""

    measurement: Measurement
    profile: ExperimentModeProfile
    manifest: Mapping[str, Any]
    manifest_path: Path
    inspection_path: Path


_MODE_PROFILES = {
    ("heisenberg", "inhomogeneous"): ExperimentModeProfile(
        mode="heisenberg",
        variant="inhomogeneous",
        system_type="inhomogeneous_heisenberg",
        view="local_bonds",
        required_axes=("site", "bias"),
        primary_channel="didv",
        minimum_sites=3,
        variable_site_count=True,
        experimental_inference_supported=True,
        recommended_observable="Sz",
        description=(
            "Bond-inhomogeneous Heisenberg chain: three adjacent site spectra "
            "predict two local exchange couplings."
        ),
    ),
    ("heisenberg", "homogeneous"): ExperimentModeProfile(
        mode="heisenberg",
        variant="homogeneous",
        system_type="homogeneous_heisenberg",
        view="global",
        required_axes=("site", "bias"),
        primary_channel="didv",
        minimum_sites=2,
        variable_site_count=False,
        experimental_inference_supported=True,
        recommended_observable="Sz",
        description=(
            "Homogeneous longer-range Heisenberg chain with global J1, J2, ... "
            "targets and a fixed experimental site count."
        ),
    ),
    ("heisenberg", "xxz_long_range"): ExperimentModeProfile(
        mode="heisenberg",
        variant="xxz_long_range",
        system_type="homogeneous_xxz_j1j2j3",
        view="global",
        required_axes=("site", "bias"),
        primary_channel="didv",
        minimum_sites=4,
        variable_site_count=False,
        experimental_inference_supported=True,
        recommended_observable="total_spin",
        description=(
            "Fixed-length homogeneous XXZ chain with transverse J1_xy, "
            "longitudinal Jz, and isotropic J2/J3 targets."
        ),
    ),
    ("heisenberg", "xxz_dmi"): ExperimentModeProfile(
        mode="heisenberg",
        variant="xxz_dmi",
        system_type="homogeneous_xxz_j1j2j3_dmi",
        view="global",
        required_axes=("site", "bias"),
        primary_channel="didv",
        minimum_sites=4,
        variable_site_count=False,
        experimental_inference_supported=True,
        recommended_observable="total_spin",
        description=(
            "Fixed-length homogeneous XXZ+J2+J3 model predicting a non-negative "
            "uniform z-axis DMI magnitude from total-spin spectra."
        ),
    ),
}


def available_experiment_modes() -> dict[str, tuple[str, ...]]:
    """Return registered user-facing modes and their variants."""
    result: dict[str, list[str]] = {}
    for mode, variant in _MODE_PROFILES:
        result.setdefault(mode, []).append(variant)
    return {mode: tuple(sorted(variants)) for mode, variants in sorted(result.items())}


def resolve_experiment_mode(mode: str, variant: str | None = None) -> ExperimentModeProfile:
    """Resolve user-facing names without leaking aliases into saved manifests."""
    normalized_mode = _normalize_name(mode)
    normalized_variant = _normalize_name(variant or "")
    combined_aliases = {
        "inhomogeneous_heisenberg": ("heisenberg", "inhomogeneous"),
        "heisenberg_inhomogeneous": ("heisenberg", "inhomogeneous"),
        "homogeneous_heisenberg": ("heisenberg", "homogeneous"),
        "heisenberg_homogeneous": ("heisenberg", "homogeneous"),
    }
    variant_aliases = {
        "inhomo": "inhomogeneous",
        "homo": "homogeneous",
        "xxz": "xxz_long_range",
        "dmi": "xxz_dmi",
    }
    if normalized_mode in combined_aliases and not normalized_variant:
        key = combined_aliases[normalized_mode]
    else:
        normalized_variant = variant_aliases.get(normalized_variant, normalized_variant)
        key = (normalized_mode, normalized_variant)
    if key not in _MODE_PROFILES:
        choices = ", ".join(
            f"{registered_mode}/{registered_variant}"
            for registered_mode, registered_variant in sorted(_MODE_PROFILES)
        )
        raise ValueError(
            f"unsupported experiment mode/variant {mode!r}/{variant!r}; available: {choices}"
        )
    return _MODE_PROFILES[key]


def inspect_experiment_recipe(
    recipe: TextImportRecipe | str | Path,
    *,
    mode: str,
    variant: str | None = None,
    output_dir: str | Path | None = None,
    candidate_cutoffs_mev: Sequence[float] = DEFAULT_CUTOFF_CANDIDATES_MEV,
    overwrite: bool = False,
) -> ExperimentInspectionResult:
    """Import raw per-site files and build a physics-aware experiment project.

    This stage deliberately retains raw canonical signals. Model-specific
    cropping, baseline subtraction, scaling, and interpolation happen only
    after an artifact/dataset compatibility decision.
    """
    profile = resolve_experiment_mode(mode, variant)
    recipe_source = str(recipe) if not isinstance(recipe, TextImportRecipe) else None
    resolved_recipe = (
        recipe if isinstance(recipe, TextImportRecipe) else TextImportRecipe.from_file(recipe)
    )
    if output_dir is not None:
        directory = Path(output_dir).resolve()
        resolved_recipe = replace(
            resolved_recipe,
            output_csv=directory / "spectroscopy.csv",
            output_measurement=directory / "measurement.npz",
            output_report=directory / "import_report.json",
            output_preview=(
                directory / "import_preview.html"
                if resolved_recipe.output_preview is not None
                else None
            ),
            overwrite=overwrite,
        )
    elif overwrite and not resolved_recipe.overwrite:
        resolved_recipe = replace(resolved_recipe, overwrite=True)

    project_dir = resolved_recipe.output_report.parent
    manifest_path = project_dir / "experiment_manifest.json"
    inspection_path = project_dir / "experiment_inspection.html"
    existing = [path for path in (manifest_path, inspection_path) if path.exists()]
    if existing and not resolved_recipe.overwrite:
        raise FileExistsError(
            f"experiment-project outputs already exist: {[str(path) for path in existing]}; "
            "choose a new --output-dir or pass --overwrite"
        )

    imported = import_text_measurement(resolved_recipe)
    measurement = _with_mode_metadata(imported.measurement, profile, recipe_source)
    # The guided canonical NPZ carries the selected physics profile. CSV remains
    # a simple interoperability view and never becomes the provenance authority.
    measurement.save(resolved_recipe.output_measurement)

    checks, warnings, site_statistics = _inspect_measurement(measurement, profile)
    structural_pass = all(item["status"] == "pass" for item in checks)
    cutoff_rows = _cutoff_coverage(
        np.asarray(measurement.axes.get("bias", []), dtype=float),
        candidate_cutoffs_mev,
    )
    available_cutoffs = [item["cutoff_mev"] for item in cutoff_rows if item["covered"]]
    if not available_cutoffs:
        warnings.append(
            "none of the requested cutoff candidates is covered; inspect the measured "
            "positive-bias range and supply laboratory-appropriate candidates"
        )
    if not profile.experimental_inference_supported:
        warnings.append(
            "this mode can be imported and trained, but its experimental analyzer/report "
            "is not implemented in the current limited release"
        )

    bias = np.asarray(measurement.axes.get("bias", []), dtype=float)
    manifest: dict[str, Any] = {
        "experiment_project_schema_version": EXPERIMENT_PROJECT_SCHEMA_VERSION,
        "toolkit": brand_manifest(),
        "status": "ready_for_cutoff_selection" if structural_pass else "review_required",
        "mode": profile.mode,
        "variant": profile.variant,
        "system_type": profile.system_type,
        "view": profile.view,
        "simulation_observable": profile.recommended_observable,
        "profile": asdict(profile),
        "selected_cutoff_mev": None,
        "inspection_stage": "raw_canonical_before_model_preprocessing",
        "measurement": {
            "shape": list(measurement.shape),
            "axes": list(measurement.axes),
            "channels": list(measurement.channels),
            "primary_channel": measurement.primary_channel,
            "channel_roles": {
                name: "inference" if name == measurement.primary_channel else "plot_qc_only"
                for name in measurement.channels
            },
            "bias_min_mev": float(np.min(bias)) if bias.size else None,
            "bias_max_mev": float(np.max(bias)) if bias.size else None,
            "n_sites": int(measurement.shape[0]) if measurement.shape else 0,
            "n_bias_points": int(bias.size),
        },
        "checks": checks,
        "warnings": [*imported.warnings, *warnings],
        "candidate_cutoffs": cutoff_rows,
        "available_candidate_cutoffs_mev": available_cutoffs,
        "site_statistics": site_statistics,
        "outputs": {
            "csv": str(resolved_recipe.output_csv),
            "measurement": str(resolved_recipe.output_measurement),
            "import_report": str(resolved_recipe.output_report),
            "import_preview": (
                str(resolved_recipe.output_preview)
                if resolved_recipe.output_preview is not None
                else None
            ),
            "inspection_html": str(inspection_path),
        },
        "next_step": (
            "Inspect the signals, choose one common covered cutoff manually, then run "
            f"hamlet advise {manifest_path} --cutoff CUTOFF_MEV"
        ),
    }
    project_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_experiment_inspection(inspection_path, measurement, manifest)
    return ExperimentInspectionResult(
        measurement=measurement,
        profile=profile,
        import_result=replace(imported, measurement=measurement),
        manifest=manifest,
        manifest_path=manifest_path,
        inspection_path=inspection_path,
    )


def load_experiment_project(path: str | Path) -> tuple[Measurement, Mapping[str, Any]]:
    """Load and validate a guided experiment-project manifest."""
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("experiment_project_schema_version") != EXPERIMENT_PROJECT_SCHEMA_VERSION:
        raise ValueError("unsupported experiment-project schema version")
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs.get("measurement"):
        raise ValueError("experiment manifest has no canonical measurement output")
    measurement_path = Path(str(outputs["measurement"]))
    if not measurement_path.is_absolute():
        measurement_path = (manifest_path.parent / measurement_path).resolve()
    return Measurement.load(measurement_path), payload


def select_experiment_mode(
    manifest_path: str | Path,
    *,
    mode: str,
    variant: str | None = None,
    output_dir: str | Path,
    overwrite: bool = False,
) -> ExperimentModeSelectionResult:
    """Apply a physics mode after raw inspection without re-importing site files."""
    measurement, previous = load_experiment_project(manifest_path)
    profile = resolve_experiment_mode(mode, variant)
    destination = Path(output_dir).resolve()
    new_manifest_path = destination / "experiment_manifest.json"
    inspection_path = destination / "experiment_inspection.html"
    measurement_path = destination / "measurement.npz"
    csv_path = destination / "spectroscopy.csv"
    existing = [
        path
        for path in (new_manifest_path, inspection_path, measurement_path, csv_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"mode-selection outputs already exist: {[str(path) for path in existing]}; "
            "choose a new --output-dir or pass --overwrite"
        )

    selected = _with_mode_metadata(measurement, profile, None)
    checks, warnings, statistics = _inspect_measurement(selected, profile)
    structural_pass = all(item["status"] == "pass" for item in checks)
    cutoff_rows = _cutoff_coverage(
        np.asarray(selected.axes.get("bias", []), dtype=float),
        [row["cutoff_mev"] for row in previous.get("candidate_cutoffs", [])],
    )
    available = [row["cutoff_mev"] for row in cutoff_rows if row["covered"]]
    destination.mkdir(parents=True, exist_ok=True)
    selected.save(measurement_path)
    selected.to_spectroscopy_csv(csv_path)

    payload = dict(previous)
    payload.update(
        {
            "status": "ready_for_cutoff_selection" if structural_pass else "review_required",
            "mode": profile.mode,
            "variant": profile.variant,
            "system_type": profile.system_type,
            "view": profile.view,
            "profile": asdict(profile),
            "selected_cutoff_mev": None,
            "checks": checks,
            "warnings": warnings,
            "candidate_cutoffs": cutoff_rows,
            "available_candidate_cutoffs_mev": available,
            "site_statistics": statistics,
            "mode_selection_provenance": {
                "source_manifest": str(Path(manifest_path).resolve()),
                "data_reimported": False,
            },
            "outputs": {
                **dict(previous.get("outputs", {})),
                "csv": str(csv_path),
                "measurement": str(measurement_path),
                "inspection_html": str(inspection_path),
            },
            "next_step": (
                "Inspect the signals under this mode, choose one common covered cutoff "
                f"manually, then run hamlet advise {new_manifest_path} --cutoff CUTOFF_MEV"
            ),
        }
    )
    new_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_experiment_inspection(inspection_path, selected, payload)
    return ExperimentModeSelectionResult(
        measurement=selected,
        profile=profile,
        manifest=payload,
        manifest_path=new_manifest_path,
        inspection_path=inspection_path,
    )


def load_canonical_experiment(
    experiment: Measurement | str | Path,
) -> tuple[Measurement, str, str | None, str | None]:
    """Load a guided manifest, canonical NPZ, legacy CSV, or in-memory measurement."""
    if isinstance(experiment, Measurement):
        measurement = experiment
        source = str(measurement.metadata.get("source", "canonical measurement"))
    else:
        path = Path(experiment).resolve()
        if path.suffix.lower() == ".npz":
            measurement = Measurement.load(path)
        elif path.suffix.lower() == ".json":
            measurement, manifest = load_experiment_project(path)
            return (
                measurement,
                str(path),
                str(manifest["system_type"]),
                str(manifest["view"]),
            )
        else:
            measurement = load_measurement_csv(path)
        source = str(path)
    declared_system = measurement.metadata.get("system_type")
    declared_view = measurement.metadata.get("view")
    return (
        measurement,
        source,
        str(declared_system) if declared_system is not None else None,
        str(declared_view) if declared_view is not None else None,
    )


def _with_mode_metadata(
    measurement: Measurement,
    profile: ExperimentModeProfile,
    recipe_source: str | None,
) -> Measurement:
    metadata = {
        **measurement.metadata,
        "experiment_project_schema_version": EXPERIMENT_PROJECT_SCHEMA_VERSION,
        "experiment_mode": profile.mode,
        "experiment_variant": profile.variant,
        "system_type": profile.system_type,
        "view": profile.view,
    }
    if recipe_source is not None:
        metadata["import_recipe"] = str(Path(recipe_source).resolve())
    return Measurement(
        axes=measurement.axes,
        channels=measurement.channels,
        axis_units=measurement.axis_units,
        channel_units=measurement.channel_units,
        primary_channel=measurement.primary_channel,
        masks=measurement.masks,
        uncertainties=measurement.uncertainties,
        metadata=metadata,
    )


def _inspect_measurement(
    measurement: Measurement, profile: ExperimentModeProfile
) -> tuple[list[dict[str, str]], list[str], list[dict[str, Any]]]:
    checks: list[dict[str, str]] = []
    warnings: list[str] = []

    def check(name: str, passed: bool, success: str, failure: str) -> None:
        checks.append(
            {"name": name, "status": "pass" if passed else "fail", "message": success if passed else failure}
        )

    check(
        "axes",
        tuple(measurement.axes) == profile.required_axes,
        f"axes match {profile.required_axes}",
        f"expected ordered axes {profile.required_axes}, found {tuple(measurement.axes)}",
    )
    check(
        "primary_channel",
        measurement.primary_channel == profile.primary_channel,
        f"primary inference channel is {profile.primary_channel}",
        f"mode requires primary channel {profile.primary_channel!r}",
    )
    check(
        "primary_completeness",
        measurement.is_primary_complete,
        "primary inference channel is complete",
        "primary inference channel contains unresolved missing values",
    )
    check(
        "site_count",
        measurement.shape[0] >= profile.minimum_sites,
        f"{measurement.shape[0]} sites satisfy the minimum of {profile.minimum_sites}",
        f"{profile.system_type} requires at least {profile.minimum_sites} sites",
    )
    bias = np.asarray(measurement.axes.get("bias", []), dtype=float)
    bias_mev = measurement.axis_units.get("bias") == "meV"
    check("bias_units", bias_mev, "bias is expressed in meV", "bias axis must be in meV")
    ordered = bool(bias.size >= 2 and np.all(np.isfinite(bias)) and np.all(np.diff(bias) > 0))
    check(
        "bias_axis",
        ordered,
        "bias axis is finite and strictly increasing",
        "bias axis must be finite, strictly increasing, and contain at least two points",
    )
    spans_zero = bool(bias.size and float(np.min(bias)) <= 0.0 <= float(np.max(bias)))
    check(
        "zero_bias_coverage",
        spans_zero,
        "measurement spans zero bias",
        "current spectroscopy preprocessing requires the measurement to span zero bias",
    )
    if ordered:
        steps = np.diff(bias)
        median_step = float(np.median(steps))
        if not np.allclose(steps, median_step, rtol=1e-3, atol=1e-9):
            warnings.append(
                "bias grid is nonuniform; artifact preprocessing will interpolate explicitly"
            )

    site_statistics: list[dict[str, Any]] = []
    try:
        _, spectra = measurement.site_spectra(require_complete=False)
    except ValueError:
        return checks, warnings, site_statistics
    mask = measurement.masks[measurement.primary_channel]
    if measurement.axis_order == ("bias", "site"):
        mask = mask.T
    for index, site in enumerate(np.asarray(measurement.axes["site"])):
        values = spectra[index]
        valid = mask[index] & np.isfinite(values)
        finite_values = values[valid]
        entry: dict[str, Any] = {
            "site": str(site),
            "finite_fraction": float(np.mean(valid)),
            "minimum": float(np.min(finite_values)) if finite_values.size else None,
            "maximum": float(np.max(finite_values)) if finite_values.size else None,
            "median": float(np.median(finite_values)) if finite_values.size else None,
            "standard_deviation": float(np.std(finite_values)) if finite_values.size else None,
        }
        site_statistics.append(entry)
        if finite_values.size and np.isclose(np.std(finite_values), 0.0, atol=1e-14):
            warnings.append(f"site {site!s} has an effectively constant primary spectrum")
    return checks, warnings, site_statistics


def _cutoff_coverage(
    bias: np.ndarray, candidate_cutoffs: Sequence[float]
) -> list[dict[str, Any]]:
    values = sorted({float(item) for item in candidate_cutoffs})
    if any(not np.isfinite(item) or item <= 0 for item in values):
        raise ValueError("candidate cutoffs must be positive finite values")
    low = float(np.min(bias)) if bias.size else np.inf
    high = float(np.max(bias)) if bias.size else -np.inf
    rows = []
    for cutoff in values:
        covered = low <= 0.0 + 1e-8 and high >= cutoff - 1e-8
        rows.append(
            {
                "cutoff_mev": cutoff,
                "covered": bool(covered),
                "reason": (
                    f"measurement covers the required 0–{cutoff:g} meV window"
                    if covered
                    else f"measurement range {low:g}–{high:g} meV does not cover 0–{cutoff:g} meV"
                ),
            }
        )
    return rows


def _write_experiment_inspection(
    path: Path, measurement: Measurement, manifest: Mapping[str, Any]
) -> None:
    image_html = ""
    try:
        figure = measurement.plot_spectroscopy()
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
        try:
            import matplotlib.pyplot as plt

            plt.close(figure)
        except ImportError:  # pragma: no cover
            pass
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        image_html = f'<img alt="Raw canonical spectroscopy channels" src="data:image/png;base64,{encoded}">'
    except ImportError:
        image_html = "<p>Install the <code>io</code> extra to embed plots.</p>"

    checks = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['name']))}</td>"
        f"<td class={'pass' if item['status'] == 'pass' else 'fail'}>{html.escape(str(item['status']))}</td>"
        f"<td>{html.escape(str(item['message']))}</td></tr>"
        for item in manifest["checks"]
    )
    cutoffs = "".join(
        "<tr>"
        f"<td>{item['cutoff_mev']:g}</td>"
        f"<td class={'pass' if item['covered'] else 'fail'}>{'covered' if item['covered'] else 'not covered'}</td>"
        f"<td>{html.escape(str(item['reason']))}</td></tr>"
        for item in manifest["candidate_cutoffs"]
    )
    warnings = "".join(f"<li>{html.escape(str(item))}</li>" for item in manifest["warnings"])
    roles = "".join(
        f"<li><code>{html.escape(name)}</code>: {html.escape(role)}</li>"
        for name, role in manifest["measurement"]["channel_roles"].items()
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{BRAND_NAME} experiment inspection</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:1200px;color:#17242d}}img{{max-width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}th,td{{border:1px solid #ccd;padding:.45rem;text-align:left}}
.pass{{color:#176b32;font-weight:700}}.fail{{color:#a12222;font-weight:700}}code{{background:#eef;padding:.1rem .25rem}}</style></head><body>
<h1>{BRAND_NAME} experiment inspection</h1><p class="{'pass' if manifest['status'] == 'ready_for_cutoff_selection' else 'fail'}">{html.escape(str(manifest['status']))}</p>
<p>{FULL_NAME}</p>
<p>Mode: <strong>{html.escape(str(manifest['mode']))}</strong>; variant: <strong>{html.escape(str(manifest['variant']))}</strong>;
system: <code>{html.escape(str(manifest['system_type']))}</code>.</p>
<p>This page shows raw canonical signals before model-specific cropping, baseline subtraction, normalization, or resampling.</p>
<h2>Channels and roles</h2><ul>{roles}</ul>{image_html}
<h2>Structural checks</h2><table><thead><tr><th>Check</th><th>Status</th><th>Meaning</th></tr></thead><tbody>{checks}</tbody></table>
<h2>Candidate cutoff coverage</h2><p>Coverage is not a recommendation. Inspect the signal and choose one common physically usable cutoff manually.</p>
<table><thead><tr><th>Cutoff [meV]</th><th>Coverage</th><th>Reason</th></tr></thead><tbody>{cutoffs}</tbody></table>
<h2>Warnings</h2><ul>{warnings or '<li>None</li>'}</ul>
<h2>Next step</h2><p><code>{html.escape(str(manifest['next_step']))}</code></p>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
