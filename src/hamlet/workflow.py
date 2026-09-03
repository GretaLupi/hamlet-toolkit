"""Preflight decisions for experiment → reuse, retrain, or regenerate workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import SpectroscopyDataset
from .branding import brand_manifest
from .experiments import load_canonical_experiment
from .measurements import Measurement
from .training.guided import ARTIFACT_SCHEMA_VERSION, TrainingRun


@dataclass(frozen=True)
class ResourceAssessment:
    """Compatibility result for one artifact or simulation dataset."""

    path: Path
    resource_type: str
    compatible: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    cutoff_mev: float | None = None
    validation_mae_mev: float | None = None
    test_mae_mev: float | None = None
    n_samples: int | None = None


@dataclass(frozen=True)
class WorkflowDecision:
    """Actionable, serializable preflight decision for one manual cutoff."""

    action: str
    summary: str
    experiment_source: str
    manual_cutoff_mev: float
    system_type: str
    view: str
    n_sites: int
    bias_min_mev: float
    bias_max_mev: float
    selected_artifact: Path | None
    selected_dataset: Path | None
    experiment_checks: tuple[str, ...]
    artifact_assessments: tuple[ResourceAssessment, ...]
    dataset_assessments: tuple[ResourceAssessment, ...]
    next_steps: tuple[str, ...]

    @property
    def can_use_existing_model(self) -> bool:
        return self.action == "use_existing_model"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Keep this small machine-readable contract stable for a future GUI.
        # Bump the schema version only when consumers must change.
        payload["workflow_decision_schema_version"] = 1
        payload["package_version"] = "0.1.0"
        payload["toolkit"] = brand_manifest()
        payload["selected_artifact"] = (
            str(self.selected_artifact) if self.selected_artifact is not None else None
        )
        payload["selected_dataset"] = (
            str(self.selected_dataset) if self.selected_dataset is not None else None
        )
        for key in ("artifact_assessments", "dataset_assessments"):
            for item in payload[key]:
                item["path"] = str(item["path"])
        payload["can_use_existing_model"] = self.can_use_existing_model
        return payload

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination

    def save_html(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        status_class = "pass" if self.can_use_existing_model else "review"
        artifact_rows = _assessment_rows(self.artifact_assessments)
        dataset_rows = _assessment_rows(self.dataset_assessments)
        checks = "".join(f"<li>{html.escape(item)}</li>" for item in self.experiment_checks)
        steps = "".join(f"<li>{html.escape(item)}</li>" for item in self.next_steps)
        document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>HamLeT workflow decision</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:1200px;color:#17242d}}
table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}th,td{{border:1px solid #ccd;padding:.45rem;text-align:left;vertical-align:top}}
.pass{{color:#176b32;font-weight:700}}.review{{color:#9a5300;font-weight:700}}
code{{background:#eef;padding:.1rem .25rem}}small{{color:#586873}}</style></head><body>
<h1>HamLeT workflow decision</h1><p class="{status_class}">{html.escape(self.action)}</p>
<p>{html.escape(self.summary)}</p>
<p>Manual cutoff: <strong>{self.manual_cutoff_mev:g} meV</strong>; system:
<code>{html.escape(self.system_type)}</code>; view: <code>{html.escape(self.view)}</code>.</p>
<p>Experiment: {self.n_sites} sites, {self.bias_min_mev:g}–{self.bias_max_mev:g} meV.</p>
<h2>Experiment checks</h2><ul>{checks}</ul>
<h2>Artifact assessments</h2>{_assessment_table(artifact_rows)}
<h2>Dataset assessments</h2>{_assessment_table(dataset_rows)}
<h2>Next steps</h2><ol>{steps}</ol>
<p><small>Suitability checks verify stored model contracts; they do not prove that the physical simulator is complete.</small></p>
</body></html>"""
        destination.write_text(document, encoding="utf-8")
        return destination


def advise_experiment(
    experiment: Measurement | str | Path,
    *,
    manual_cutoff_mev: float,
    artifact_roots: Sequence[str | Path] = (),
    dataset_paths: Sequence[str | Path] = (),
    system_type: str | None = None,
    view: str | None = None,
    allow_development_artifacts: bool = False,
    max_validation_mae_mev: float | None = None,
    max_test_mae_mev: float | None = None,
) -> WorkflowDecision:
    """Choose reuse/retrain/regenerate from explicit experiment and cutoff contracts.

    Artifact reuse requires a matching physical system and learning view, an
    exact cutoff match, complete preprocessing coverage, compatible chain
    length, a loadable artifact, and finite stored metrics. A different trained
    cutoff is never silently substituted.
    """
    if not np.isfinite(manual_cutoff_mev) or manual_cutoff_mev <= 0:
        raise ValueError("manual_cutoff_mev must be a positive finite value")
    for name, value in (
        ("max_validation_mae_mev", max_validation_mae_mev),
        ("max_test_mae_mev", max_test_mae_mev),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be a positive finite value")
    measurement, source, declared_system, declared_view = load_canonical_experiment(experiment)
    if system_type is not None and declared_system is not None and system_type != declared_system:
        raise ValueError(
            f"requested system {system_type!r} conflicts with experiment manifest "
            f"system {declared_system!r}"
        )
    if view is not None and declared_view is not None and view != declared_view:
        raise ValueError(
            f"requested view {view!r} conflicts with experiment manifest view {declared_view!r}"
        )
    system_type = system_type or declared_system or "inhomogeneous_heisenberg"
    view = view or declared_view or (
        "global" if system_type == "homogeneous_heisenberg" else "local_bonds"
    )
    if view not in {"local_bonds", "global"}:
        raise ValueError("view must be local_bonds or global")
    # Structural failures should become an actionable workflow decision rather
    # than an importer-shaped exception. Artifact preprocessing still requires
    # completeness and is assessed separately below.
    bias, spectra = measurement.site_spectra(require_complete=False)
    n_sites = int(spectra.shape[0])
    bias_min = float(np.min(bias))
    bias_max = float(np.max(bias))
    checks: list[str] = []
    experiment_valid = True
    if measurement.axis_units.get("bias", "meV") != "meV":
        checks.append("FAIL: canonical bias axis is not in meV")
        experiment_valid = False
    else:
        checks.append("PASS: canonical bias axis is in meV")
    if view == "local_bonds" and n_sites < 3:
        checks.append("FAIL: local-bond inference requires at least three sites")
        experiment_valid = False
    else:
        checks.append(f"PASS: site count ({n_sites}) is structurally valid for {view}")
    if bias_min > 0.0 + 1e-8 or bias_max < manual_cutoff_mev - 1e-8:
        checks.append(
            f"FAIL: experiment covers {bias_min:g}–{bias_max:g} meV but the selected "
            f"window requires 0–{manual_cutoff_mev:g} meV"
        )
        experiment_valid = False
    else:
        checks.append(
            f"PASS: experiment covers the manually selected 0–{manual_cutoff_mev:g} meV window"
        )
    if measurement.is_primary_complete:
        checks.append(f"PASS: primary channel {measurement.primary_channel!r} is complete")
    else:
        checks.append("FAIL: primary channel contains unresolved missing values")
        experiment_valid = False

    discovered_artifacts = _discover_artifacts(artifact_roots)
    artifact_results: list[ResourceAssessment] = []
    for supplied in artifact_roots:
        root = Path(supplied).resolve()
        if not root.exists():
            artifact_results.append(
                ResourceAssessment(root, "artifact", False, ("artifact root does not exist",))
            )
            continue
        covered = any(
            candidate == root or root in candidate.parents
            for candidate in discovered_artifacts
        )
        if not covered:
            artifact_results.append(
                ResourceAssessment(
                    root, "artifact", False, ("no model artifact manifests were found",)
                )
            )
    artifact_results.extend(
        _assess_artifact(
            path,
            measurement,
            manual_cutoff_mev,
            system_type,
            view,
            allow_development_artifacts,
            max_validation_mae_mev,
            max_test_mae_mev,
        )
        for path in discovered_artifacts
    )
    artifact_assessments = tuple(artifact_results)
    eligible_artifacts = [item for item in artifact_assessments if item.compatible]
    dataset_assessments = tuple(
        _assess_dataset(Path(path), measurement, manual_cutoff_mev, system_type, view)
        for path in dataset_paths
    )
    eligible_datasets = [item for item in dataset_assessments if item.compatible]

    selected_artifact: Path | None = None
    selected_dataset: Path | None = None
    if not experiment_valid:
        action = "fix_experiment_or_choose_lower_cutoff"
        summary = (
            "The measurement does not satisfy the manually selected input window. "
            "Training another model cannot repair missing experimental coverage."
        )
        next_steps = (
            "Inspect the import report and physical site order.",
            "Choose a lower cutoff that remains scientifically usable, or acquire/export the missing range.",
            "Run the advisor again before training or inference.",
        )
    elif eligible_artifacts:
        selected = min(
            eligible_artifacts,
            key=lambda item: (
                float("inf") if item.validation_mae_mev is None else item.validation_mae_mev,
                str(item.path),
            ),
        )
        selected_artifact = selected.path
        action = "use_existing_model"
        summary = (
            f"A saved artifact trained at exactly {manual_cutoff_mev:g} meV satisfies "
            "the experiment and model contract."
        )
        next_steps = (
            f"Use artifact: {selected.path}",
            "Run inference and retain its ensemble and overlap-consistency warnings.",
            "Do not interpret ensemble spread as a calibrated confidence interval.",
        )
    elif eligible_datasets:
        selected = max(
            eligible_datasets,
            key=lambda item: (item.n_samples or 0, str(item.path)),
        )
        selected_dataset = selected.path
        action = "retrain_with_existing_dataset"
        summary = (
            f"No distributable artifact passes the exact {manual_cutoff_mev:g} meV "
            "contract, but an existing raw simulation dataset covers this cutoff."
        )
        next_steps = (
            f"Use dataset: {selected.path}",
            "Train fresh weights at the exact selected cutoff.",
            "Validate the model on held-out simulated chains and register the accepted artifact.",
            "Run the advisor again before inference.",
        )
    else:
        action = "generate_dataset_and_retrain"
        summary = (
            f"No saved artifact or listed dataset satisfies the {manual_cutoff_mev:g} meV "
            "experiment contract. New or expanded simulations are required."
        )
        next_steps = (
            "Define a simulation family whose bias range covers the selected cutoff and whose targets span the intended physics.",
            "Generate a versioned, checkpointed dataset with the package recipe.",
            "Run grouped training and held-out validation at the selected cutoff.",
            "Register the accepted artifact in a cutoff bank and rerun this advisor.",
        )
    return WorkflowDecision(
        action=action,
        summary=summary,
        experiment_source=source,
        manual_cutoff_mev=float(manual_cutoff_mev),
        system_type=system_type,
        view=view,
        n_sites=n_sites,
        bias_min_mev=bias_min,
        bias_max_mev=bias_max,
        selected_artifact=selected_artifact,
        selected_dataset=selected_dataset,
        experiment_checks=tuple(checks),
        artifact_assessments=artifact_assessments,
        dataset_assessments=dataset_assessments,
        next_steps=next_steps,
    )


def _discover_artifacts(roots: Sequence[str | Path]) -> tuple[Path, ...]:
    found: set[Path] = set()
    for value in roots:
        root = Path(value).resolve()
        if root.is_file() and root.name == "manifest.json":
            found.add(root.parent)
        elif (root / "manifest.json").exists():
            found.add(root)
        elif root.exists():
            for manifest in root.rglob("manifest.json"):
                found.add(manifest.parent)
    return tuple(sorted(found))


def _assess_artifact(
    path: Path,
    measurement: Measurement,
    cutoff: float,
    system_type: str,
    view: str,
    allow_development: bool,
    max_validation_mae: float | None,
    max_test_mae: float | None,
) -> ResourceAssessment:
    reasons: list[str] = []
    warnings: list[str] = []
    manifest_path = path / "manifest.json"
    stored_cutoff: float | None = None
    validation_mae: float | None = None
    test_mae: float | None = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ResourceAssessment(path, "artifact", False, (f"cannot read manifest: {exc}",))
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        reasons.append("unsupported artifact schema")
    if manifest.get("system_type") != system_type:
        reasons.append(
            f"system mismatch: artifact={manifest.get('system_type')!r}, requested={system_type!r}"
        )
    if manifest.get("view") != view:
        reasons.append(f"view mismatch: artifact={manifest.get('view')!r}, requested={view!r}")
    if view == "global":
        try:
            trained_sites = int(manifest["n_sites"])
        except (KeyError, TypeError, ValueError):
            reasons.append("global artifact has no valid trained chain length")
        else:
            experiment_sites = int(measurement.shape[0])
            if trained_sites != experiment_sites:
                reasons.append(
                    f"chain-length mismatch: artifact={trained_sites} sites, "
                    f"experiment={experiment_sites} sites"
                )
    preprocessing = manifest.get("preprocessing", {})
    try:
        stored_cutoff = float(preprocessing["bias_cutoff_mev"])
        bias_min = float(preprocessing.get("bias_min_mev", 0.0))
    except (KeyError, TypeError, ValueError):
        reasons.append("artifact has no valid preprocessing cutoff contract")
        bias_min = 0.0
    else:
        if not np.isclose(stored_cutoff, cutoff, rtol=0.0, atol=1e-8):
            reasons.append(
                f"cutoff mismatch: artifact={stored_cutoff:g} meV, manual choice={cutoff:g} meV"
            )
        bias = np.asarray(measurement.axes["bias"], dtype=float)
        if float(bias.min()) > bias_min + 1e-8 or float(bias.max()) < stored_cutoff - 1e-8:
            reasons.append(
                f"experiment does not cover artifact window {bias_min:g}–{stored_cutoff:g} meV"
            )
    preset_name = str(manifest.get("training_preset", {}).get("name", "unknown"))
    if preset_name == "quick" and not allow_development:
        reasons.append("quick-preset artifact is development-only")
    metrics = manifest.get("metrics", {})
    try:
        validation_mae = float(metrics["validation"]["ensemble"]["mae"])
        test_mae = float(metrics["test"]["ensemble"]["mae"])
        if not np.isfinite(validation_mae) or not np.isfinite(test_mae):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        reasons.append("artifact does not contain finite validation and test MAE")
    else:
        if max_validation_mae is not None and validation_mae > max_validation_mae:
            reasons.append(
                f"validation MAE {validation_mae:.3g} meV exceeds the user limit "
                f"{max_validation_mae:.3g} meV"
            )
        if max_test_mae is not None and test_mae > max_test_mae:
            reasons.append(
                f"test MAE {test_mae:.3g} meV exceeds the user limit {max_test_mae:.3g} meV"
            )
        if max_validation_mae is None and max_test_mae is None:
            warnings.append(
                f"no user MAE acceptance limit supplied; stored validation/test MAE are "
                f"{validation_mae:.3g}/{test_mae:.3g} meV"
            )
    protocol = manifest.get("dataset_metadata", {}).get("protocol", {})
    quantity = protocol.get("output_quantity")
    if quantity is not None and quantity != measurement.primary_channel:
        reasons.append(
            f"observable mismatch: artifact trained on {quantity!r}, experiment uses "
            f"{measurement.primary_channel!r}"
        )
    # Schema-v1 Heisenberg artifacts predating explicit observable contracts
    # were generated with the package's historical Sz-only approximation.
    trained_observable = protocol.get("observable", "Sz")
    expected_observable = measurement.metadata.get("simulation_observable")
    if (
        trained_observable is not None
        and expected_observable is not None
        and trained_observable != expected_observable
    ):
        reasons.append(
            "simulation-observable mismatch: artifact="
            f"{trained_observable!r}, experiment mode expects={expected_observable!r}"
        )
    if reasons:
        return ResourceAssessment(
            path, "artifact", False, tuple(reasons), tuple(warnings), stored_cutoff,
            validation_mae,
            test_mae,
        )
    try:
        run = TrainingRun.load(path)
        bias, spectra = measurement.site_spectra()
        processed = run.preprocessing_config.build_preprocessor().transform_map(spectra, bias)
        if not np.all(np.isfinite(processed)):
            reasons.append("artifact preprocessing produced non-finite values")
    except Exception as exc:
        reasons.append(f"artifact cannot preprocess this experiment: {exc}")
    return ResourceAssessment(
        path=path,
        resource_type="artifact",
        compatible=not reasons,
        reasons=(
            tuple(reasons)
            if reasons
            else ("system, view, cutoff, chain length, observable, and metrics match",)
        ),
        warnings=tuple(warnings),
        cutoff_mev=stored_cutoff,
        validation_mae_mev=validation_mae,
        test_mae_mev=test_mae,
    )


def _assess_dataset(
    path: Path,
    measurement: Measurement,
    cutoff: float,
    system_type: str,
    view: str,
) -> ResourceAssessment:
    reasons: list[str] = []
    warnings: list[str] = []
    try:
        dataset = SpectroscopyDataset.load(path)
    except Exception as exc:
        return ResourceAssessment(path, "dataset", False, (f"cannot load dataset: {exc}",))
    if dataset.system_type != system_type:
        reasons.append(
            f"system mismatch: dataset={dataset.system_type!r}, requested={system_type!r}"
        )
    if float(dataset.bias_mev.min()) > 0.0 + 1e-8 or float(dataset.bias_mev.max()) < cutoff - 1e-8:
        reasons.append(
            f"dataset covers {float(dataset.bias_mev.min()):g}–"
            f"{float(dataset.bias_mev.max()):g} meV, not 0–{cutoff:g} meV"
        )
    if view == "local_bonds":
        if dataset.targets_mev.shape[1] != dataset.n_sites - 1 or dataset.n_sites < 3:
            reasons.append("dataset does not satisfy the local one-target-per-bond contract")
    elif dataset.n_sites != measurement.shape[0]:
        reasons.append(
            f"global view requires the experiment and dataset to have the same site count "
            f"({measurement.shape[0]} vs {dataset.n_sites})"
        )
    protocol = dataset.metadata.get("protocol", {})
    quantity = protocol.get("output_quantity") if isinstance(protocol, dict) else None
    if quantity is not None and quantity != measurement.primary_channel:
        reasons.append(
            f"observable mismatch: dataset={quantity!r}, experiment={measurement.primary_channel!r}"
        )
    trained_observable = (
        protocol.get("observable", "Sz") if isinstance(protocol, dict) else "Sz"
    )
    expected_observable = measurement.metadata.get("simulation_observable")
    if (
        trained_observable is not None
        and expected_observable is not None
        and trained_observable != expected_observable
    ):
        reasons.append(
            "simulation-observable mismatch: dataset="
            f"{trained_observable!r}, experiment mode expects={expected_observable!r}"
        )
    if dataset.n_samples < 10:
        warnings.append(
            f"only {dataset.n_samples} physical samples; use for development, not a distributable model"
        )
    return ResourceAssessment(
        path=path,
        resource_type="dataset",
        compatible=not reasons,
        reasons=tuple(reasons) if reasons else ("system, view, observable, and cutoff coverage match",),
        warnings=tuple(warnings),
        cutoff_mev=cutoff,
        n_samples=dataset.n_samples,
    )


def _assessment_rows(items: Sequence[ResourceAssessment]) -> str:
    if not items:
        return ""
    rows = []
    for item in items:
        details = "; ".join((*item.reasons, *item.warnings))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.path))}</td>"
            f"<td class={'pass' if item.compatible else 'review'}>{'compatible' if item.compatible else 'not compatible'}</td>"
            f"<td>{html.escape(details)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _assessment_table(rows: str) -> str:
    if not rows:
        return "<p>No resources were supplied.</p>"
    return (
        "<table><thead><tr><th>Path</th><th>Status</th><th>Reasons and warnings</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )
