"""JSON-able operations behind the browser interface.

Kept free of HTTP so each operation can be tested directly, and so the
interface cannot acquire behaviour that the library does not have. Everything
here either reads state or plans work; the only things that *run* work go
through :class:`JobRegistry`, which keeps them off the request thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import threading
import time
import traceback
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def _published_root() -> Path:
    return REPO_ROOT / "models" / "published"


def _examples_root() -> Path:
    return REPO_ROOT / "examples"


# --- guidance ---------------------------------------------------------------

def workflow_overview() -> dict[str, Any]:
    """What the package can do, phrased as the user's situation.

    This is the landing content. It exists because the common failure is not
    knowing which workflow applies, so each entry leads with what the user
    *has* rather than what the tool is called.
    """
    return {
        "situations": [
            {
                "id": "inspect",
                "have": "I have measured dI/dV data and want to look at it",
                "does": (
                    "Reads a per-site spectroscopy file, reports the chain "
                    "length, bias window and whether anything is missing, and "
                    "plots it. Nothing is trained or modified."
                ),
                "needs": "a CSV or .dat file with site, bias and dI/dV columns",
                "cost": "seconds",
            },
            {
                "id": "advise",
                "have": "I have data and want to know if an existing model fits it",
                "does": (
                    "Compares your measurement against every published model's "
                    "contract -- physical system, chain length, bias cutoff, "
                    "observable, parameter ranges -- and says reuse, retrain or "
                    "regenerate, with the reason for each rejection."
                ),
                "needs": "a measurement, and the bias cutoff you have chosen",
                "cost": "seconds",
            },
            {
                "id": "models",
                "have": "I want to see which models already exist",
                "does": (
                    "Lists the published models with what each was trained on, "
                    "how accurate it is per coupling, and the conditions under "
                    "which it must not be reused."
                ),
                "needs": "nothing",
                "cost": "instant",
            },
            {
                "id": "train",
                "have": "I need a model for my own system",
                "does": (
                    "Plans a generate-and-train run from a configuration file "
                    "and shows every output it would write plus a compute "
                    "estimate, before running anything."
                ),
                "needs": "a project configuration; start from an example",
                "cost": "planning is instant; running is hours",
            },
            {
                "id": "dmi",
                "have": "I want to measure DMI",
                "does": (
                    "Ranks candidate impurity arrangements by whether they can "
                    "expose DMI at all. DMI is exactly unmeasurable in a "
                    "conventional chain, so this is a sample-design question "
                    "and has to be answered before any data is taken."
                ),
                "needs": "your chain parameters and the arrangements you could build",
                "cost": "about a minute per candidate",
            },
        ]
    }


# --- published models -------------------------------------------------------

def describe_published_models() -> dict[str, Any]:
    """Published artifacts with the contract that governs reusing them."""
    models: list[dict[str, Any]] = []
    root = _published_root()
    for manifest_path in sorted(root.glob("*/manifest.json")):
        directory = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            models.append({"name": directory.name, "error": str(exc)})
            continue

        metrics = manifest.get("metrics", {})
        recipe = manifest.get("dataset_metadata", {}).get("generation_recipe", {})
        preprocessing = manifest.get("preprocessing", {})

        per_parameter = []
        target_names = list(manifest.get("target_names", []))
        ranges = recipe.get("coupling_ranges_mev") or []
        for index, name in enumerate(target_names):
            entry: dict[str, Any] = {"name": name}
            if index < len(ranges) and isinstance(ranges[index], (list, tuple)):
                entry["trained_range_mev"] = list(ranges[index])
            per_parameter.append(entry)

        # Conditions beyond system_type that a reuse must match exactly.
        conditions: dict[str, Any] = {}
        if recipe.get("impurity_sites") is not None:
            conditions["impurity_sites"] = recipe["impurity_sites"]
            conditions["impurity_spin"] = recipe.get("impurity_spin")
            conditions["impurity_transverse_mev"] = recipe.get("impurity_transverse_mev")
            conditions["impurity_axial_mev"] = recipe.get("impurity_axial_mev", 0.0)
        if recipe.get("impurities") is not None:
            conditions["impurities"] = recipe["impurities"]
        if recipe.get("transverse_field_mev"):
            conditions["transverse_field_mev"] = recipe["transverse_field_mev"]

        models.append(
            {
                "name": directory.name,
                "path": str(directory),
                "system_type": manifest.get("system_type"),
                "view": manifest.get("view"),
                "n_sites": manifest.get("n_sites"),
                "model": manifest.get("model"),
                "preset": manifest.get("training_preset", {}).get("name"),
                "bias_cutoff_mev": preprocessing.get("bias_cutoff_mev"),
                "observable": recipe.get("observable"),
                "n_training_chains": recipe.get("n_samples"),
                "validation_mae_mev": metrics.get("validation", {}).get("ensemble", {}).get("mae"),
                "test_mae_mev": metrics.get("test", {}).get("ensemble", {}).get("mae"),
                "parameters": per_parameter,
                "fixed_conditions": conditions,
                "has_model_card": (directory / "MODEL_CARD.md").exists(),
                "model_card": str(directory / "MODEL_CARD.md"),
            }
        )
    return {"root": str(root), "models": models}


def read_model_card(name: str) -> dict[str, Any]:
    """The model card for one published artifact.

    ``name`` is matched against directory names only, so a path cannot be used
    to read files elsewhere on disk.
    """
    root = _published_root()
    candidates = {path.name: path for path in root.iterdir() if path.is_dir()} if root.exists() else {}
    directory = candidates.get(name)
    if directory is None:
        raise FileNotFoundError(f"no published model named {name!r}")
    card = directory / "MODEL_CARD.md"
    if not card.exists():
        raise FileNotFoundError(f"{name} has no model card")
    return {"name": name, "markdown": card.read_text(encoding="utf-8")}


# --- experiments ------------------------------------------------------------

def inspect_experiment(path: str | Path, *, max_points: int = 400) -> dict[str, Any]:
    """Read a measurement and report what the workflow will make of it.

    The reported fields are exactly the ones that decide whether a model can be
    reused, so this doubles as a preflight check.
    """
    from ..experiments import load_canonical_experiment

    measurement, source, system_type, view = load_canonical_experiment(str(path))
    bias, spectra = measurement.site_spectra(require_complete=False)
    bias = np.asarray(bias, dtype=float)
    spectra = np.asarray(spectra, dtype=float)

    # Thin for transport; the browser never needs full resolution to show shape.
    if bias.size > max_points:
        keep = np.linspace(0, bias.size - 1, max_points).astype(int)
        bias_out, spectra_out = bias[keep], spectra[:, keep]
    else:
        bias_out, spectra_out = bias, spectra

    finite = np.isfinite(spectra)
    return {
        "source": source,
        "path": str(Path(path).resolve()),
        "declared_system_type": system_type,
        "declared_view": view,
        "n_sites": int(spectra.shape[0]),
        "n_bias_points": int(bias.size),
        "bias_min_mev": float(bias.min()),
        "bias_max_mev": float(bias.max()),
        "bias_units": measurement.axis_units.get("bias", "meV"),
        "primary_channel": measurement.primary_channel,
        "is_complete": bool(measurement.is_primary_complete),
        "missing_points": int((~finite).sum()),
        "starts_at_zero": bool(abs(float(bias.min())) < 1e-8),
        "plot": {
            "bias_mev": bias_out.tolist(),
            "sites": [row.tolist() for row in spectra_out],
        },
    }


def advise_for_experiment(
    path: str | Path,
    cutoff_mev: float,
    *,
    artifact_roots: list[str] | None = None,
    system_type: str | None = None,
    view: str | None = None,
) -> dict[str, Any]:
    """Reuse / retrain / regenerate, with the reason for every rejection.

    Defaults to searching the published models, since that is what a user
    without their own artifacts has.
    """
    from ..workflow import advise_experiment

    roots = artifact_roots if artifact_roots else [str(_published_root())]
    decision = advise_experiment(
        str(path),
        manual_cutoff_mev=float(cutoff_mev),
        artifact_roots=[r for r in roots if Path(r).exists()],
        system_type=system_type,
        view=view,
    )
    return {
        "action": decision.action,
        "summary": decision.summary,
        "can_use_existing_model": bool(decision.can_use_existing_model),
        "manual_cutoff_mev": decision.manual_cutoff_mev,
        "system_type": decision.system_type,
        "view": decision.view,
        "n_sites": decision.n_sites,
        "bias_min_mev": decision.bias_min_mev,
        "bias_max_mev": decision.bias_max_mev,
        "experiment_checks": list(decision.experiment_checks),
        "next_steps": list(decision.next_steps),
        "artifacts": [
            {
                "path": str(item.path),
                "compatible": bool(item.compatible),
                "reasons": list(item.reasons),
            }
            for item in decision.artifact_assessments
        ],
        "selected_artifact": (
            str(decision.selected_artifact) if decision.selected_artifact else None
        ),
    }


# --- projects ---------------------------------------------------------------

def list_example_configs() -> dict[str, Any]:
    """Shipped configurations, with the first comment line as a description."""
    root = _examples_root()
    entries = []
    for path in sorted(root.glob("*.yaml")):
        description = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("name:"):
                description = stripped.split(":", 1)[1].strip()
                break
        entries.append(
            {"name": path.name, "path": str(path), "description": description}
        )
    return {"root": str(root), "configs": entries}


def plan_project(
    config_path: str | Path, *, seconds_per_chain: float | None = None
) -> dict[str, Any]:
    """What a run would do, without doing any of it.

    Side-effect free by construction, so this is safe to call on any
    configuration a user pastes in.
    """
    from ..project import HamiltonianLearningProject

    project = HamiltonianLearningProject.from_config(str(config_path))
    plan = project.plan(seconds_per_chain=seconds_per_chain)
    payload = plan.to_dict()
    payload["config_path"] = str(Path(config_path).resolve())
    return payload


def read_config_text(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    return {"path": str(path), "text": path.read_text(encoding="utf-8")}


def write_config_text(config_path: str | Path, text: str) -> dict[str, Any]:
    """Save an edited configuration, validating it before writing.

    Validation first means a user cannot save a file that the workflow will
    later refuse, which is the sort of delayed failure this interface exists to
    avoid.
    """
    from ..project import ProjectConfig

    path = Path(config_path).resolve()
    scratch = path.with_suffix(path.suffix + ".checking")
    scratch.write_text(text, encoding="utf-8")
    try:
        ProjectConfig.from_file(scratch)
    except Exception as exc:
        scratch.unlink(missing_ok=True)
        raise ValueError(f"configuration is not valid: {exc}") from exc
    scratch.replace(path)
    return {"path": str(path), "saved": True}


# --- DMI screening ----------------------------------------------------------

def screening_preview(config_path: str | Path) -> dict[str, Any]:
    """Expand a screening configuration and report the free symmetry verdict.

    No simulation. Designs that cannot break the S^z symmetry are hopeless
    regardless of measurement quality, and saying so instantly is the single
    most useful thing this screen can do.
    """
    from ..dmi_design import load_screening_config

    designs, protocol = load_screening_config(str(config_path))
    candidates = [
        {
            "label": design.name,
            "n_sites": design.n_sites,
            "impurities": [
                {
                    "site": impurity.site,
                    "spin": impurity.spin,
                    "axial_mev": impurity.axial_mev,
                    "transverse_mev": impurity.transverse_mev,
                }
                for impurity in design.impurities
            ],
            "transverse_field_mev": design.transverse_field_mev,
            "breaks_symmetry": bool(design.breaks_symmetry),
        }
        for design in designs
    ]
    viable = sum(1 for item in candidates if item["breaks_symmetry"])
    return {
        "config_path": str(Path(config_path).resolve()),
        "n_candidates": len(candidates),
        "n_can_break_symmetry": viable,
        "candidates": candidates,
        "bias_points": int(np.asarray(protocol.bias_mev).size),
        "broadening_mev": float(protocol.broadening_mev),
        "note": (
            "Designs that cannot break the S^z symmetry are skipped when you "
            "run the screening; they cannot constrain D_z however good the data."
        ),
    }


# --- background jobs --------------------------------------------------------

@dataclass
class GuiJob:
    """One long-running operation, with its captured output."""

    job_id: str
    kind: str
    label: str
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    lines: list[str] = field(default_factory=list)
    error: str | None = None
    result: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round(
                (self.finished_at or time.time()) - self.started_at, 1
            ),
            "lines": list(self.lines),
            "error": self.error,
            "result": self.result,
        }


class JobRegistry:
    """Runs long operations off the request thread and keeps their output.

    Generation and training take hours, and screening minutes, so the interface
    must never block on them. Each job's stdout is captured so the browser can
    show real progress rather than a spinner.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, GuiJob] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def submit(self, kind: str, label: str, work: Callable[[], Any]) -> GuiJob:
        with self._lock:
            self._counter += 1
            job_id = f"{kind}-{self._counter}"
            job = GuiJob(job_id=job_id, kind=kind, label=label)
            self._jobs[job_id] = job

        def run() -> None:
            import contextlib

            stream = _JobStream(job, self._lock)
            status = "finished"
            error: str | None = None
            try:
                with contextlib.redirect_stdout(stream):
                    job.result = work()
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    job.lines.append("")
                    job.lines.extend(traceback.format_exc().splitlines()[-12:])
            finally:
                job.finished_at = time.time()
                job.error = error
                # Status is assigned last, so that anything polling and seeing a
                # terminal status is guaranteed to also see the complete output
                # and the error. Setting it first leaves a window in which a
                # failure looks like it produced no diagnostics at all.
                job.status = status

        threading.Thread(target=run, name=f"hamlet-gui-{job_id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> GuiJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        # Newest first: a user watching a run wants the current one on top.
        return [job.to_dict() for job in sorted(jobs, key=lambda j: -j.started_at)]


class _JobStream(io.TextIOBase):
    """Collects a job's stdout line by line for the browser to poll."""

    def __init__(self, job: GuiJob, lock: threading.Lock) -> None:
        self._job = job
        self._lock = lock
        self._partial = ""

    def write(self, text: str) -> int:
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            with self._lock:
                self._job.lines.append(line)
                # Bound memory on a chatty multi-hour run.
                if len(self._job.lines) > 2000:
                    del self._job.lines[:500]
        return len(text)

    def flush(self) -> None:
        return None
