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


def _workspace_root() -> Path:
    """Where runs started from the interface put their artifacts."""
    return REPO_ROOT / "results" / "gui-projects"


def _artifact_directories() -> list[tuple[Path, str]]:
    """Every artifact the interface should know about, with where it came from.

    A model you trained yourself is exactly the model you most want offered
    back, so the workspace is searched alongside the published models rather
    than only the latter.
    """
    found: list[tuple[Path, str]] = []
    published = _published_root()
    if published.exists():
        found.extend(
            (path.parent, "published") for path in sorted(published.glob("*/manifest.json"))
        )
    workspace = _workspace_root()
    if workspace.exists():
        for path in sorted(workspace.rglob("manifest.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            # Only real training artifacts; a project directory holds other
            # manifests too, such as generation recipes.
            if "artifact_schema_version" in payload:
                found.append((path.parent, "yours"))
    return found


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

def describe_available_models() -> dict[str, Any]:
    """Every usable artifact with the contract that governs reusing it.

    Covers both the models shipped with the package and any trained through the
    interface, tagged by origin so provenance stays obvious.
    """
    models: list[dict[str, Any]] = []
    for directory, origin in _artifact_directories():
        manifest_path = directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            models.append({"name": directory.name, "origin": origin, "error": str(exc)})
            continue

        metrics = manifest.get("metrics", {})
        dataset_metadata = manifest.get("dataset_metadata", {})
        recipe = dataset_metadata.get("generation_recipe", {})
        preprocessing = manifest.get("preprocessing", {})

        # Published artifacts come from two pipelines -- datasets generated by
        # this package, and datasets imported from earlier work -- whose
        # manifests record the same facts under different keys. Reading only one
        # shape silently renders the other as blanks.
        model_name = manifest.get("model") or manifest.get("model_name")
        observable = recipe.get("observable") or dataset_metadata.get("observable")
        # The split-group counts are preferred over the recipe's n_samples
        # because a dataset merged from shards records the *per-shard* count in
        # its recipe: reading that would report a 3000-chain model as having
        # been trained on 20.
        split = metrics.get("split", {})
        groups = [
            split.get(key)
            for key in ("train_groups", "validation_groups", "test_groups")
        ]
        if all(isinstance(value, int) for value in groups):
            n_chains = sum(groups)
        else:
            n_chains = recipe.get("n_samples")

        per_parameter = []
        target_names = list(manifest.get("target_names", []))
        ranges = recipe.get("coupling_ranges_mev") or []
        # Fall back to the range actually present in the stored training
        # distribution, so the validity range is shown even for an artifact
        # whose manifest does not carry the sampling recipe.
        observed_ranges: list[list[float]] = []
        if not ranges:
            profile = directory / "training_distribution.npz"
            if profile.exists():
                try:
                    with np.load(profile) as stored:
                        targets = np.asarray(stored["reference_targets"], dtype=float)
                    observed_ranges = [
                        [float(targets[:, i].min()), float(targets[:, i].max())]
                        for i in range(targets.shape[1])
                    ]
                except (OSError, KeyError, ValueError, IndexError):
                    observed_ranges = []
        for index, name in enumerate(target_names):
            entry: dict[str, Any] = {"name": name}
            if index < len(ranges) and isinstance(ranges[index], (list, tuple)):
                entry["trained_range_mev"] = list(ranges[index])
            elif index < len(observed_ranges):
                entry["trained_range_mev"] = observed_ranges[index]
                entry["range_source"] = "observed in the stored training distribution"
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
                "origin": origin,
                "label": (
                    directory.name
                    if origin == "published"
                    else _workspace_label(directory)
                ),
                "path": str(directory),
                "system_type": manifest.get("system_type"),
                "view": manifest.get("view"),
                "n_sites": manifest.get("n_sites"),
                "model": model_name,
                "preset": manifest.get("training_preset", {}).get("name"),
                "bias_cutoff_mev": preprocessing.get("bias_cutoff_mev"),
                "observable": observable,
                "n_training_chains": n_chains,
                "validation_mae_mev": metrics.get("validation", {}).get("ensemble", {}).get("mae"),
                "test_mae_mev": metrics.get("test", {}).get("ensemble", {}).get("mae"),
                "parameters": per_parameter,
                "fixed_conditions": conditions,
                "has_model_card": (directory / "MODEL_CARD.md").exists(),
                "model_card": str(directory / "MODEL_CARD.md"),
            }
        )
    return {
        "published_root": str(_published_root()),
        "workspace_root": str(_workspace_root()),
        "models": models,
    }


def _workspace_label(directory: Path) -> str:
    """A readable name for a model trained through the interface.

    The artifact directory is called "artifact"; the project directory above it
    carries the name the user typed, which is the useful one.
    """
    workspace = _workspace_root()
    try:
        relative = directory.relative_to(workspace)
    except ValueError:
        return directory.name
    return relative.parts[0] if relative.parts else directory.name


def read_model_card(name: str) -> dict[str, Any]:
    """The model card for one published artifact.

    ``name`` is matched against directory names only, so a path cannot be used
    to read files elsewhere on disk.
    """
    candidates: dict[str, Path] = {}
    for directory, origin in _artifact_directories():
        key = directory.name if origin == "published" else _workspace_label(directory)
        candidates.setdefault(key, directory)
    directory = candidates.get(name)
    if directory is None:
        raise FileNotFoundError(f"no model named {name!r}")
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

    # Both roots by default: a model the user just trained is the one they are
    # most likely to be asking about.
    roots = artifact_roots or [str(_published_root()), str(_workspace_root())]
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


# --- guided builder ---------------------------------------------------------
# The form is described here rather than in the page, so that the choices it
# offers cannot drift from what the library actually accepts. Every default is
# the library's own default or a value taken from a published reference model.

_SYSTEM_SPECS: tuple[dict[str, Any], ...] = (
    {
        "system_type": "inhomogeneous_heisenberg",
        "title": "Bond-inhomogeneous Heisenberg",
        "recovers": "one exchange coupling per bond, varying along the chain",
        "when": (
            "The chain is not uniform and you want the coupling profile. This "
            "is the validated workflow, with a published model from "
            "peer-reviewed work."
        ),
        "view": "local_bonds",
        "coupling_mode": "single_range",
        "couplings": [{"name": "J", "low": 30.0, "high": 45.0}],
        "supports_impurities": False,
        "supports_field": False,
        "defaults": {
            "n_sites": 12,
            "bias_range_mev": [0.0, 100.0],
            "bias_points": 200,
            "broadening_mev": 0.5,
            "observable": "Sz",
            "cutoff_mev": 50.0,
            "output_points": 200,
            "model": "keras_mlp",
        },
    },
    {
        "system_type": "homogeneous_heisenberg",
        "title": "Homogeneous Heisenberg (J1, J2, ...)",
        "recovers": "uniform couplings shared by the whole chain",
        "when": (
            "The chain is uniform and you want its exchange constants. One "
            "range per interaction distance."
        ),
        "view": "global",
        "coupling_mode": "per_parameter",
        "couplings": [
            {"name": "J1", "low": 30.0, "high": 45.0},
            {"name": "J2", "low": 0.0, "high": 10.0},
        ],
        "supports_impurities": False,
        "supports_field": False,
        "defaults": {
            "n_sites": 8,
            "bias_range_mev": [0.0, 100.0],
            "bias_points": 200,
            "broadening_mev": 0.5,
            "observable": "Sz",
            "cutoff_mev": 60.0,
            "output_points": 200,
            "model": "random_forest",
        },
    },
    {
        "system_type": "homogeneous_xxz_j1j2j3",
        "title": "Anisotropic XXZ + J2 + J3",
        "recovers": "J1_xy, J2, J3 and Jz",
        "when": "The chain has easy-axis or easy-plane anisotropy.",
        "view": "global",
        "coupling_mode": "per_parameter",
        "couplings": [
            {"name": "J1_xy", "low": 2.0, "high": 8.0},
            {"name": "J2", "low": -1.5, "high": 1.5},
            {"name": "J3", "low": -1.0, "high": 1.0},
            {"name": "Jz", "low": 2.0, "high": 8.0},
        ],
        "supports_impurities": False,
        "supports_field": False,
        "defaults": {
            "n_sites": 8,
            "bias_range_mev": [0.0, 20.0],
            "bias_points": 81,
            "broadening_mev": 0.25,
            "observable": "total_spin",
            "cutoff_mev": 20.0,
            "output_points": 61,
            "model": "ridge",
        },
    },
    {
        "system_type": "homogeneous_xxz_j1j2j3_dmi",
        "title": "XXZ + J2 + J3 + DMI (no symmetry breaking)",
        "recovers": "J1_xy, J2, J3, Jz and a D_z magnitude",
        "when": (
            "Rarely what you want. D_z is exactly unidentifiable in a chain "
            "that conserves total S^z, and it has been measured here as "
            "unlearnable. Use the impurity system below instead."
        ),
        "view": "global",
        "coupling_mode": "per_parameter",
        "couplings": [
            {"name": "J1_xy", "low": 2.0, "high": 8.0},
            {"name": "J2", "low": -1.5, "high": 1.5},
            {"name": "J3", "low": -1.0, "high": 1.0},
            {"name": "Jz", "low": 2.0, "high": 8.0},
            {"name": "D_z", "low": 0.0, "high": 2.0, "min": 0.0},
        ],
        "supports_impurities": False,
        "supports_field": False,
        "warning": (
            "D_z cannot be recovered from this system. Measured skill against "
            "a training-mean baseline is about zero at every dataset size "
            "tried."
        ),
        "defaults": {
            "n_sites": 8,
            "bias_range_mev": [0.0, 20.0],
            "bias_points": 81,
            "broadening_mev": 0.25,
            "observable": "total_spin",
            "cutoff_mev": 20.0,
            "output_points": 61,
            "model": "ridge",
        },
    },
    {
        "system_type": "homogeneous_xxz_j1j2j3_dmi_impurity",
        "title": "XXZ + J2 + J3 + DMI with impurities",
        "recovers": "J1_xy, J2, J3, Jz and a D_z magnitude",
        "when": (
            "You want DMI. Impurities carrying transverse anisotropy break the "
            "symmetry that hides it. Screen your arrangement first on the DMI "
            "page: two impurities at distinct sites is the minimum, and three "
            "worked best in testing."
        ),
        "view": "global",
        "coupling_mode": "per_parameter",
        "couplings": [
            {"name": "J1_xy", "low": 2.0, "high": 6.0},
            {"name": "J2", "low": -1.5, "high": 1.5},
            {"name": "J3", "low": -1.0, "high": 1.0},
            {"name": "Jz", "low": 2.0, "high": 6.0},
            {"name": "D_z", "low": 0.3, "high": 2.5, "min": 0.0},
        ],
        "supports_impurities": True,
        "supports_field": True,
        "default_impurities": [
            {"site": 1, "spin": "S=1", "transverse_mev": 2.0, "axial_mev": 0.0},
            {"site": 4, "spin": "S=1", "transverse_mev": 2.0, "axial_mev": 0.0},
            {"site": 6, "spin": "S=1", "transverse_mev": 2.0, "axial_mev": 0.0},
        ],
        "defaults": {
            "n_sites": 8,
            "bias_range_mev": [0.0, 20.0],
            "bias_points": 81,
            "broadening_mev": 0.25,
            "observable": "total_spin",
            "cutoff_mev": 20.0,
            "output_points": 61,
            "model": "ridge",
        },
    },
)

_MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "ridge",
        "title": "Ridge regression",
        "notes": "Fast and deterministic. A strong baseline; it won on the DMI dataset.",
        "needs_tensorflow": False,
        "options": [
            {"name": "alpha", "label": "regularisation strength", "default": 0.001,
             "type": "number"},
        ],
    },
    {
        "name": "random_forest",
        "title": "Random forest",
        "notes": "Handles non-linearity without tuning. Slower and larger on disk.",
        "needs_tensorflow": False,
        "options": [
            {"name": "n_estimators", "label": "number of trees", "default": 400,
             "type": "integer"},
            {"name": "min_samples_leaf", "label": "minimum samples per leaf",
             "default": 2, "type": "integer"},
        ],
    },
    {
        "name": "keras_mlp",
        "title": "Neural network (MLP)",
        "notes": (
            "Used by the published inhomogeneous model. Needs the ml extra "
            "(pip install \"hamlet-toolkit[ml]\")."
        ),
        "needs_tensorflow": True,
        "options": [],
    },
    {
        "name": "keras_cnn",
        "title": "Neural network (CNN)",
        "notes": "Convolutional over the bias axis. Needs the ml extra.",
        "needs_tensorflow": True,
        "options": [],
    },
)

_PRESET_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "standard",
        "title": "Standard",
        "notes": "Three seeds, full training. Use this for anything you will rely on.",
    },
    {
        "name": "quick",
        "title": "Quick",
        "notes": (
            "One seed, few epochs. For checking a pipeline runs; artifacts are "
            "marked development-only and the advisor refuses them by default."
        ),
    },
)


def _reference_rate() -> float:
    """The project's own simulation-cost anchor, so the form quotes one number."""
    from ..project import REFERENCE_SECONDS_PER_CORRELATOR_L8

    return float(REFERENCE_SECONDS_PER_CORRELATOR_L8)


def describe_builder_options() -> dict[str, Any]:
    """Everything the guided form needs in order to render itself."""
    try:
        import tensorflow  # noqa: F401

        tensorflow_available = True
    except Exception:  # noqa: BLE001
        tensorflow_available = False

    return {
        "systems": [dict(spec) for spec in _SYSTEM_SPECS],
        "models": [dict(spec) for spec in _MODEL_SPECS],
        "presets": [dict(spec) for spec in _PRESET_SPECS],
        "observables": [
            {
                "name": "Sz",
                "title": "Sz only",
                "notes": "The longitudinal autocorrelator alone.",
            },
            {
                "name": "total_spin",
                "title": "Total spin (Sxx + Syy + Szz)",
                "notes": "Weighted sum of all three components; the DMI work uses this.",
            },
        ],
        "spins": ["S=1", "S=3/2", "S=2", "S=5/2"],
        "tensorflow_available": tensorflow_available,
        "reference_seconds_per_correlator_l8": _reference_rate(),
    }


def _spec_for(system_type: str) -> dict[str, Any]:
    for spec in _SYSTEM_SPECS:
        if spec["system_type"] == system_type:
            return spec
    raise ValueError(f"unknown system_type {system_type!r}")


def build_project_config(form: dict[str, Any], *, workspace: Path | None = None) -> dict[str, Any]:
    """Turn the form's answers into a validated project configuration.

    The configuration file is written where the run can find it but is not
    presented to the user: the point of the guided builder is that nobody has
    to read or edit YAML. It is still a real file on disk, so a run started
    from the interface is reproducible from the command line afterwards.
    """
    from ..project import ProjectConfig

    spec = _spec_for(str(form["system_type"]))
    root = Path(workspace) if workspace else (REPO_ROOT / "results" / "gui-projects")
    name = str(form.get("name") or "").strip() or f"{spec['system_type']}_project"
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.lower())
    project_dir = root / slug
    project_dir.mkdir(parents=True, exist_ok=True)

    # Combinations the configuration layer accepts but training would later
    # refuse. Catching them here is the whole point of a guided form: the
    # alternative is discovering it after the generation stage has run.
    n_sites = int(form["n_sites"])
    if spec["view"] == "local_bonds" and n_sites < 3:
        raise ValueError(
            "the local sliding window spans three sites, so a bond-resolved "
            f"model needs at least 3 sites; {n_sites} was requested"
        )
    low, high = (float(v) for v in form["bias_range_mev"])
    cutoff = float(form["cutoff_mev"])
    if not low <= cutoff <= high:
        raise ValueError(
            f"the cutoff ({cutoff:g} meV) has to lie inside the simulated bias "
            f"window ({low:g} to {high:g} meV)"
        )
    if int(form["output_points"]) > int(form["bias_points"]):
        raise ValueError(
            f"asking for {form['output_points']} output points from "
            f"{form['bias_points']} simulated points would interpolate beyond "
            "the resolution actually simulated"
        )
    if spec["supports_impurities"]:
        sites = [int(item["site"]) for item in form.get("impurities") or ()]
        if any(site >= n_sites for site in sites):
            raise ValueError(
                f"impurity sites {sorted(s for s in sites if s >= n_sites)} lie "
                f"outside a {n_sites}-site chain (sites are numbered from 0)"
            )

    generate: dict[str, Any] = {
        "system": spec["system_type"],
        "output": str(project_dir / "dataset.npz"),
        "n_sites": n_sites,
        "n_samples": int(form["n_samples"]),
        "bias_range_mev": [float(v) for v in form["bias_range_mev"]],
        "bias_points": int(form["bias_points"]),
        "broadening_mev": float(form["broadening_mev"]),
        "observable": str(form["observable"]),
        "output_quantity": "didv",
        "backend": "dmrgpy",
        "seed": int(form.get("seed", 42)),
    }
    if form["observable"] == "total_spin":
        generate["observable_weights"] = [
            float(v) for v in form.get("observable_weights", [1.0, 1.0, 1.0])
        ]
    ranges = [[float(low), float(high)] for low, high in form["coupling_ranges_mev"]]
    if spec["coupling_mode"] == "single_range":
        generate["coupling_range_mev"] = ranges[0]
    else:
        generate["coupling_ranges_mev"] = ranges
    if spec["supports_impurities"] and form.get("impurities"):
        generate["impurities"] = [
            {
                "site": int(item["site"]),
                "spin": str(item.get("spin", "S=1")),
                "axial_mev": float(item.get("axial_mev", 0.0)),
                "transverse_mev": float(item.get("transverse_mev", 0.0)),
            }
            for item in form["impurities"]
        ]
    if spec["supports_field"]:
        generate["transverse_field_mev"] = float(form.get("transverse_field_mev", 0.0))

    payload: dict[str, Any] = {
        "config_schema_version": 1,
        "name": name,
        "system_type": spec["system_type"],
        "output_dir": str(project_dir / "run"),
        "dataset": {"format": "generated", "generate": generate},
        "training": {
            "cutoffs_mev": [float(form["cutoff_mev"])],
            "manual_cutoff_mev": float(form["cutoff_mev"]),
            "output_points": int(form["output_points"]),
            "view": spec["view"],
            "model": str(form["model"]),
            "preset": str(form.get("preset", "standard")),
        },
    }
    options = {k: v for k, v in (form.get("model_options") or {}).items() if v not in (None, "")}
    if options:
        payload["training"]["model_options"] = options

    config_path = project_dir / "project.yaml"
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("writing a configuration requires PyYAML") from exc
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    # Validate by loading it back the way a run would, so an impossible
    # combination is reported now rather than hours into a job.
    try:
        ProjectConfig.from_file(config_path)
    except Exception as exc:
        raise ValueError(f"those settings are not a valid project: {exc}") from exc

    return {"config_path": str(config_path), "project_dir": str(project_dir), "name": name}


def _build_family(form: dict[str, Any]):
    """The sampler for one set of form answers.

    Module level so a worker process can rebuild it from the plain form rather
    than having a family pickled across the process boundary.
    """
    from ..systems import (
        HomogeneousHeisenbergFamily,
        HomogeneousXXZDMIImpurityFamily,
        HomogeneousXXZDMILongRangeFamily,
        HomogeneousXXZLongRangeFamily,
        InhomogeneousHeisenbergFamily,
        SiteImpurity,
    )

    system_type = str(form["system_type"])
    n_sites = int(form["n_sites"])
    ranges = tuple((float(low), float(high)) for low, high in form["coupling_ranges_mev"])

    if system_type == "inhomogeneous_heisenberg":
        return InhomogeneousHeisenbergFamily(n_sites, ranges[0])
    if system_type == "homogeneous_heisenberg":
        return HomogeneousHeisenbergFamily(n_sites, ranges)
    if system_type == "homogeneous_xxz_j1j2j3":
        return HomogeneousXXZLongRangeFamily(n_sites, ranges)
    if system_type == "homogeneous_xxz_j1j2j3_dmi":
        return HomogeneousXXZDMILongRangeFamily(n_sites, ranges)
    impurities = tuple(
        SiteImpurity(
            int(item["site"]),
            str(item.get("spin", "S=1")),
            axial_mev=float(item.get("axial_mev", 0.0)),
            transverse_mev=float(item.get("transverse_mev", 0.0)),
        )
        for item in form.get("impurities") or ()
    )
    return HomogeneousXXZDMIImpurityFamily(
        n_sites,
        ranges,
        impurities=impurities,
        transverse_field_mev=float(form.get("transverse_field_mev", 0.0)),
    )


def _build_protocol(form: dict[str, Any]):
    from ..simulation import SpectroscopyProtocol

    weights = (
        tuple(float(v) for v in form.get("observable_weights", (1.0, 1.0, 1.0)))
        if form["observable"] == "total_spin"
        else None
    )
    return SpectroscopyProtocol.uniform(
        tuple(float(v) for v in form["bias_range_mev"]),
        points=int(form["bias_points"]),
        broadening_mev=float(form["broadening_mev"]),
        observable=str(form["observable"]),
        observable_weights=weights,
        output_quantity="didv",
    )


def preview_sites(form: dict[str, Any], *, max_sites: int = 4) -> tuple[int, ...]:
    """Representative sites to evaluate for a preview.

    Simulation cost is linear in the number of correlators evaluated, and a
    correlator is computed per site, so previewing every site of a chain is the
    single largest avoidable expense. The ends are included because an open
    chain differs most there, and any impurity sites because that is where the
    physics being previewed actually lives.
    """
    n_sites = int(form["n_sites"])
    if n_sites <= max_sites:
        return tuple(range(n_sites))

    wanted: list[int] = [0, n_sites - 1]
    for item in form.get("impurities") or ():
        site = int(item["site"])
        if 0 <= site < n_sites:
            wanted.append(site)
    # Fill any remaining slots with evenly spaced interior sites.
    for site in np.linspace(0, n_sites - 1, max_sites).round().astype(int).tolist():
        wanted.append(int(site))
    ordered: list[int] = []
    for site in wanted:
        if site not in ordered:
            ordered.append(site)
        if len(ordered) == max_sites:
            break
    return tuple(sorted(ordered))


def _preview_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Simulate one sample chain. Runs in its own process and directory.

    The directory matters: the DMRGPy backend writes scratch state (.mpsfolder,
    .pychainfolder) into the working directory, so two workers sharing one
    would corrupt each other's runs.
    """
    import os
    import tempfile

    from ..simulation import DmrgpySimulator

    form = task["form"]
    workdir = tempfile.mkdtemp(prefix="hamlet-preview-")
    os.chdir(workdir)

    family = _build_family(form)
    protocol = _build_protocol(form)
    chain = family.sample(np.random.default_rng(int(task["seed"])))
    simulator = DmrgpySimulator(
        dynamics_mode=task["dynamics_mode"],
        evaluate_sites=tuple(task["sites"]),
        # A DMRG preview carries truncation error, which shows up as a small
        # imaginary residue in these Hermitian autocorrelators. The strict
        # research default would reject a perfectly usable preview.
        max_relative_imaginary_residue=(
            1e-3 if task["dynamics_mode"] == "DMRG" else 1e-6
        ),
    )
    result = simulator.simulate(chain, protocol)
    spectra = np.asarray(result.spectral_map, dtype=float)
    return {
        "index": int(task["index"]),
        "couplings_mev": np.asarray(chain.as_array(), dtype=float).tolist(),
        "sites": [row.tolist() for row in spectra],
        "bias_mev": np.asarray(result.bias_mev, dtype=float).tolist(),
    }


def preview_samples(
    form: dict[str, Any],
    *,
    n_samples: int = 1,
    max_sites: int = 4,
    workers: int | None = None,
) -> dict[str, Any]:
    """Simulate a few chains with the chosen settings and return them to plot.

    Worth the wait before committing to a full run: it is the only way to see
    that the bias window contains the excitations, that the broadening is not
    washing them out, and that the couplings produce structure at all.

    Two things keep it quick, neither of which changes the physics being shown:
    only a few representative sites are evaluated, and DMRG replaces exact
    diagonalisation once the basis is too large for ED to be the cheap option.
    What was done is reported back so the page can say so rather than implying
    a full simulation.

    Running the chains in separate processes was tried and measured at 0.57x,
    i.e. almost twice as slow as doing them one after another: a single
    simulation already keeps several cores busy, so a second process
    oversubscribes rather than adding throughput. ``workers`` is left as an
    escape hatch for a machine where that does not hold, but the default is
    sequential because that is what measured faster here.
    """
    from concurrent.futures import ProcessPoolExecutor

    from ..simulation.dmrgpy import hilbert_dimension, recommended_dynamics_mode

    n_samples = max(1, int(n_samples))
    sites = preview_sites(form, max_sites=max_sites)
    probe = _build_family(form).sample(np.random.default_rng(0))
    dynamics_mode = recommended_dynamics_mode(probe)
    dimension = hilbert_dimension(probe)
    n_sites = int(form["n_sites"])

    correlators = len(sites) * (3 if form["observable"] == "total_spin" else 1)
    print(
        f"{n_samples} chain(s), {len(sites)} of {n_sites} sites, "
        f"{correlators} correlators each"
    )
    print(f"basis dimension {dimension}, so using {dynamics_mode}")

    tasks = [
        {
            "form": form,
            "seed": int(form.get("seed", 42)) + index,
            "sites": sites,
            "dynamics_mode": dynamics_mode,
            "index": index,
        }
        for index in range(n_samples)
    ]

    started = time.time()
    if (workers or 1) <= 1:
        results = []
        for position, task in enumerate(tasks, start=1):
            results.append(_preview_worker(task))
            print(f"chain {position} of {n_samples} done")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_preview_worker, tasks))
    results.sort(key=lambda item: item["index"])
    print(f"done in {time.time() - started:.0f} s")

    family = _build_family(form)
    return {
        "bias_mev": results[0]["bias_mev"],
        "target_names": list(family.parameter_names),
        "evaluated_sites": list(sites),
        "n_sites": n_sites,
        "showing_all_sites": len(sites) == n_sites,
        "dynamics_mode": dynamics_mode,
        "hilbert_dimension": dimension,
        "samples": [
            {"couplings_mev": item["couplings_mev"], "sites": item["sites"]}
            for item in results
        ],
    }
