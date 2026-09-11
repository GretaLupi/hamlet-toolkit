"""JSON-able operations behind the browser interface.

Kept free of HTTP so each operation can be tested directly, and so the
interface cannot acquire behaviour that the library does not have. Everything
here either reads state or plans work; the only things that *run* work go
through :class:`JobRegistry`, which keeps them off the request thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import threading
import time
import traceback
from typing import Any, Callable, Mapping

import numpy as np

from ..cancellation import CancelToken, OperationCancelled, check_cancelled

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]

# Anything the interface writes is capped here. Measurements are small -- a
# per-site dI/dV map is a few hundred kB -- so a file this large is a mistake
# worth refusing rather than a dataset worth accepting.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

# Suffixes the file browser offers. Everything else is hidden rather than
# listed and then rejected on read.
DATA_SUFFIXES = frozenset(
    {".csv", ".tsv", ".dat", ".txt", ".npz", ".json", ".yaml", ".yml"}
)


def _is_source_checkout() -> bool:
    return (REPO_ROOT / "pyproject.toml").exists()


def _workspace_base() -> Path:
    """Where everything the interface writes lives.

    A source checkout keeps its output beside the rest of the project, which is
    what a developer expects. An installed package must not: ``REPO_ROOT`` is
    then inside site-packages, and writing results there would put a user's
    measurements somewhere pip can delete. ``HAMLET_WORKSPACE`` overrides both.
    """
    import os

    override = os.environ.get("HAMLET_WORKSPACE")
    if override:
        return Path(override).expanduser().resolve()
    if _is_source_checkout():
        return REPO_ROOT / "results"
    return Path.home() / ".hamlet" / "workspace"


def _published_root() -> Path:
    """Published artifacts shipped inside both source and wheel installs."""
    return PACKAGE_ROOT / "resources" / "models"


def _examples_root() -> Path:
    return REPO_ROOT / "examples"


def _workspace_root() -> Path:
    """Where runs started from the interface put their artifacts."""
    return _workspace_base() / "gui-projects"


def _uploads_root() -> Path:
    """Where files dropped onto the page are kept."""
    return _workspace_base() / "gui-uploads"


def _screening_root() -> Path:
    """Where DMI screening configurations built by the form are kept."""
    return _workspace_base() / "gui-screenings"


def _analysis_root() -> Path:
    """Where measurements analysed through the interface put their reports."""
    return _workspace_base() / "gui-analyses"


def _experiment_root() -> Path:
    """Where folders of raw, per-site STS files are imported."""
    return _workspace_base() / "gui-experiments"


# What each place under the workspace is for, in the order a run fills them.
# Kept beside the roots themselves so a new one cannot be added without a
# description, which is how a folder ends up on disk that nobody can explain.
OUTPUT_LOCATIONS: tuple[tuple[str, str, str], ...] = (
    ("Uploads", "gui-uploads", "Copies of files dropped onto the page."),
    (
        "Imported experiments",
        "gui-experiments",
        "Folders of raw per-site STS files, converted to one measurement.",
    ),
    (
        "Projects",
        "gui-projects",
        "One folder per training run: its configuration, generated dataset, "
        "and the trained artifact.",
    ),
    (
        "Analyses",
        "gui-analyses",
        "One folder per inference: report.html, summary.png, couplings.csv "
        "and report.json.",
    ),
    ("DMI designs", "gui-screenings", "Saved sample-design screening configurations."),
)


def describe_output_locations() -> dict[str, Any]:
    """Every directory the interface writes to, and what lands in each.

    Asked for often enough to be worth answering unprompted: the workspace is
    not in the same place for everyone. A source checkout writes beside the
    project, an installed package writes under the home directory -- it cannot
    write beside itself, because that is inside site-packages, where pip may
    delete a user's measurements on the next upgrade -- and
    ``HAMLET_WORKSPACE`` overrides both. Someone who does not know which of
    those applies to them cannot find their own results.
    """
    import os

    base = _workspace_base()
    override = os.environ.get("HAMLET_WORKSPACE")
    locations = []
    for title, name, purpose in OUTPUT_LOCATIONS:
        path = base / name
        entries = sorted(path.iterdir()) if path.is_dir() else []
        locations.append(
            {
                "title": title,
                "path": str(path),
                "purpose": purpose,
                "exists": path.is_dir(),
                "entries": len(entries),
                "latest": str(entries[-1].name) if entries else "",
            }
        )
    return {
        "workspace": str(base),
        "locations": locations,
        "source_checkout": _is_source_checkout(),
        "override": override or "",
        "explanation": (
            f"This is a source checkout, so results are kept beside the project "
            f"in {base}."
            if _is_source_checkout() and not override
            else (
                f"HAMLET_WORKSPACE is set, so everything is written under {base}."
                if override
                else f"HamLeT is installed as a package, so results are kept in "
                f"{base} rather than beside the installed files, which pip "
                f"replaces on upgrade."
            )
        ),
    }


def _readable_roots() -> tuple[Path, ...]:
    """Directories the interface is willing to serve files back out of.

    Reading is otherwise unrestricted -- this server already opens any path the
    user types in order to inspect a measurement -- but *serving a file to the
    browser* is a different act, and is confined to what the interface itself
    produced.
    """
    return (
        _workspace_root(),
        _uploads_root(),
        _screening_root(),
        _analysis_root(),
        _experiment_root(),
        _published_root(),
    )


def use_headless_plotting() -> None:
    """Draw figures without a GUI backend.

    Every figure this interface produces is saved to a file; none is shown. The
    default backend may be an interactive one, and matplotlib documents
    creating a figure on such a backend from a worker thread as likely to fail
    -- which is exactly where every job here runs. Selecting Agg turns that
    from unlikely-to-work into cannot-arise. It is a process-wide setting, but
    the process is a server that has no window of its own.
    """
    try:
        import matplotlib
    except ImportError:  # pragma: no cover - plotting is optional
        return
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)


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
                    "Opens a prepared measurement or imports per-site STS "
                    "files. Reports the chain length, bias range, missing "
                    "values, and spectra for every site."
                ),
                "needs": "a measurement file, or a folder with one .dat/.txt spectrum per site",
                "cost": "seconds",
            },
            {
                "id": "advise",
                "have": "I have data and want to know if an existing model fits it",
                "does": (
                    "Checks the physical system, chain length, bias cutoff, "
                    "observable, and parameter ranges against each available "
                    "model."
                ),
                "needs": "a measurement, and the bias cutoff you have chosen",
                "cost": "seconds",
            },
            {
                "id": "models",
                "have": "I want to see which models already exist",
                "does": (
                    "Lists the training data, test accuracy, and validity "
                    "conditions for each model."
                ),
                "needs": "nothing",
                "cost": "instant",
            },
            {
                "id": "analyse",
                "have": "I have data and a model that fits it — give me the couplings",
                "does": (
                    "Applies a compatible model and exports the coupling table, "
                    "an HTML report, and a quality-control figure."
                ),
                "needs": "a measurement, and a model the reuse check accepted",
                "cost": "under a minute",
            },
            {
                "id": "train",
                "have": "I need a model for my own system",
                "does": (
                    "Creates a simulation and training configuration, estimates "
                    "the compute time, and lists the output files."
                ),
                "needs": "the physical system and measurement settings",
                "cost": "configuration takes seconds; training may take hours",
            },
            {
                "id": "dmi",
                "have": "I want to measure DMI",
                "does": (
                    "Tests whether candidate impurity arrangements or transverse "
                    "fields break the symmetry that hides DMI."
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


# --- choosing a file --------------------------------------------------------
# Typing an absolute path is a real barrier for the audience this is for, and a
# silent source of typos. Two mechanisms replace it, because the file can be in
# two different places: dropped from the machine running the browser, or
# already sitting on the machine running the server (the usual case over an SSH
# tunnel to a cluster).

def _safe_upload_name(name: str) -> str:
    """A file name that cannot escape the uploads directory or hide its type.

    The stem and the suffix are cleaned separately so the extension survives
    intact: the loader dispatches on it, and a measurement whose ``.npz`` became
    ``_npz`` would be read as a CSV and fail confusingly. Runs of substituted
    characters collapse, because this name is what the user sees quoted back in
    every later manifest.
    """
    candidate = Path(str(name)).name
    stem, dot, suffix = candidate.rpartition(".")
    if not dot:
        stem, suffix = candidate, ""

    def clean(text: str) -> str:
        substituted = "".join(
            character if (character.isalnum() or character in "-_") else "_"
            for character in text
        )
        while "__" in substituted:
            substituted = substituted.replace("__", "_")
        return substituted.strip("._-")

    stem = clean(stem)
    suffix = clean(suffix)
    if not stem:
        return "measurement"
    return f"{stem}.{suffix}" if suffix else stem


def save_upload(name: str, data: bytes) -> dict[str, Any]:
    """Store a file the browser sent and report where it landed.

    Uploads are kept rather than read and discarded: every later stage records
    the measurement's path in a manifest, and a path that stopped existing the
    moment the request ended would make those records useless.
    """
    if not data:
        raise ValueError("that file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"that file is {len(data) / 1e6:.0f} MB; the interface accepts up to "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB. A per-site dI/dV map is "
            "normally well under a megabyte, so check you sent the right file."
        )
    root = _uploads_root()
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = root / f"{stamp}-{_safe_upload_name(name)}"
    counter = 1
    while destination.exists():
        counter += 1
        destination = root / f"{stamp}-{counter}-{_safe_upload_name(name)}"
    destination.write_bytes(data)
    return {
        "path": str(destination),
        "name": destination.name,
        "size_bytes": len(data),
        "root": str(root),
    }


def save_folder_upload(
    session: str, folder_name: str, name: str, data: bytes
) -> dict[str, Any]:
    """Store one member of a browser-selected experiment folder.

    The browser sends the files individually so it can show progress without
    first creating a potentially large archive in memory. A short session key
    keeps the files from one selection together and separate from every other
    upload. Only flat basenames are accepted by design: one folder is one
    measurement and one file is one site.
    """
    if not data:
        raise ValueError(f"{name}: that file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"{name}: that file is {len(data) / 1e6:.0f} MB; the interface "
            f"accepts up to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB per spectrum"
        )
    safe_session = _safe_upload_name(session).rsplit(".", 1)[0]
    safe_folder = _safe_upload_name(folder_name).rsplit(".", 1)[0]
    if not safe_session:
        raise ValueError("folder upload has no valid session identifier")
    directory = _uploads_root() / "folders" / f"{safe_folder}-{safe_session}"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / _safe_upload_name(name)
    if destination.exists():
        raise ValueError(
            f"the selected folder contains more than one file named {destination.name!r}; "
            "site-spectrum filenames must be unique"
        )
    destination.write_bytes(data)
    return {
        "path": str(destination),
        "folder_path": str(directory),
        "name": destination.name,
        "size_bytes": len(data),
        "root": str(_uploads_root()),
    }


def browse_directory(
    path: str | Path | None = None, *, max_entries: int = 400
) -> dict[str, Any]:
    """List one directory, for choosing a file that is already on this machine.

    Only directories and files this workflow can actually read are listed;
    everything else is counted and reported, so an empty-looking directory is
    distinguishable from one full of files of the wrong kind.
    """
    target = Path(path).expanduser().resolve() if path else Path.home().resolve()
    if not target.exists():
        raise FileNotFoundError(f"no such directory: {target}")
    if not target.is_dir():
        # Being handed a file is a natural mistake when pasting a path; show
        # the directory it lives in rather than refusing.
        target = target.parent

    directories: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    hidden_kinds = 0
    truncated = False
    try:
        entries = sorted(target.iterdir(), key=lambda item: item.name.lower())
    except PermissionError as exc:
        raise PermissionError(f"not allowed to read {target}") from exc
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if len(directories) + len(files) >= max_entries:
            truncated = True
            break
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue
        if is_dir:
            directories.append({"name": entry.name, "path": str(entry), "kind": "directory"})
        elif entry.suffix.lower() in DATA_SUFFIXES:
            try:
                size = entry.stat().st_size
            except OSError:
                size = None
            files.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "kind": "file",
                    "size_bytes": size,
                }
            )
        else:
            hidden_kinds += 1

    shortcuts = [
        {"label": "Home", "path": str(Path.home())},
        {"label": "Uploads", "path": str(_uploads_root())},
        {"label": "Your runs", "path": str(_workspace_root())},
    ]
    if _is_source_checkout():
        shortcuts.append({"label": "Examples", "path": str(_examples_root())})
    raw_sts_file_count = sum(
        1
        for item in entries
        if not item.name.startswith(".")
        and item.is_file()
        and item.suffix.lower() in {".dat", ".txt"}
    )
    return {
        "path": str(target),
        "parent": str(target.parent) if target.parent != target else None,
        "entries": directories + files,
        "n_hidden_other_files": hidden_kinds,
        "truncated": truncated,
        "readable_suffixes": sorted(DATA_SUFFIXES),
        "raw_sts_file_count": raw_sts_file_count,
        "selectable_as_measurement": bool(raw_sts_file_count),
        "shortcuts": [item for item in shortcuts if Path(item["path"]).exists()],
    }


def resolve_readable_file(path: str | Path) -> Path:
    """A file the interface may serve back to the browser, or an error.

    Confined to what the interface produced: a report, a figure, a coupling
    table, a model card. Serving arbitrary paths would turn a local analysis
    tool into a file server for the whole machine.
    """
    target = Path(path).expanduser().resolve()
    roots = [root.resolve() for root in _readable_roots()]
    if not any(target == root or root in target.parents for root in roots):
        raise PermissionError(
            "the interface only serves files it produced; "
            f"{target} is outside {[str(root) for root in roots]}"
        )
    if not target.is_file():
        raise FileNotFoundError(f"no such file: {target}")
    return target


# --- experiments ------------------------------------------------------------

def _natural_filename_key(path: Path) -> list[tuple[int, str | int]]:
    """Sort site 2 before site 10 while retaining predictable text ordering."""
    return [
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", path.name)
    ]


def _sectioned_table_headers(path: Path) -> tuple[str, ...]:
    """Read the Nanonis table header without loading the numeric data."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        marker = next(i for i, line in enumerate(lines) if line.strip() == "[DATA]")
    except StopIteration as exc:
        raise ValueError(
            f"{path.name}: no [DATA] section was found. Automatic folder import "
            "currently recognises Nanonis-style STS exports; use an import recipe "
            "for another laboratory format."
        ) from exc
    if marker + 1 >= len(lines):
        raise ValueError(f"{path.name}: [DATA] is not followed by a column header")
    return tuple(value.strip() for value in lines[marker + 1].split("\t"))


def _common_column(
    header_sets: list[set[str]], candidates: tuple[str, ...], label: str
) -> str:
    for candidate in candidates:
        if all(candidate in headers for headers in header_sets):
            return candidate
    available = sorted(set.intersection(*header_sets)) if header_sets else []
    raise ValueError(
        f"the folder does not have one common {label} column in every spectrum; "
        f"expected one of {list(candidates)}, common columns are {available}"
    )


def prepare_measurement_input(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Turn a folder of one-site Nanonis spectra into one canonical measurement.

    Files remain untouched. The generated NPZ, CSV, import report and preview
    live in the GUI workspace. A content fingerprint makes repeat inspection of
    the same unchanged directory reuse that import instead of producing copies.
    """
    given = Path(path).expanduser().resolve()
    if not given.exists():
        raise FileNotFoundError(f"no such measurement or folder: {given}")
    if given.is_file():
        return given, {}
    if not given.is_dir():
        raise ValueError(f"measurement input is neither a file nor directory: {given}")

    raw_files = sorted(
        (
            item for item in given.iterdir()
            if item.is_file()
            and not item.name.startswith(".")
            and item.suffix.lower() in {".dat", ".txt"}
        ),
        key=_natural_filename_key,
    )
    if not raw_files:
        raise ValueError(
            f"{given} contains no .dat or .txt spectra. Choose the folder that "
            "contains one STS export per measured site."
        )

    headers = [set(_sectioned_table_headers(item)) for item in raw_files]
    energy_column = _common_column(
        headers, ("Bias calc (V)", "Bias (V)"), "bias"
    )
    didv_column = _common_column(
        headers, ("LI Demod 1 X (A)", "LI Demod 1 X(A)"), "dI/dV"
    )
    d2_candidates = ("LI Demod 2 X (A)", "LI Demod 2 X(A)")
    d2_column = next(
        (candidate for candidate in d2_candidates if all(candidate in h for h in headers)),
        None,
    )

    fingerprint = hashlib.sha256()
    fingerprint.update(str(given).encode())
    for item in raw_files:
        stat = item.stat()
        fingerprint.update(f"{item.name}\0{stat.st_size}\0{stat.st_mtime_ns}".encode())
    digest = fingerprint.hexdigest()[:12]
    output_dir = _experiment_root() / f"{_slug(given.name) or 'experiment'}-{digest}"
    measurement_path = output_dir / "measurement.npz"
    report_path = output_dir / "import_report.json"
    if not measurement_path.exists():
        from ..io import TextImportRecipe, import_text_measurement

        signals: dict[str, Any] = {
            "didv": {"column": didv_column, "unit": "A"},
        }
        if d2_column is not None:
            # Preserved for plotting and QC. The inverse model still receives
            # only the primary dI/dV channel.
            signals["d2idv2"] = {"column": d2_column, "unit": "A"}
        recipe = TextImportRecipe.from_mapping(
            {
                "input": {"files": [str(item) for item in raw_files]},
                "format": {
                    "delimiter": "tab",
                    "data_marker": "[DATA]",
                    "header_offset": 1,
                    "encoding": "utf-8",
                },
                "columns": {
                    "energy": {"column": energy_column, "unit": "V"},
                    "primary": "didv",
                    "signals": signals,
                },
                "site": {"mode": "sequential", "start": 1},
                "processing": {
                    "energy_order": "ascending",
                    "grid": "require_equal",
                    "missing": "drop_common",
                },
                "output": {
                    "directory": str(output_dir),
                    "csv": "spectroscopy.csv",
                    "measurement": "measurement.npz",
                    "report": "import_report.json",
                    "preview": "import_preview.html",
                    "include_auxiliary_csv": True,
                    "overwrite": False,
                },
            },
            base=given,
        )
        import_text_measurement(recipe)

    return measurement_path, {
        "input_kind": "sts_folder",
        "input_folder": str(given),
        "source_file_count": len(raw_files),
        "source_files": [item.name for item in raw_files],
        "auxiliary_d2idv2": d2_column is not None,
        "import_report": str(report_path),
    }

def inspect_experiment(path: str | Path, *, max_points: int = 400) -> dict[str, Any]:
    """Read a measurement and report what the workflow will make of it.

    The reported fields are exactly the ones that decide whether a model can be
    reused, so this doubles as a preflight check.
    """
    from ..experiments import load_canonical_experiment

    measurement_path, import_details = prepare_measurement_input(path)
    measurement, source, system_type, view = load_canonical_experiment(
        str(measurement_path)
    )
    bias, spectra = measurement.site_spectra(require_complete=False)
    bias = np.asarray(bias, dtype=float)
    spectra = np.asarray(spectra, dtype=float)

    # Thin for transport; the browser never needs full resolution to show shape.
    if bias.size > max_points:
        keep = np.linspace(0, bias.size - 1, max_points).astype(int)
        bias_out = bias[keep]
    else:
        keep = np.arange(bias.size)
        bias_out = bias

    finite = np.isfinite(spectra)
    site_labels = np.asarray(measurement.axes.get("site", np.arange(spectra.shape[0])))
    channel_plots: dict[str, Any] = {}
    for channel in measurement.channels:
        _, channel_spectra = measurement.site_spectra(
            channel=channel, require_complete=False
        )
        channel_values = np.asarray(channel_spectra, dtype=float)[:, keep]
        role = "inference" if channel == measurement.primary_channel else "plot/QC only"
        symbol = {
            "didv": "dI/dV",
            "d2idv2": "d²I/dV²",
        }.get(channel, channel)
        unit = measurement.channel_units.get(channel, "arbitrary")
        channel_plots[channel] = {
            "bias_mev": bias_out.tolist(),
            "sites": [row.tolist() for row in channel_values],
            "site_labels": site_labels.tolist(),
            "channel": channel,
            "title": symbol,
            "role": role,
            "y_label": f"{symbol} [{unit}]",
        }
    primary_plot = channel_plots[measurement.primary_channel]
    return {
        "source": source,
        "path": str(measurement_path),
        "input": import_details,
        "declared_system_type": system_type,
        "declared_view": view,
        "n_sites": int(spectra.shape[0]),
        "n_bias_points": int(bias.size),
        "bias_min_mev": float(bias.min()),
        "bias_max_mev": float(bias.max()),
        "bias_units": measurement.axis_units.get("bias", "meV"),
        "primary_channel": measurement.primary_channel,
        "channels": list(measurement.channels),
        "is_complete": bool(measurement.is_primary_complete),
        "missing_points": int((~finite).sum()),
        "starts_at_zero": bool(abs(float(bias.min())) < 1e-8),
        "covers_zero": bool(float(bias.min()) <= 0.0 <= float(bias.max())),
        "plot": primary_plot,
        "plots": channel_plots,
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
    measurement_path, import_details = prepare_measurement_input(path)
    decision = advise_experiment(
        str(measurement_path),
        manual_cutoff_mev=float(cutoff_mev),
        artifact_roots=[r for r in roots if Path(r).exists()],
        system_type=system_type,
        view=view,
    )
    return {
        "action": decision.action,
        "path": str(measurement_path),
        "input": import_details,
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
# Screening is a sample-design decision, and the people who take it are the
# ones who will build the sample -- so it is asked as a form about the chain
# and the impurities they can place, not as a configuration file. The file is
# still written, because a design decision that cannot be reproduced from the
# command line later is not much of a record.

_SCREENING_DEFAULTS: dict[str, Any] = {
    "name": "which arrangement can measure DMI?",
    "chain": {
        "n_sites": 8,
        "j_eff_mev": 5.0,
        "d_z_mev": 1.5,
        "jz_mev": 5.5,
        "j2_mev": 0.0,
        "j3_mev": 0.0,
        "site_spin": "S=1/2",
    },
    "protocol": {
        "bias_range_mev": [0.0, 20.0],
        "bias_points": 81,
        "broadening_mev": 0.25,
        "observable": "total_spin",
    },
    "candidates": [
        {"label": "one impurity", "sites": [3], "spin": "S=1",
         "transverse_mev": 2.0, "axial_mev": 0.0, "transverse_field_mev": 0.0},
        {"label": "two impurities", "sites": [1, 6], "spin": "S=1",
         "transverse_mev": 2.0, "axial_mev": 0.0, "transverse_field_mev": 0.0},
        {"label": "three impurities", "sites": [1, 4, 6], "spin": "S=1",
         "transverse_mev": 2.0, "axial_mev": 0.0, "transverse_field_mev": 0.0},
        {"label": "no impurities, 1 meV transverse field", "sites": [], "spin": "S=1",
         "transverse_mev": 0.0, "axial_mev": 0.0, "transverse_field_mev": 1.0},
    ],
}


def describe_screening_options() -> dict[str, Any]:
    """Everything the DMI design form needs in order to render itself.

    The calibration table travels with it because an imprint is meaningless on
    its own: the numbers only mean something against the D_z skill that models
    trained on those designs actually reached.
    """
    from ..dmi_design import (
        HIDDEN_IMPRINT,
        IMPRINT_CALIBRATION,
        PROMISING_IMPRINT,
        STRONG_IMPRINT,
        TOO_WEAK_IMPRINT,
    )

    return {
        "defaults": json.loads(json.dumps(_SCREENING_DEFAULTS)),
        # Every spin, for both the host and the impurities. Which pairings
        # are legal depends on the host -- an impurity must differ from the
        # chain it sits in, and single-ion anisotropy vanishes at S=1/2, so a
        # spin-1/2 impurity cannot break the symmetry whatever the host is.
        # Both rules are enforced where the host is known rather than by
        # pruning a list that cannot see it.
        "spins": ["S=1", "S=3/2", "S=2", "S=5/2"],
        "chain_spins": ["S=1/2", "S=1", "S=3/2", "S=2", "S=5/2"],
        "observables": ["total_spin", "Sz"],
        "workspace_root": str(_screening_root()),
        "calibration": [
            {"imprint": value, "note": note} for value, note in IMPRINT_CALIBRATION
        ],
        "verdicts": [
            {
                "name": "hidden",
                "below": HIDDEN_IMPRINT,
                "means": (
                    "the gauge pair is degenerate to numerical precision: no "
                    "amount of data can constrain D_z"
                ),
            },
            {
                "name": "too weak",
                "below": TOO_WEAK_IMPRINT,
                "means": "measured to train to about zero D_z skill",
            },
            {
                "name": "marginal",
                "below": PROMISING_IMPRINT,
                "means": "between the design that failed and the one that reached 0.19",
            },
            {
                "name": "promising",
                "below": STRONG_IMPRINT,
                "means": "around the 0.19-skill design, which plateaued with more data",
            },
            {
                "name": "strong",
                "below": None,
                "means": "at or above the design that reached 0.54 D_z skill",
            },
        ],
    }


def _screening_candidate(entry: Any, index: int, n_sites: int) -> dict[str, Any]:
    """One arrangement, checked against the chain it claims to sit in."""
    if not isinstance(entry, dict):
        raise ValueError(f"arrangement {index + 1} is not a set of answers")
    raw_sites = entry.get("sites")
    if isinstance(raw_sites, str):
        raw_sites = [part for part in raw_sites.replace(";", ",").split(",") if part.strip()]
    sites: list[int] = []
    for value in raw_sites or ():
        try:
            site = int(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"arrangement {index + 1}: {value!r} is not a site number"
            ) from None
        if not 0 <= site < n_sites:
            raise ValueError(
                f"arrangement {index + 1}: site {site} lies outside a "
                f"{n_sites}-site chain (sites are numbered from 0)"
            )
        if site in sites:
            raise ValueError(
                f"arrangement {index + 1}: site {site} is listed twice; one "
                "impurity per site"
            )
        sites.append(site)
    field = float(entry.get("transverse_field_mev", 0.0) or 0.0)
    if not sites and not field:
        raise ValueError(
            f"arrangement {index + 1} has neither impurities nor a transverse "
            "field, so nothing in it can break the symmetry that hides D_z. "
            "Remove it, or give it something to break the symmetry with."
        )
    transverse = float(entry.get("transverse_mev", 0.0) or 0.0)
    if transverse < 0.0:
        raise ValueError(f"arrangement {index + 1}: transverse anisotropy cannot be negative")
    label = str(entry.get("label") or "").strip()
    if not label:
        label = (
            f"{len(sites)} impurity(s) at {sites}, E={transverse:g}"
            if sites
            else f"field-only, B={field:g}"
        )
        if sites and field:
            label += f", B={field:g}"
    # `spins` gives one species per site; `spin` is the single species used
    # for all of them, and stays accepted because every configuration written
    # before this, and the shipped example, use it.
    given = entry.get("spins")
    if given is not None and entry.get("spin") is not None and given:
        # Matches the library, which refuses a configuration giving both
        # rather than silently preferring one.
        raise ValueError(
            f"arrangement {index + 1} gives both spin and spins; use spins for "
            "one species per site, or spin for one species throughout"
        )
    if not given:
        spins = [str(entry.get("spin", "S=1"))] * len(sites)
    else:
        spins = [str(item) for item in given]
        if len(spins) != len(sites):
            raise ValueError(
                f"arrangement {index + 1} lists {len(spins)} spin(s) for "
                f"{len(sites)} site(s); give one spin per site"
            )
    return {
        "label": label,
        "impurities": [
            {
                "site": site,
                "spin": spin,
                "transverse_mev": transverse,
                "axial_mev": float(entry.get("axial_mev", 0.0) or 0.0),
            }
            for site, spin in zip(sites, spins)
        ],
        "transverse_field_mev": field,
    }


def build_screening_config(
    form: dict[str, Any], *, workspace: Path | None = None
) -> dict[str, Any]:
    """Turn the design form's answers into a validated screening configuration.

    Returns the free symmetry verdict along with the path, because that verdict
    is the whole reason to screen: a design that cannot break the symmetry is
    hopeless before a single simulation runs, and saying so at the moment the
    design is written down is worth more than saying so a minute later.
    """
    from ..dmi_design import load_screening_config

    chain = dict(form.get("chain") or {})
    protocol = dict(form.get("protocol") or {})
    n_sites = int(chain.get("n_sites", 8))
    if n_sites < 2:
        raise ValueError("a chain needs at least two sites")
    j_eff = float(chain.get("j_eff_mev", 5.0))
    d_z = float(chain.get("d_z_mev", 1.5))
    if j_eff <= 0:
        raise ValueError("the exchange scale sqrt(J1_xy^2 + D_z^2) must be positive")
    if not 0.0 < d_z <= j_eff:
        raise ValueError(
            f"the DMI to resolve ({d_z:g} meV) must be positive and no larger "
            f"than the exchange scale it is part of ({j_eff:g} meV): the two "
            "chains being told apart share sqrt(J1_xy^2 + D_z^2)"
        )
    low, high = (float(v) for v in protocol.get("bias_range_mev", (0.0, 20.0)))
    if not low < high:
        raise ValueError(f"the bias window runs from {low:g} to {high:g} meV, which is empty")
    bias_points = int(protocol.get("bias_points", 81))
    if bias_points < 2:
        raise ValueError("a bias window needs at least two points")
    broadening = float(protocol.get("broadening_mev", 0.25))
    if broadening <= 0:
        raise ValueError("broadening must be positive")

    from ..systems.heisenberg import validate_site_spin

    site_spin = str(chain.get("site_spin", "S=1/2") or "S=1/2")
    validate_site_spin(site_spin)

    entries = form.get("candidates") or []
    if not entries:
        raise ValueError("add at least one arrangement to screen")
    candidates = [
        _screening_candidate(entry, index, n_sites) for index, entry in enumerate(entries)
    ]
    # Named here rather than left to escape from the chain class, which knows
    # the rule but not which arrangement on the page broke it.
    for index, candidate in enumerate(candidates):
        clashing = sorted(
            impurity["site"]
            for impurity in candidate["impurities"]
            if impurity["spin"] == site_spin
        )
        if clashing:
            raise ValueError(
                f"arrangement {index + 1} "
                f"({candidate['label'] or 'unnamed'}) puts {site_spin} at "
                f"site(s) {clashing}, which is the chain's own spin; a "
                f"substituted site has to differ from the chain it sits in"
            )

    payload: dict[str, Any] = {
        "screening_schema_version": 1,
        "name": str(form.get("name") or "DMI sample design").strip(),
        "chain": {
            "n_sites": n_sites,
            "j_eff_mev": j_eff,
            "d_z_mev": d_z,
            "jz_mev": float(chain.get("jz_mev", 5.5)),
            "j2_mev": float(chain.get("j2_mev", 0.0)),
            "j3_mev": float(chain.get("j3_mev", 0.0)),
            "site_spin": site_spin,
        },
        "protocol": {
            "bias_range_mev": [low, high],
            "bias_points": bias_points,
            "broadening_mev": broadening,
            "observable": str(protocol.get("observable", "total_spin")),
            "output_quantity": "didv",
        },
        "candidates": candidates,
    }
    if payload["protocol"]["observable"] == "total_spin":
        payload["protocol"]["observable_weights"] = [1.0, 1.0, 1.0]

    root = Path(workspace) if workspace else _screening_root()
    slug = _slug(payload["name"]) or "dmi_screening"
    directory = root / slug
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "screening.yaml"
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("writing a configuration requires PyYAML") from exc
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    # Load it back the way the screening run will, so a design the library
    # refuses is reported now rather than when the job starts.
    try:
        load_screening_config(config_path)
    except Exception as exc:
        raise ValueError(f"those answers are not a valid screening: {exc}") from exc
    return {"config_path": str(config_path), **screening_preview(config_path)}


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.lower()).strip("_")


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


# --- where the work runs -----------------------------------------------------

def describe_compute_options() -> dict[str, Any]:
    """What this machine offers, and what a cluster configuration would need."""
    from ..cluster import RESOURCE_FIELDS, available_profiles
    from ..compute import describe_compute, gpu_unavailable_summary

    import sys

    report = describe_compute()
    no_gpu_here = gpu_unavailable_summary(report)
    devices = [
        {"name": "auto", "title": "Automatic",
         "notes": "Use an available GPU; otherwise use the CPU.",
         "unavailable_here": ""},
        {"name": "cpu", "title": "Force the CPU",
         "notes": "Use CPU training, including on systems with an available GPU.",
         "unavailable_here": ""},
    ]
    # "Require a GPU" is offered on Linux only, because Linux is the only
    # platform where TensorFlow can use one: native Windows has had no GPU
    # build since 2.11 and macOS has no CUDA path at all. Offering a choice
    # that cannot ever be honoured -- even labelled -- is an invitation to
    # spend an afternoon on drivers. It is not needed for cluster work either:
    # "Automatic" already takes a GPU when the job lands on a node that has
    # one, which is how a Windows user reaches a card.
    if sys.platform == "linux":
        devices.append(
            {"name": "gpu", "title": "Require a GPU",
             "notes": "Use a GPU when available; fall back to the CPU otherwise.",
             "unavailable_here": no_gpu_here}
        )
    return {
        **report.to_dict(),
        "devices": devices,
        "gpu_possible_here": sys.platform == "linux",
        "gpu_elsewhere_note": (
            ""
            if sys.platform == "linux"
            else (
                "TensorFlow cannot use a GPU on this platform at all, so the "
                "choice is not offered. Send the run to a cluster instead: "
                "Automatic uses a GPU whenever the job lands on a node that "
                "has one."
            )
        ),
        "gpu_help_url": (
            "https://github.com/GretaLupi/hamlet-toolkit/blob/main/"
            "docs/user-guide.md#using-a-gpu"
        ),
        "cluster": {
            "schedulers": available_profiles(),
            "resources": list(RESOURCE_FIELDS),
            "config_path": str(_cluster_config_path()),
            "configured": _cluster_config_path().exists(),
        },
    }


# What the cluster form asks for, and nothing else. The scheduler profiles
# behind these names carry a dozen directive templates each, but none of that
# is a decision a user makes -- picking "slurm" is the decision, and the
# profile follows from it. Sites that genuinely differ still edit the YAML by
# hand; that escape hatch is the file, not the form.
CLUSTER_SCHEDULER_CHOICES: tuple[tuple[str, str], ...] = (
    ("slurm", "Slurm — sbatch. The usual one in academic HPC."),
    ("pbs", "PBS / Torque — qsub."),
    ("lsf", "LSF — bsub."),
    ("sge", "Grid Engine — qsub."),
    ("none", "No scheduler — run it in the background on that machine."),
)


def build_cluster_config(form: Mapping[str, Any]) -> dict[str, Any]:
    """Turn the cluster form into a saved configuration.

    Deliberately few fields. The address to ssh to and the scheduler name are
    the two facts a user actually has; everything else either has a working
    default or is a resource request they can leave alone.
    """
    host = str(form.get("host", "")).strip()
    if not host:
        raise ValueError(
            "enter the address you ssh to, for example greta@triton.aalto.fi"
        )
    if " " in host:
        raise ValueError(f"that does not look like an ssh address: {host!r}")

    scheduler = str(form.get("scheduler", "slurm")).strip() or "slurm"
    known = {name for name, _ in CLUSTER_SCHEDULER_CHOICES}
    if scheduler not in known:
        raise ValueError(f"scheduler must be one of {sorted(known)}; got {scheduler!r}")

    remote_dir = str(form.get("remote_dir", "")).strip()
    if not remote_dir:
        raise ValueError(
            "enter the directory on the cluster to run in, for example "
            "/scratch/work/yourname/hamlet"
        )

    resources: dict[str, Any] = {}
    for field_name in ("cpus", "gpus"):
        value = form.get(field_name)
        if value not in (None, "", 0, "0"):
            resources[field_name] = int(value)
    for field_name in ("memory", "walltime", "queue", "account"):
        value = str(form.get(field_name, "") or "").strip()
        if value:
            resources[field_name] = value

    setup = form.get("setup") or ""
    if isinstance(setup, str):
        setup = [line.strip() for line in setup.splitlines() if line.strip()]
    else:
        setup = [str(line).strip() for line in setup if str(line).strip()]

    payload: dict[str, Any] = {
        "cluster_schema_version": 1,
        "host": host,
        "remote_dir": remote_dir,
        "scheduler": scheduler,
    }
    if resources:
        payload["resources"] = resources
    if setup:
        payload["setup"] = setup
    python = str(form.get("python", "") or "").strip()
    if python:
        payload["python"] = python

    import yaml

    header = (
        "# Written by the HamLeT interface. Access is over your own ssh, with\n"
        "# keys only -- HamLeT never types a password and never stores one.\n"
    )
    text = header + yaml.safe_dump(payload, sort_keys=False)
    written = write_cluster_config(text)
    return {**written, "settings": payload}


def read_cluster_form() -> dict[str, Any]:
    """The saved settings as form fields, plus the choices the form offers."""
    path = _cluster_config_path()
    saved: dict[str, Any] = {}
    if path.exists():
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, Mapping):
            saved = dict(loaded)
    resources = dict(saved.get("resources") or {})
    setup = saved.get("setup") or []
    if isinstance(setup, str):
        setup = [line for line in setup.splitlines() if line.strip()]
    scheduler = saved.get("scheduler")
    return {
        "configured": path.exists(),
        "config_path": str(path),
        "schedulers": [
            {"name": name, "title": title} for name, title in CLUSTER_SCHEDULER_CHOICES
        ],
        "form": {
            "host": str(saved.get("host") or ""),
            "remote_dir": str(saved.get("remote_dir") or ""),
            # A hand-written `scheduler:` block cannot be shown in a dropdown,
            # so it falls back to the default rather than being misreported.
            "scheduler": scheduler if isinstance(scheduler, str) else "slurm",
            "custom_scheduler": not isinstance(scheduler, (str, type(None))),
            "cpus": resources.get("cpus", ""),
            "gpus": resources.get("gpus", ""),
            "memory": str(resources.get("memory") or ""),
            "walltime": str(resources.get("walltime") or ""),
            "queue": str(resources.get("queue") or ""),
            "account": str(resources.get("account") or ""),
            "setup": "\n".join(str(line) for line in setup),
            "python": str(saved.get("python") or ""),
        },
    }


def _cluster_config_path() -> Path:
    return _workspace_base() / "cluster.yaml"


def read_cluster_config() -> dict[str, Any]:
    """The saved cluster settings, or a starting point if there are none."""
    from ..cluster import EXAMPLE_CLUSTER_CONFIG

    path = _cluster_config_path()
    if path.exists():
        return {"path": str(path), "text": path.read_text(encoding="utf-8"),
                "exists": True}
    return {"path": str(path), "text": EXAMPLE_CLUSTER_CONFIG, "exists": False}


def write_cluster_config(text: str) -> dict[str, Any]:
    """Save cluster settings, refusing anything that will not load.

    Validated before writing so a mistake is caught here rather than at submit
    time, which on a cluster means after a queue wait.
    """
    from ..cluster import ClusterConfig

    path = _cluster_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(".checking")
    scratch.write_text(text, encoding="utf-8")
    try:
        cluster = ClusterConfig.from_file(scratch)
    except Exception as exc:
        scratch.unlink(missing_ok=True)
        raise ValueError(f"those cluster settings are not usable: {exc}") from exc
    scratch.replace(path)
    return {"path": str(path), "saved": True, "summary": cluster.to_dict()}


def check_cluster() -> dict[str, Any]:
    """Confirm the cluster answers and has the scheduler it claims."""
    from ..cluster import ClusterConfig, ClusterSession

    cluster = ClusterConfig.from_file(_cluster_config_path())
    return ClusterSession(cluster).check_connection()


def cluster_script(config_path: str | Path) -> dict[str, Any]:
    """The batch script that would be submitted for one project.

    Shown before anything is sent, because a job script is the thing a cluster
    user will want to check, and often to hand-edit for a local quirk.
    """
    from ..cluster import (
        ClusterConfig,
        make_portable,
        project_command,
        render_job_script,
    )

    cluster = ClusterConfig.from_file(_cluster_config_path())
    project = Path(config_path).expanduser().resolve()
    portable = make_portable(project)
    command = project_command(cluster, project.name)
    return {
        "script": render_job_script(
            cluster, command, job_name=project.parent.name
        ),
        "project_dir": str(project.parent),
        "remote_dir": cluster.remote_dir,
        "host": cluster.host or "this machine",
        "scheduler": cluster.scheduler.name,
        "portable": portable["portable"],
        "outside_project_dir": portable["outside_project_dir"],
    }


def submit_to_cluster(config_path: str | Path) -> dict[str, Any]:
    """Stage a project on the cluster and submit it."""
    from ..cluster import ClusterConfig, ClusterSession

    cluster = ClusterConfig.from_file(_cluster_config_path())
    session = ClusterSession(cluster)
    prepared = cluster_script(config_path)
    directory = Path(prepared["project_dir"])
    print(f"copying {directory} to {cluster.host or 'the working directory'}")
    session.stage(directory).raise_for_status("copying the project")
    print("submitting")
    submitted = session.submit(prepared["script"])
    print(f"job {submitted['job_id']} submitted with {submitted['scheduler']}")
    return {
        "kind": "cluster",
        **submitted,
        "host": cluster.host or "this machine",
        "project_dir": str(directory),
    }


def cluster_job_status(job_id: str) -> dict[str, Any]:
    from ..cluster import ClusterConfig, ClusterSession

    cluster = ClusterConfig.from_file(_cluster_config_path())
    return ClusterSession(cluster).status(str(job_id))


def cancel_cluster_job(job_id: str) -> dict[str, Any]:
    from ..cluster import ClusterConfig, ClusterSession

    cluster = ClusterConfig.from_file(_cluster_config_path())
    return ClusterSession(cluster).cancel(str(job_id))


def fetch_from_cluster(project_dir: str | Path) -> dict[str, Any]:
    from ..cluster import ClusterConfig, ClusterSession

    cluster = ClusterConfig.from_file(_cluster_config_path())
    result = ClusterSession(cluster).fetch(Path(project_dir))
    result.raise_for_status("fetching results")
    return {"kind": "fetch", "project_dir": str(project_dir),
            "output": result.stdout.strip()}


# --- applying a model to a measurement --------------------------------------
# The step that turns a trained model into an answer. It was the one part of
# the workflow the interface could not do: you could train a model here and
# then had to leave for the command line to use it, which is exactly the point
# at which a configuration file reappears.

def _artifact_for(name_or_path: str) -> Path:
    """Resolve a model by the name the interface shows, or by its path.

    A path is accepted on its own terms -- whether it holds a real artifact,
    not whether this interface has heard of it -- so a model trained elsewhere,
    or on a cluster, can be used without first being moved into the workspace.
    """
    candidates: dict[str, Path] = {}
    for directory, origin in _artifact_directories():
        key = directory.name if origin == "published" else _workspace_label(directory)
        candidates.setdefault(key, directory)
        candidates.setdefault(str(directory), directory)
    directory = candidates.get(str(name_or_path))
    if directory is not None:
        return directory

    given = Path(str(name_or_path)).expanduser()
    if given.is_dir():
        manifest = given / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"{given} is a directory but holds no manifest.json, so it is "
                "not a trained model"
            )
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{manifest} is not readable: {exc}") from exc
        if "artifact_schema_version" not in payload:
            raise FileNotFoundError(
                f"{given} holds a manifest, but not a trained model's: no "
                "artifact_schema_version"
            )
        return given.resolve()

    raise FileNotFoundError(
        f"no model named {name_or_path!r}; known models are "
        f"{sorted(k for k in candidates if not k.startswith('/'))}"
    )


def build_analysis_config(
    measurement_path: str | Path,
    model: str,
    *,
    name: str | None = None,
    allow_development_artifacts: bool = False,
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Assemble a project that applies one saved model to one measurement.

    Every contract term -- system, view, cutoff, input resolution -- is copied
    from the model's own manifest rather than asked for again. A cutoff is not
    a preference here: the weights are specific to it, and offering the field
    would only create the opportunity to disagree with the artifact and be
    refused later.
    """
    from ..project import ProjectConfig

    artifact = _artifact_for(model)
    manifest_path = artifact / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{artifact} has no readable manifest: {exc}") from exc
    preprocessing = manifest.get("preprocessing") or {}
    # Each of these is a contract term. A missing one cannot be defaulted --
    # guessing the system or the cutoff is precisely the mistake the contract
    # exists to prevent -- so it is an error rather than a fallback.
    missing = [
        key
        for key, value in (
            ("system_type", manifest.get("system_type")),
            ("view", manifest.get("view")),
            ("preprocessing.bias_cutoff_mev", preprocessing.get("bias_cutoff_mev")),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"{artifact} does not record {', '.join(missing)}, so there is no "
            "contract to apply it under"
        )
    cutoff = preprocessing["bias_cutoff_mev"]

    measurement, _ = prepare_measurement_input(measurement_path)

    label = str(name or "").strip() or f"{measurement.stem} with {artifact.name}"
    root = Path(workspace) if workspace else _analysis_root()
    directory = root / (_slug(label) or "analysis")
    # An analysis is cheap and gets repeated; a run refuses to overwrite its own
    # outputs, so each one gets its own directory rather than a collision the
    # user has to resolve by inventing a new name. The counter matters: two
    # analyses started in the same second is an ordinary thing to do.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = directory / stamp
    counter = 1
    while run_dir.exists():
        counter += 1
        run_dir = directory / f"{stamp}-{counter}"
    run_dir.mkdir(parents=True)

    payload: dict[str, Any] = {
        "config_schema_version": 1,
        "name": label,
        "system_type": str(manifest["system_type"]),
        "artifact": str(artifact),
        "output_dir": str(run_dir / "run"),
        "experiment": {"measurement": str(measurement)},
        "training": {
            "cutoffs_mev": [float(cutoff)],
            "manual_cutoff_mev": float(cutoff),
            "output_points": int(preprocessing.get("output_points", 200)),
            "view": str(manifest["view"]),
            "model": str(manifest.get("model_name", "ridge")),
            "allow_development_artifacts": bool(allow_development_artifacts),
        },
    }
    config_path = run_dir / "analysis.yaml"
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("writing a configuration requires PyYAML") from exc
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    try:
        ProjectConfig.from_file(config_path)
    except Exception as exc:
        raise ValueError(f"that combination is not a valid analysis: {exc}") from exc
    return {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "artifact_path": str(artifact),
        "model_label": artifact.name,
        "cutoff_mev": float(cutoff),
        "system_type": manifest.get("system_type"),
        "view": manifest.get("view"),
        "n_sites": manifest.get("n_sites"),
        "preset": (manifest.get("training_preset") or {}).get("name"),
        "measurement_path": str(measurement),
    }


def run_analysis(
    config_path: str | Path, *, should_stop: Callable[[], bool] | None = None
) -> dict[str, Any]:
    """Apply the configured model and report where the answers were written."""
    from ..project import HamiltonianLearningProject

    # Inference draws the quality-control figure, and this runs on a job thread.
    use_headless_plotting()
    project = HamiltonianLearningProject.from_config(str(config_path))
    print("checking the model against this measurement")
    outcome = project.run(should_stop=should_stop)
    analysis = outcome.analysis_dir
    print(f"status: {outcome.status}")
    couplings = _read_couplings(analysis / "couplings.csv")
    report_path = analysis / "report.json"
    artifact_path = Path(outcome.artifact_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads((artifact_path / "manifest.json").read_text(encoding="utf-8"))
    return {
        "kind": "analysis",
        "status": str(outcome.status),
        "artifact_path": str(artifact_path),
        "model_label": artifact_path.name,
        "model_name": manifest.get("model_name"),
        "view": report.get("view", manifest.get("view")),
        "n_sites": report.get("n_sites", manifest.get("n_sites")),
        "diagnostics": report.get("diagnostics", {}),
        "report_html": str(analysis / "report.html"),
        "report_json": str(report_path),
        "couplings_csv": str(analysis / "couplings.csv"),
        "summary_png": str(analysis / "summary.png"),
        # Reported only when present: the .tex is always written, but the PDF
        # needs a LaTeX toolchain, and offering a link to a file that is not
        # there is worse than not offering one.
        "report_tex": _existing(analysis / "report.tex"),
        "report_pdf": _existing(analysis / "report.pdf"),
        "couplings": couplings,
        "cutoff_mev": outcome.selected_cutoff_mev,
    }


def _existing(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _read_couplings(path: Path, *, max_rows: int = 60) -> dict[str, Any]:
    """The coupling table, small enough to show on the page."""
    import csv

    if not path.exists():
        return {"columns": [], "rows": []}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return {"columns": [], "rows": []}
    return {
        "columns": rows[0],
        "rows": rows[1 : max_rows + 1],
        "truncated": len(rows) - 1 > max_rows,
        "n_rows": len(rows) - 1,
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
    token: CancelToken = field(default_factory=CancelToken)

    @property
    def stopping(self) -> bool:
        """Asked to stop, but not yet at a point where it safely can."""
        return self.token.cancelled and self.status == "running"

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "stopping": self.stopping,
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

    def submit(self, kind: str, label: str, work: Callable[..., Any]) -> GuiJob:
        """Start ``work`` on a worker thread.

        ``work`` may take one argument, in which case it is handed the job's
        :class:`~hamlet.cancellation.CancelToken` and can be stopped. Taking
        none is still allowed, for work short enough that stopping it is not a
        question worth asking.
        """
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
                    job.result = (
                        work(job.token) if _accepts_token(work) else work()
                    )
            except OperationCancelled as exc:
                # A stop the user asked for is not a failure, and reporting it
                # as one would make the Stop button look broken.
                status = "cancelled"
                error = str(exc)
                with self._lock:
                    job.lines.append("stopped at your request")
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

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Ask a running job to stop at its next safe point.

        Cooperative, and says so: a thread cannot be killed, and killing one
        mid-write would leave a dataset the next run has to distrust. What the
        caller gets back is what will actually happen.
        """
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError(f"no such job: {job_id}")
        if job.status != "running":
            return {
                "job_id": job_id,
                "stopping": False,
                "status": job.status,
                "detail": f"that job already {job.status}",
            }
        job.token.cancel()
        return {
            "job_id": job_id,
            "stopping": True,
            "status": job.status,
            "detail": _STOP_DETAIL.get(job.kind, _STOP_DETAIL["default"]),
        }

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        # Newest first: a user watching a run wants the current one on top.
        return [job.to_dict() for job in sorted(jobs, key=lambda j: -j.started_at)]


# What stopping actually means, per kind of work. Vague reassurance would be
# worse than nothing here: the user is deciding whether to wait.
_STOP_DETAIL = {
    "project": (
        "Stopping after the chain or trial in flight. Simulations already "
        "written are kept and the next run resumes from them; no model is saved."
    ),
    "screening": "Stopping after the candidate being simulated.",
    "preview": "Stopping after the sample chain being simulated.",
    "analysis": "Stopping before the next stage; inference itself is seconds.",
    "default": "Stopping at the next safe point.",
}


def _accepts_token(work: Callable[..., Any]) -> bool:
    """Whether a job's work function wants the cancellation token."""
    import inspect

    try:
        return bool(inspect.signature(work).parameters)
    except (TypeError, ValueError):  # pragma: no cover - builtins and C callables
        return False


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
            "For non-uniform chains where the coupling of each bond is inferred "
            "separately. A pretrained model from the associated study is included."
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
            "For uniform chains with one exchange constant per interaction distance."
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
    # `homogeneous_xxz_j1j2j3_dmi` -- an XXZ chain carrying DMI with nothing to
    # break the symmetry -- is deliberately absent. The family still exists in
    # the library, because reproducing the measurement that D_z is unlearnable
    # there needs it, but offering it in a form whose whole purpose is to stop
    # people spending compute on impossible runs would be a contradiction:
    # every model trained on it scores about zero D_z skill by construction.
    # The impurity system below is the one to use.
    {
        "system_type": "homogeneous_xxz_j1j2j3_dmi_impurity",
        "title": "XXZ + J2 + J3 + DMI with impurities",
        "recovers": "J1_xy, J2, J3, Jz and a D_z magnitude",
        "when": (
            "For DMI inference when transverse anisotropy or a transverse field "
            "breaks the relevant symmetry. Check the geometry under DMI sample design."
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

# Hyperparameters, described so the form can render an editor for each one and
# refuse an impossible value before any compute is spent. Every default is the
# library's own, so leaving the whole block alone reproduces an untuned run
# exactly. `type` is one of number, integer, layers, boolean, choice.
_MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "ridge",
        "title": "Ridge regression",
        "notes": "Fast and deterministic. This model gave the best DMI validation result.",
        "needs_tensorflow": False,
        "options": [
            {
                "name": "alpha",
                "label": "regularisation strength",
                "default": 0.001,
                "type": "number",
                "min": 0.0,
                "hint": "larger is smoother and less able to fit fine structure",
            },
        ],
    },
    {
        "name": "random_forest",
        "title": "Random forest",
        "notes": "Handles non-linearity without tuning. Slower and larger on disk.",
        "needs_tensorflow": False,
        "options": [
            {
                "name": "n_estimators",
                "label": "number of trees",
                "default": 400,
                "type": "integer",
                "min": 1,
                "max": 5000,
                "hint": "more trees usually reduce variance but increase training time and file size",
            },
            {
                "name": "min_samples_leaf",
                "label": "minimum samples per leaf",
                "default": 2,
                "type": "integer",
                "min": 1,
                "max": 100,
            },
            {
                "name": "max_depth",
                "label": "maximum depth",
                "default": None,
                "type": "integer",
                "min": 1,
                "max": 200,
                "hint": "leave blank for unlimited, which is the default",
            },
            {
                "name": "n_jobs",
                "label": "cores to use",
                "default": -1,
                "type": "integer",
                "min": -1,
                "max": 1024,
                "hint": (
                    "Trees are grown independently, so this scales almost "
                    "linearly. -1 means every core on the machine, which is "
                    "the fastest and also the least polite thing to do on a "
                    "shared login node; set a number to leave some for "
                    "everyone else."
                ),
            },
        ],
    },
    {
        "name": "keras_mlp",
        "title": "Neural network (MLP)",
        "notes": (
            "Used by the published inhomogeneous model. Set the layers below "
            "or enable hyperparameter search. Requires the ml extra "
            "(pip install \"hamlet-toolkit[ml]\")."
        ),
        "needs_tensorflow": True,
        "options": [
            {
                "name": "hidden_units",
                "label": "hidden layers",
                "default": [512, 256, 128],
                "type": "layers",
                "hint": (
                    "How much the network can represent. One row per layer, "
                    "its width being the number of neurons. Wider and deeper "
                    "fits more complicated spectra but needs more training "
                    "chains to pin down, and overfits sooner when it does "
                    "not have them. Narrowing towards the output, as the "
                    "default does, is a safe starting shape."
                ),
            },
            {
                "name": "activation",
                "label": "activation",
                "default": "relu",
                "type": "choice",
                "choices": ["relu", "gelu", "tanh", "elu", "selu"],
                "hint": (
                    "The bend that lets layers stack into something other "
                    "than one big linear map. `relu` is the standard choice "
                    "and rarely the thing worth changing; `gelu` and `elu` "
                    "are smoother and occasionally help on small datasets."
                ),
            },
            {
                "name": "dropout",
                "label": "dropout",
                "default": 0.25,
                "type": "number",
                "min": 0.0,
                "max": 0.95,
                "hint": (
                    "Anti-memorisation. Each training step ignores this "
                    "fraction of neurons at random, so the network cannot "
                    "lean on any one of them. Raise it when the training "
                    "error is far better than the held-out error; lower it "
                    "towards 0 if the model never fits well in the first "
                    "place. Applied between hidden layers, not after the last."
                ),
            },
            {
                "name": "l2",
                "label": "weight decay (L2)",
                "default": 0.0003,
                "type": "number",
                "min": 0.0,
                "hint": (
                    "The other anti-memorisation knob: a running penalty on "
                    "large weights, which keeps the fitted function smooth. "
                    "Same symptom to watch as dropout. 0 turns it off."
                ),
            },
            {
                "name": "learning_rate",
                "label": "learning rate",
                "default": 0.0003,
                "type": "number",
                "min": 1e-8,
                "hint": (
                    "How big a step each update takes. Too high and the "
                    "training error jumps around or turns into NaN; too low "
                    "and it crawls and stops early before it has arrived. "
                    "Change it by factors of ten, not percentages."
                ),
            },
            {
                "name": "batch_normalization",
                "label": "batch normalisation",
                "default": True,
                "type": "boolean",
                "hint": (
                    "Rescales the values flowing between layers so they stay "
                    "in a comfortable range. Mostly it makes training faster "
                    "and less sensitive to the learning rate. Leave it on "
                    "unless you are chasing a specific problem."
                ),
            },
            {
                "name": "huber_delta",
                "label": "Huber delta",
                "default": 0.02,
                "type": "number",
                "min": 1e-8,
                "hint": (
                    "Where the loss stops punishing an error quadratically "
                    "and starts punishing it only linearly. Its effect is on "
                    "outliers: below this residual a chain is fitted "
                    "normally, above it the chain stops dominating the "
                    "gradient. Raise it to take unusual chains more "
                    "seriously, lower it to let the bulk of the data win. "
                    "Measured on the scaled target, not in meV."
                ),
            },
        ],
    },
    {
        "name": "keras_cnn",
        "title": "Neural network (CNN)",
        "notes": (
            "Applies convolutions along the bias axis to learn local spectral "
            "features. Requires the ml extra."
        ),
        "needs_tensorflow": True,
        "options": [
            {
                "name": "filters",
                "label": "convolution blocks",
                "default": [32, 64],
                "type": "layers",
                "hint": (
                    "How many spectral features each stage may look for, one "
                    "row per stage. Early blocks pick up narrow structure -- "
                    "a step, a peak edge -- and later ones combine those into "
                    "broader shapes. Each block halves the bias axis, so you "
                    "cannot have more blocks than log2(input points)."
                ),
            },
            {
                "name": "kernel_size",
                "label": "kernel width",
                "default": 7,
                "type": "integer",
                "min": 1,
                "max": 129,
                "hint": (
                    "How wide a window the network sees at once, in bias "
                    "points. Set it to roughly the width of the feature you "
                    "care about: too narrow and a broad step looks like "
                    "noise, too wide and sharp excitations get smeared "
                    "together. Compare it against the bias spacing of your "
                    "measurement rather than picking a number in the abstract."
                ),
            },
            {
                "name": "dense_units",
                "label": "dense layers after the convolutions",
                "default": [128],
                "type": "layers",
                "hint": (
                    "The part that turns the detected features into actual "
                    "coupling numbers. One row per layer. Usually one modest "
                    "layer is enough -- the convolutions have already done "
                    "the work of finding what matters."
                ),
            },
            {
                "name": "activation",
                "label": "activation",
                "default": "relu",
                "type": "choice",
                "choices": ["relu", "gelu", "tanh", "elu", "selu"],
                "hint": (
                    "The bend that lets layers stack into something other "
                    "than one big linear map. `relu` is the standard choice "
                    "and rarely the thing worth changing."
                ),
            },
            {"name": "dropout", "label": "dropout", "default": 0.2,
             "type": "number", "min": 0.0, "max": 0.95,
             "hint": (
                 "Anti-memorisation: each step ignores this fraction of "
                 "neurons at random. Raise it when the training error is far "
                 "better than the held-out error; lower it towards 0 if the "
                 "model never fits well at all."
             )},
            {"name": "l2", "label": "weight decay (L2)", "default": 0.0003,
             "type": "number", "min": 0.0,
             "hint": (
                 "A running penalty on large weights, which keeps the fitted "
                 "function smooth. Same symptom to watch as dropout; 0 turns "
                 "it off."
             )},
            {"name": "learning_rate", "label": "learning rate", "default": 0.0003,
             "type": "number", "min": 1e-8,
             "hint": (
                 "How big a step each update takes. Too high and the error "
                 "jumps around or goes NaN; too low and it crawls. Change it "
                 "by factors of ten."
             )},
            {"name": "batch_normalization", "label": "batch normalisation",
             "default": True, "type": "boolean",
             "hint": (
                 "Rescales values flowing between layers so they stay in a "
                 "comfortable range. Mostly makes training faster and less "
                 "sensitive to the learning rate. Leave it on."
             )},
            {"name": "huber_delta", "label": "Huber delta", "default": 0.02,
             "type": "number", "min": 1e-8,
             "hint": (
                 "Where the loss stops punishing an error quadratically and "
                 "starts punishing it linearly -- above this residual an odd "
                 "chain stops dominating the gradient. On the scaled target, "
                 "not in meV."
             )},
        ],
    },
)

# A width or depth beyond these is a typo rather than an architecture: the
# inputs here are a few hundred numbers per chain, and a run that only fails
# when Keras tries to allocate it wastes the whole generation stage first.
MAX_LAYER_WIDTH = 8192
MAX_LAYERS = 12

_PRESET_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "standard",
        "title": "Standard",
        "notes": "Three seeds and full training. Recommended for routine analysis.",
    },
    {
        "name": "research",
        "title": "Research",
        "notes": (
            "Five seeds and longer training. Intended for final model comparison "
            "and publication; costs roughly twice a standard run."
        ),
    },
    {
        "name": "quick",
        "title": "Quick",
        "notes": (
            "One seed and few epochs. Intended for workflow tests; artifacts are "
            "marked as development-only and excluded from normal inference."
        ),
    },
)


def _model_spec(name: str) -> dict[str, Any]:
    for spec in _MODEL_SPECS:
        if spec["name"] == name:
            return spec
    raise ValueError(f"unknown model {name!r}")


def _coerce_layers(label: str, value: Any) -> list[int]:
    """Read a layer stack from a list, or from text a person typed."""
    if isinstance(value, str):
        parts = [item for item in value.replace(";", ",").split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError(f"{label}: expected a list of widths, got {value!r}")
    widths: list[int] = []
    for part in parts:
        try:
            width = int(str(part).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"{label}: {part!r} is not a whole number of units"
            ) from None
        if width < 1:
            raise ValueError(f"{label}: every layer needs at least one unit, got {width}")
        if width > MAX_LAYER_WIDTH:
            raise ValueError(
                f"{label}: {width} units in one layer is beyond anything this "
                f"input size justifies; the cap is {MAX_LAYER_WIDTH}"
            )
        widths.append(width)
    if not widths:
        raise ValueError(f"{label}: at least one layer is required")
    if len(widths) > MAX_LAYERS:
        raise ValueError(
            f"{label}: {len(widths)} layers is beyond the {MAX_LAYERS} this form allows"
        )
    return widths


def coerce_model_options(model: str, values: Any) -> dict[str, Any]:
    """Type and range-check the hyperparameters the form collected.

    Blank means "the library default", which is what makes it safe to render
    every field: clearing one returns that single setting to the default rather
    than sending an empty value the model layer would have to interpret.
    """
    spec = _model_spec(model)
    by_name = {option["name"]: option for option in spec["options"]}
    coerced: dict[str, Any] = {}
    for name, value in dict(values or {}).items():
        option = by_name.get(name)
        if option is None:
            raise ValueError(
                f"{spec['title']} has no hyperparameter called {name!r}; "
                f"it takes {sorted(by_name) or 'none'}"
            )
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        label = option["label"]
        kind = option["type"]
        if kind == "layers":
            coerced[name] = _coerce_layers(label, value)
            continue
        if kind == "boolean":
            if isinstance(value, str):
                coerced[name] = value.strip().lower() in {"1", "true", "yes", "on"}
            else:
                coerced[name] = bool(value)
            continue
        if kind == "choice":
            text = str(value)
            if text not in option["choices"]:
                raise ValueError(
                    f"{label}: {text!r} is not one of {option['choices']}"
                )
            coerced[name] = text
            continue
        try:
            number = int(value) if kind == "integer" else float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label}: {value!r} is not a number") from None
        if option.get("min") is not None and number < option["min"]:
            raise ValueError(f"{label}: {number:g} is below the minimum {option['min']:g}")
        if option.get("max") is not None and number > option["max"]:
            raise ValueError(f"{label}: {number:g} is above the maximum {option['max']:g}")
        coerced[name] = number

    _validate_with_the_library(model, coerced)
    return coerced


def _validate_with_the_library(model: str, options: dict[str, Any]) -> None:
    """Let the model layer itself reject a combination, before any compute.

    The form's per-field limits cannot catch a rule that spans fields, and the
    library already states those rules once. Constructing the configuration is
    free; building the network is not.
    """
    if not model.startswith("keras_"):
        return
    from ..models.supervised import CNNConfig, MLPConfig

    config_type = MLPConfig if model == "keras_mlp" else CNNConfig
    try:
        config_type(**options)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"those hyperparameters are not usable: {exc}") from exc


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

    from ..training.tuning import SEARCH_SPACES, optuna_available

    return {
        "systems": [dict(spec) for spec in _SYSTEM_SPECS],
        "models": [dict(spec) for spec in _MODEL_SPECS],
        "presets": [dict(spec) for spec in _PRESET_SPECS],
        "tuning": {
            "backend": "optuna" if optuna_available() else "random search",
            "optuna_available": optuna_available(),
            "tunable_models": sorted(SEARCH_SPACES),
            "default_trials": 20,
            "max_trials": MAX_TUNING_TRIALS,
            "trial_preset": "quick",
            "searched": {
                name: [dim.label or dim.name.replace("_", " ") for dim in space.dimensions]
                for name, space in SEARCH_SPACES.items()
            },
            "notes": (
                "Each trial is one short training run at the quick preset, so a "
                "search costs about that many quick runs and happens after "
                "generation. The library defaults are evaluated first and kept "
                "if nothing beats them, and only the validation split is read, "
                "so the held-out score stays honest."
            ),
        },
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
        # Every spin the simulator knows, for both the chain and the
        # impurities. S=1/2 is in the impurity list too, because which spins
        # are valid depends on what the chain is: an S=1/2 impurity is a
        # perfectly good substitution in an S=1 chain. The rule that they must
        # differ is enforced where it is known, not by pruning a static list.
        "spins": ["S=1/2", "S=1", "S=3/2", "S=2", "S=5/2"],
        "chain_spins": ["S=1/2", "S=1", "S=3/2", "S=2", "S=5/2"],
        "tensorflow_available": tensorflow_available,
        "reference_seconds_per_correlator_l8": _reference_rate(),
    }


# A search is one short training run per trial. Past this many, the honest
# advice is to generate more chains instead.
MAX_TUNING_TRIALS = 200


def _tuning_payload(form: dict[str, Any], model: str) -> dict[str, Any] | None:
    """Read the search settings out of the form, or None if it was not asked for."""
    from ..training.tuning import SEARCH_SPACES

    settings = form.get("tuning") or {}
    if not isinstance(settings, dict):
        raise ValueError("tuning must be a mapping")
    if not settings.get("enabled"):
        return None
    if model not in SEARCH_SPACES:
        raise ValueError(
            f"there is no search space for {model!r}; tunable models are "
            f"{sorted(SEARCH_SPACES)}"
        )
    try:
        n_trials = int(settings.get("n_trials", 20))
    except (TypeError, ValueError):
        raise ValueError("the number of search trials must be a whole number") from None
    if not 1 <= n_trials <= MAX_TUNING_TRIALS:
        raise ValueError(
            f"the number of search trials must be between 1 and "
            f"{MAX_TUNING_TRIALS}; {n_trials} was requested"
        )
    payload: dict[str, Any] = {
        "n_trials": n_trials,
        "preset": str(settings.get("preset", "quick")),
        "seed": int(settings.get("seed", 0)),
    }
    timeout = settings.get("timeout_minutes")
    if timeout not in (None, ""):
        minutes = float(timeout)
        if minutes <= 0:
            raise ValueError("the search time limit must be positive")
        payload["timeout_seconds"] = minutes * 60.0
    return payload


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
    project_dir = root / (_slug(name) or "project")
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

    # The chain's own spin. Validated here as well as in the library, because
    # a rejection after the generation stage has started costs hours.
    site_spin = str(form.get("site_spin", "S=1/2") or "S=1/2")
    from ..systems.heisenberg import validate_site_spin

    validate_site_spin(site_spin)
    if site_spin != "S=1/2":
        generate["site_spin"] = site_spin
    clashing = sorted(
        int(item.get("site"))
        for item in (form.get("impurities") or [])
        if str(item.get("spin", "S=1")) == site_spin
    )
    if clashing:
        raise ValueError(
            f"the impurity at site(s) {clashing} has the same spin as the chain "
            f"({site_spin}); a substituted site has to differ from the chain it "
            f"sits in"
        )

    device = str(form.get("device", "auto"))
    if device not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"device must be auto, cpu or gpu; got {device!r}")
    if device == "gpu" and sys.platform != "linux":
        # Refused rather than silently downgraded, because the request cannot
        # be met here and never will be: TensorFlow has no GPU build for this
        # platform. `auto` is what a run bound for a GPU cluster wants anyway.
        raise ValueError(
            "TensorFlow can only use a GPU on Linux, so 'Require a GPU' is not "
            "available on this platform. Choose Automatic -- it uses a GPU "
            "whenever one is present, including on a cluster node."
        )

    # Cores for generation. Refused at the form rather than at generation
    # time, which on this stage would mean failing after the first chunk of an
    # hours-long run.
    workers = int(form.get("workers", 1) or 1)
    if workers < 0:
        raise ValueError("chains at once must not be negative; 0 means one per core")
    if workers != 1:
        generate["workers"] = workers

    payload: dict[str, Any] = {
        "config_schema_version": 1,
        "name": name,
        "system_type": spec["system_type"],
        "output_dir": None,  # filled in below, from the settings themselves
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
    if device != "auto":
        payload["training"]["device"] = device
    options = coerce_model_options(str(form["model"]), form.get("model_options"))
    if options:
        payload["training"]["model_options"] = options
    tuning = _tuning_payload(form, str(form["model"]))
    if tuning is not None:
        payload["training"]["tuning"] = tuning

    # One directory per distinct set of settings, named by their fingerprint.
    #
    # A run refuses to write into a directory that holds a different resolved
    # configuration, which is the right rule -- it is what stops two runs
    # quietly sharing an artifact. But the interface used to send every run of
    # a given name to the same `run/`, so the ordinary act of stopping a job,
    # changing one number and starting it again hit that refusal and asked the
    # user to invent a new name.
    #
    # Keying the directory on the settings makes both cases behave the way
    # they read: a tweak is a different run and gets a clean directory, while
    # starting the same settings again returns to the directory that already
    # has the checkpoints, and generation resumes instead of starting over.
    fingerprint = _settings_fingerprint(payload)
    run_dir = project_dir / f"run-{fingerprint}"
    # Existing, but empty, is not resuming: the directory is created by
    # building the configuration a moment before anything runs, so its mere
    # presence would report "continuing" on a run that has produced nothing.
    resuming = run_dir.is_dir() and any(run_dir.iterdir())
    payload["output_dir"] = str(run_dir)

    config_path = project_dir / f"project-{fingerprint}.yaml"
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

    return {
        "config_path": str(config_path),
        "project_dir": str(project_dir),
        "run_dir": str(run_dir),
        "name": name,
        "resuming": resuming,
        "fingerprint": fingerprint,
    }


def _settings_fingerprint(payload: Mapping[str, Any]) -> str:
    """A short stable name for one set of settings.

    ``output_dir`` is excluded because it is what this computes. Sorted keys
    so that a form filled in a different order is still the same run, and
    eight hex characters because this names a directory a person will read --
    a collision would need two different configurations out of four billion
    under the same project name.
    """
    material = {key: value for key, value in payload.items() if key != "output_dir"}
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:8]


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
    should_stop: Callable[[], bool] | None = None,
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
            check_cancelled(should_stop, "the preview")
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
