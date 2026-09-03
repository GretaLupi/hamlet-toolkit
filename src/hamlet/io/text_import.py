"""Recipe-driven import of one spectroscopy text file per measured site."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fnmatch import fnmatch
import html
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

import numpy as np
import yaml

from ..branding import brand_manifest
from ..measurements import Measurement


@dataclass(frozen=True)
class SignalColumn:
    column: str | int
    unit: str = "arbitrary"


@dataclass(frozen=True)
class FileImportDiagnostic:
    source: str
    site: str | int
    rows: int
    columns: int
    energy_min_mev: float
    energy_max_mev: float
    reversed_energy: bool
    missing_by_channel: Mapping[str, int]
    extra_columns: tuple[str, ...]


@dataclass(frozen=True)
class MeasurementImportResult:
    measurement: Measurement
    diagnostics: tuple[FileImportDiagnostic, ...]
    warnings: tuple[str, ...]
    csv_path: Path | None = None
    measurement_path: Path | None = None
    report_path: Path | None = None
    preview_path: Path | None = None


@dataclass(frozen=True)
class TextImportRecipe:
    """Configuration for a sectioned or ordinary delimited text export."""

    source_kind: str
    source_path: Path
    pattern: str
    delimiter: str
    data_marker: str | None
    header_offset: int
    skip_rows: int
    encoding: str
    energy_column: str | int
    energy_unit: str
    signals: Mapping[str, SignalColumn]
    primary_channel: str
    site_mode: str
    site_start: int
    energy_order: str
    grid_policy: str
    grid_rtol: float
    grid_atol_mev: float
    missing_policy: str
    metadata_keys: tuple[str, ...]
    output_csv: Path
    output_measurement: Path
    output_report: Path
    output_preview: Path | None
    include_auxiliary_csv: bool
    overwrite: bool

    @classmethod
    def from_file(cls, path: str | Path) -> "TextImportRecipe":
        recipe_path = Path(path).resolve()
        payload = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("import recipe must be a YAML/JSON mapping")
        return cls.from_mapping(payload, base=recipe_path.parent)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, base: Path) -> "TextImportRecipe":
        source = _mapping(payload.get("input"), "input")
        source_fields = [name for name in ("archive", "directory", "files") if name in source]
        if len(source_fields) != 1:
            raise ValueError("input must define exactly one of archive, directory, or files")
        source_kind = source_fields[0]
        source_value = source[source_kind]
        if source_kind == "files":
            if not isinstance(source_value, Sequence) or isinstance(source_value, str):
                raise ValueError("input.files must be a list")
            source_path = base
        else:
            source_path = _resolve(base, source_value)

        format_config = _mapping(payload.get("format", {}), "format")
        columns = _mapping(payload.get("columns"), "columns")
        energy_config = columns.get("energy")
        if isinstance(energy_config, Mapping):
            energy_column = energy_config.get("column")
            energy_unit = str(energy_config.get("unit", "meV"))
        else:
            energy_column = energy_config
            energy_unit = str(payload.get("energy_unit", "meV"))
        if not isinstance(energy_column, (str, int)):
            raise ValueError("columns.energy must select a column by exact name or index")

        signal_payload = columns.get("signals")
        if not isinstance(signal_payload, Mapping) or not signal_payload:
            raise ValueError("columns.signals must define at least the didv channel")
        signals: dict[str, SignalColumn] = {}
        for name, specification in signal_payload.items():
            if isinstance(specification, Mapping):
                column = specification.get("column")
                unit = str(specification.get("unit", "arbitrary"))
            else:
                column = specification
                unit = "arbitrary"
            if not isinstance(column, (str, int)):
                raise ValueError(f"signal {name!r} requires a column name or index")
            signals[str(name)] = SignalColumn(column, unit)

        processing = _mapping(payload.get("processing", {}), "processing")
        site = _mapping(payload.get("site", {}), "site")
        output = _mapping(payload.get("output", {}), "output")
        output_dir = _resolve(base, output.get("directory", "imported_measurement"))
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_value = output.get("preview", "import_preview.html")
        if preview_value in (None, False):
            output_preview = None
        else:
            output_preview = _output_path(output_dir, preview_value)

        delimiter = str(format_config.get("delimiter", "\t"))
        delimiter = {"\\t": "\t", "tab": "\t", "comma": ","}.get(delimiter, delimiter)
        if len(delimiter) != 1:
            raise ValueError("format.delimiter must be one character")
        primary = str(columns.get("primary", "didv"))
        if primary not in signals:
            raise ValueError(f"primary channel {primary!r} is not in columns.signals")
        signals = {primary: signals[primary], **{name: item for name, item in signals.items() if name != primary}}

        recipe = cls(
            source_kind=source_kind,
            source_path=source_path,
            pattern=str(source.get("pattern", "*.txt")),
            delimiter=delimiter,
            data_marker=(
                None if format_config.get("data_marker") is None
                else str(format_config.get("data_marker"))
            ),
            header_offset=int(format_config.get("header_offset", 1)),
            skip_rows=int(format_config.get("skip_rows", 0)),
            encoding=str(format_config.get("encoding", "utf-8")),
            energy_column=energy_column,
            energy_unit=energy_unit,
            signals=signals,
            primary_channel=primary,
            site_mode=str(site.get("mode", "sequential")),
            site_start=int(site.get("start", 1)),
            energy_order=str(processing.get("energy_order", "ascending")),
            grid_policy=str(processing.get("grid", "require_equal")),
            grid_rtol=float(processing.get("grid_rtol", 1e-7)),
            grid_atol_mev=float(processing.get("grid_atol_mev", 1e-9)),
            missing_policy=str(processing.get("missing", "error")),
            metadata_keys=tuple(str(item) for item in format_config.get("metadata_keys", [])),
            output_csv=_output_path(output_dir, output.get("csv", "spectroscopy.csv")),
            output_measurement=_output_path(
                output_dir, output.get("measurement", "measurement.npz")
            ),
            output_report=_output_path(output_dir, output.get("report", "import_report.json")),
            output_preview=output_preview,
            include_auxiliary_csv=bool(output.get("include_auxiliary_csv", True)),
            overwrite=bool(output.get("overwrite", False)),
        )
        recipe._validate()
        object.__setattr__(recipe, "_file_values", tuple(source_value) if source_kind == "files" else ())
        object.__setattr__(recipe, "_base", base)
        return recipe

    def _validate(self) -> None:
        if self.site_mode not in {"sequential", "filename_number", "filename"}:
            raise ValueError("site.mode must be sequential, filename_number, or filename")
        if self.energy_order not in {"ascending", "descending", "preserve"}:
            raise ValueError("processing.energy_order must be ascending, descending, or preserve")
        if self.grid_policy not in {"require_equal", "interpolate"}:
            raise ValueError("processing.grid must be require_equal or interpolate")
        if self.missing_policy not in {"error", "drop_common", "keep"}:
            raise ValueError("processing.missing must be error, drop_common, or keep")
        _energy_factor_to_mev(self.energy_unit)

    def source_texts(self) -> list[tuple[str, str]]:
        if self.source_kind == "archive":
            if not self.source_path.exists():
                raise FileNotFoundError(self.source_path)
            with ZipFile(self.source_path) as archive:
                names = [
                    name for name in archive.namelist()
                    if not name.endswith("/") and fnmatch(PurePosixPath(name).name, self.pattern)
                ]
                names.sort(key=_natural_key)
                return [
                    (name, archive.read(name).decode(self.encoding, errors="replace"))
                    for name in names
                ]
        if self.source_kind == "directory":
            paths = sorted(self.source_path.glob(self.pattern), key=lambda item: _natural_key(item.name))
        else:
            paths = [_resolve(self._base, item) for item in self._file_values]
        return [(str(path), path.read_text(encoding=self.encoding, errors="replace")) for path in paths]


def import_text_measurement(recipe: TextImportRecipe | str | Path) -> MeasurementImportResult:
    """Import, validate, export, and report a multi-file measurement."""
    if not isinstance(recipe, TextImportRecipe):
        recipe = TextImportRecipe.from_file(recipe)
    destinations = [recipe.output_csv, recipe.output_measurement, recipe.output_report]
    if recipe.output_preview is not None:
        destinations.append(recipe.output_preview)
    existing = [path for path in destinations if path.exists()]
    if existing and not recipe.overwrite:
        raise FileExistsError(
            "import outputs already exist; choose a new output.directory or set "
            f"output.overwrite: true explicitly: {[str(path) for path in existing]}"
        )
    sources = recipe.source_texts()
    if not sources:
        raise ValueError(f"no input files matched {recipe.pattern!r}")

    parsed: list[dict[str, Any]] = []
    all_headers: set[str] = set()
    for index, (source, text) in enumerate(sources):
        table = _parse_table(text, recipe)
        headers = table["headers"]
        all_headers.update(headers)
        energy_index = _column_index(recipe.energy_column, headers)
        energy = table["values"][:, energy_index] * _energy_factor_to_mev(recipe.energy_unit)
        channels = {
            name: table["values"][:, _column_index(spec.column, headers)]
            for name, spec in recipe.signals.items()
        }
        site = _site_id(source, index, recipe)
        reversed_energy = False
        direction = np.sign(np.nanmedian(np.diff(energy)))
        wants_reverse = (
            recipe.energy_order == "ascending" and direction < 0
        ) or (recipe.energy_order == "descending" and direction > 0)
        if wants_reverse:
            energy = energy[::-1]
            channels = {name: values[::-1] for name, values in channels.items()}
            reversed_energy = True
        if np.any(~np.isfinite(energy)):
            raise ValueError(f"{source}: energy column contains non-finite values")
        if np.any(np.diff(energy) == 0):
            raise ValueError(f"{source}: energy grid contains duplicates")
        selected_metadata = (
            {key: table["metadata"].get(key) for key in recipe.metadata_keys}
            if recipe.metadata_keys else {}
        )
        parsed.append(
            {
                "source": source,
                "site": site,
                "energy": energy,
                "channels": channels,
                "headers": headers,
                "rows": len(energy),
                "reversed": reversed_energy,
                "metadata": selected_metadata,
            }
        )

    reference = parsed[0]["energy"]
    grid_warning = False
    for item in parsed[1:]:
        matches = item["energy"].shape == reference.shape and np.allclose(
            item["energy"], reference, rtol=recipe.grid_rtol, atol=recipe.grid_atol_mev
        )
        if matches:
            continue
        if recipe.grid_policy == "require_equal":
            raise ValueError(
                f"energy grid in {item['source']} differs from {parsed[0]['source']}; "
                "set processing.grid: interpolate only if interpolation is scientifically acceptable"
            )
        grid_warning = True
        for name, values in item["channels"].items():
            valid = np.isfinite(values)
            if np.count_nonzero(valid) < 2:
                item["channels"][name] = np.full(reference.shape, np.nan)
            else:
                x = item["energy"][valid]
                y = values[valid]
                if x[0] > x[-1]:
                    x, y = x[::-1], y[::-1]
                item["channels"][name] = np.interp(reference, x, y, left=np.nan, right=np.nan)
        item["energy"] = reference

    stacked = {
        name: np.stack([item["channels"][name] for item in parsed])
        for name in recipe.signals
    }
    missing_before = {name: int(np.size(values) - np.isfinite(values).sum()) for name, values in stacked.items()}
    primary_valid = np.all(np.isfinite(stacked[recipe.primary_channel]), axis=0)
    if not np.all(primary_valid):
        if recipe.missing_policy == "error":
            raise ValueError(
                f"primary channel has {int(np.size(stacked[recipe.primary_channel]) - np.isfinite(stacked[recipe.primary_channel]).sum())} "
                "missing values; set processing.missing to drop_common or keep"
            )
        if recipe.missing_policy == "drop_common":
            reference = reference[primary_valid]
            stacked = {name: values[:, primary_valid] for name, values in stacked.items()}

    diagnostics: list[FileImportDiagnostic] = []
    selected_headers = {str(recipe.energy_column)} | {
        str(spec.column) for spec in recipe.signals.values()
    }
    for item in parsed:
        diagnostics.append(
            FileImportDiagnostic(
                source=item["source"],
                site=item["site"],
                rows=item["rows"],
                columns=len(item["headers"]),
                energy_min_mev=float(np.min(item["energy"])),
                energy_max_mev=float(np.max(item["energy"])),
                reversed_energy=item["reversed"],
                missing_by_channel={
                    name: int(values.size - np.isfinite(values).sum())
                    for name, values in item["channels"].items()
                },
                extra_columns=tuple(
                    header for header in item["headers"] if header not in selected_headers
                ),
            )
        )

    warnings: list[str] = []
    if grid_warning:
        warnings.append("one or more energy grids were interpolated onto the first file's grid")
    if any(missing_before.values()):
        warnings.append(f"missing values before policy application: {missing_before}")
    column_counts = {item.columns for item in diagnostics}
    if len(column_counts) > 1:
        warnings.append(f"input files have different column counts: {sorted(column_counts)}")

    measurement = Measurement(
        axes={"site": np.asarray([item["site"] for item in parsed]), "bias": reference},
        channels=stacked,
        axis_units={"site": "index", "bias": "meV"},
        channel_units={name: spec.unit for name, spec in recipe.signals.items()},
        primary_channel=recipe.primary_channel,
        metadata={
            "importer": "multi_file_text_v1",
            "source_kind": recipe.source_kind,
            "source_path": str(recipe.source_path),
            "source_files": [item["source"] for item in parsed],
            "file_metadata": [item["metadata"] for item in parsed],
            "energy_input_unit": recipe.energy_unit,
            "missing_policy": recipe.missing_policy,
            "grid_policy": recipe.grid_policy,
        },
    )
    measurement.save(recipe.output_measurement)
    measurement.to_spectroscopy_csv(
        recipe.output_csv,
        include_auxiliary=recipe.include_auxiliary_csv,
        allow_missing=recipe.missing_policy == "keep",
    )
    report = {
        "toolkit": brand_manifest(),
        "status": (
            "structurally_ready" if measurement.is_primary_complete else "review_missing_data"
        ),
        "shape": measurement.shape,
        "axes": list(measurement.axes),
        "channels": list(measurement.channels),
        "primary_channel": measurement.primary_channel,
        "warnings": warnings,
        "outputs": {
            "csv": str(recipe.output_csv),
            "measurement": str(recipe.output_measurement),
        },
        "files": [asdict(item) for item in diagnostics],
    }
    recipe.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if recipe.output_preview is not None:
        _write_preview(recipe.output_preview, measurement, report)
    return MeasurementImportResult(
        measurement=measurement,
        diagnostics=tuple(diagnostics),
        warnings=tuple(warnings),
        csv_path=recipe.output_csv,
        measurement_path=recipe.output_measurement,
        report_path=recipe.output_report,
        preview_path=recipe.output_preview,
    )


def _parse_table(text: str, recipe: TextImportRecipe) -> dict[str, Any]:
    lines = text.splitlines()
    metadata_end = recipe.skip_rows
    if recipe.data_marker is not None:
        matches = [index for index, line in enumerate(lines) if line.strip() == recipe.data_marker]
        if not matches:
            raise ValueError(f"data marker {recipe.data_marker!r} was not found")
        marker = matches[0]
        header_index = marker + recipe.header_offset
        metadata_end = marker
    else:
        header_index = recipe.skip_rows
    if header_index >= len(lines):
        raise ValueError("header row is beyond the end of the file")
    headers = tuple(item.strip() for item in lines[header_index].split(recipe.delimiter))
    rows: list[list[float]] = []
    for line_number, line in enumerate(lines[header_index + 1 :], start=header_index + 2):
        if not line.strip():
            continue
        fields = line.split(recipe.delimiter)
        if len(fields) != len(headers):
            raise ValueError(
                f"line {line_number} has {len(fields)} fields; expected {len(headers)}"
            )
        try:
            rows.append([float(field.strip()) if field.strip() else np.nan for field in fields])
        except ValueError as exc:
            raise ValueError(f"line {line_number} contains non-numeric table data") from exc
    if not rows:
        raise ValueError("data table contains no rows")
    metadata: dict[str, str] = {}
    for line in lines[:metadata_end]:
        if recipe.delimiter in line:
            key, value = line.split(recipe.delimiter, 1)
            metadata[key.strip()] = value.strip()
    return {"headers": headers, "values": np.asarray(rows), "metadata": metadata}


def _column_index(column: str | int, headers: Sequence[str]) -> int:
    if isinstance(column, int):
        if not -len(headers) <= column < len(headers):
            raise ValueError(f"column index {column} is outside a {len(headers)}-column table")
        return column % len(headers)
    try:
        return headers.index(column)
    except ValueError as exc:
        raise ValueError(f"column {column!r} not found; available columns: {list(headers)}") from exc


def _site_id(source: str, index: int, recipe: TextImportRecipe) -> str | int:
    name = PurePosixPath(source).name
    if recipe.site_mode == "sequential":
        return recipe.site_start + index
    if recipe.site_mode == "filename":
        return Path(name).stem
    numbers = re.findall(r"\d+", Path(name).stem)
    if not numbers:
        raise ValueError(f"cannot extract a site number from {name!r}")
    return int(numbers[-1])


def _energy_factor_to_mev(unit: str) -> float:
    normalized = unit.strip().lower()
    factors = {"ev": 1000.0, "v": 1000.0, "mev": 1.0, "mv": 1.0, "uev": 1e-3, "uv": 1e-3}
    if normalized not in factors:
        raise ValueError(f"unsupported energy/bias unit {unit!r}; use V, mV, eV, meV, uV, or ueV")
    return factors[normalized]


def _natural_key(value: str) -> list[Any]:
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", value)]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _output_path(directory: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (directory / path).resolve()


def _write_preview(path: Path, measurement: Measurement, report: Mapping[str, Any]) -> None:
    rows = []
    for item in report["files"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['site']))}</td>"
            f"<td>{html.escape(PurePosixPath(item['source']).name)}</td>"
            f"<td>{item['rows']}</td><td>{item['columns']}</td>"
            f"<td>{html.escape(str(item['missing_by_channel']))}</td>"
            f"<td>{'yes' if item['reversed_energy'] else 'no'}</td>"
            "</tr>"
        )
    channel_cards = "".join(
        f"<li><code>{html.escape(name)}</code> [{html.escape(measurement.channel_units.get(name, ''))}]"
        f" — {'complete' if np.all(measurement.masks[name]) else 'contains missing values'}</li>"
        for name in measurement.channels
    )
    warning_items = "".join(f"<li>{html.escape(item)}</li>" for item in report["warnings"])
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Measurement import preview</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:1100px}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccd;padding:.45rem;text-align:left}}code{{background:#eef;padding:.1rem .25rem}}
.ready{{color:#176b32;font-weight:700}}.review{{color:#9a5300;font-weight:700}}</style></head>
<body><h1>Measurement import preview</h1>
<p class="{'ready' if report['status'] == 'structurally_ready' else 'review'}">{html.escape(report['status'])}</p>
<p>Shape: {measurement.shape}; primary inference channel: <code>{html.escape(measurement.primary_channel)}</code>.</p>
<h2>Channels</h2><ul>{channel_cards}</ul><h2>Warnings</h2><ul>{warning_items or '<li>None</li>'}</ul>
<h2>Files</h2><table><thead><tr><th>site</th><th>file</th><th>rows</th><th>columns</th><th>missing</th><th>axis reversed</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
