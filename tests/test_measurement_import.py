from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import yaml

from hamlet.io import (
    TextImportRecipe,
    import_text_measurement,
    load_measurement_csv,
)
from hamlet.measurements import Measurement
from hamlet.project_cli import main as project_cli_main


def _instrument_file(offset: float, *, missing_tail: bool = False, filtered: bool = False) -> str:
    headers = ["Bias (V)", "LI Demod 1 X (A)", "LI Demod 2 X (A)"]
    if filtered:
        headers.append("LI Demod 1 X (A) [filt]")
    rows = []
    for index, bias in enumerate([0.002, 0.001, 0.0, -0.001]):
        didv = "nan" if missing_tail and index == 3 else str(offset + index)
        values = [str(bias), didv, str(10 + offset + index)]
        if filtered:
            values.append(str(100 + offset + index))
        rows.append("\t".join(values))
    return "\n".join(
        ["X (m)\t1e-9", "Saved Date\t01.01.2026", "[DATA]", "\t".join(headers), *rows]
    )


def _recipe(tmp_path, archive):
    payload = {
        "input": {"archive": str(archive), "pattern": "*.dat"},
        "format": {
            "delimiter": "\\t",
            "data_marker": "[DATA]",
            "header_offset": 1,
            "metadata_keys": ["X (m)"],
        },
        "columns": {
            "energy": {"column": "Bias (V)", "unit": "V"},
            "primary": "didv",
            "signals": {
                "didv": {"column": "LI Demod 1 X (A)", "unit": "A"},
                "d2idv2": {"column": "LI Demod 2 X (A)", "unit": "A"},
            },
        },
        "site": {"mode": "sequential", "start": 1},
        "processing": {
            "energy_order": "ascending",
            "grid": "require_equal",
            "missing": "drop_common",
        },
        "output": {"directory": str(tmp_path / "output")},
    }
    path = tmp_path / "import.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_multi_file_import_maps_primary_and_auxiliary_channels(tmp_path):
    archive = tmp_path / "example.zip"
    with ZipFile(archive, "w") as handle:
        # Deliberately inserted out of order; natural filename order defines the sites.
        handle.writestr("bias_Spec00011.dat", _instrument_file(11, filtered=True))
        handle.writestr("bias_Spec00010.dat", _instrument_file(10, missing_tail=True))

    result = import_text_measurement(_recipe(tmp_path, archive))
    measurement = result.measurement
    assert measurement.axis_order == ("site", "bias")
    assert measurement.primary_channel == "didv"
    assert tuple(measurement.channels) == ("didv", "d2idv2")
    assert measurement.shape == (2, 3)  # common missing tail was explicitly removed
    np.testing.assert_allclose(measurement.axes["bias"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(measurement.channels["didv"][0], [12, 11, 10])
    np.testing.assert_allclose(measurement.channels["d2idv2"][0], [22, 21, 20])
    assert measurement.metadata["file_metadata"][0]["X (m)"] == "1e-9"
    assert any("different column counts" in warning for warning in result.warnings)

    frame = pd.read_csv(result.csv_path)
    assert list(frame.columns) == ["site", "bias_meV", "didv_A", "d2idv2_A"]
    loaded = load_measurement_csv(result.csv_path)
    np.testing.assert_allclose(loaded.channels["didv"], measurement.channels["didv"])
    np.testing.assert_allclose(loaded.channels["d2idv2"], measurement.channels["d2idv2"])
    assert "structurally_ready" in result.report_path.read_text()
    assert result.preview_path.exists()


def test_canonical_measurement_round_trip_and_missing_guard(tmp_path):
    measurement = Measurement(
        axes={"site": ["left", "right"], "bias": [0.0, 1.0]},
        channels={"didv": [[1.0, np.nan], [2.0, 3.0]]},
        axis_units={"bias": "meV"},
        channel_units={"didv": "A"},
        metadata={"operator": "test"},
    )
    assert not measurement.is_primary_complete
    try:
        measurement.site_spectra()
    except ValueError as exc:
        assert "missing values" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("incomplete measurements must not silently enter inference")

    path = measurement.save(tmp_path / "measurement.npz")
    loaded = Measurement.load(path)
    assert loaded.axis_order == ("site", "bias")
    assert loaded.metadata["operator"] == "test"
    np.testing.assert_array_equal(loaded.masks["didv"], measurement.masks["didv"])


def test_recipe_can_use_plain_text_directory(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "site2.txt").write_text(_instrument_file(2), encoding="utf-8")
    (source / "site1.txt").write_text(_instrument_file(1), encoding="utf-8")
    payload = yaml.safe_load(_recipe(tmp_path, tmp_path / "unused.zip").read_text())
    payload["input"] = {"directory": str(source), "pattern": "*.txt"}
    recipe = TextImportRecipe.from_mapping(payload, base=tmp_path)
    result = import_text_measurement(recipe)
    np.testing.assert_array_equal(result.measurement.axes["site"], [1, 2])
    np.testing.assert_allclose(result.measurement.channels["didv"][:, -1], [1, 2])


def _simple_recipe(tmp_path, source, *, unit="mV", processing=None, site=None, output=None):
    payload = {
        "input": {"directory": str(source), "pattern": "*.txt"},
        "format": {"delimiter": ",", "skip_rows": 0},
        "columns": {
            "energy": {"column": "energy", "unit": unit},
            "primary": "didv",
            "signals": {"didv": {"column": "signal", "unit": "A"}},
        },
        "site": site or {"mode": "sequential", "start": 1},
        "processing": processing
        or {"energy_order": "ascending", "grid": "require_equal", "missing": "error"},
        "output": {"directory": str(tmp_path / "simple-output"), **(output or {})},
    }
    return TextImportRecipe.from_mapping(payload, base=tmp_path)


@pytest.mark.parametrize(
    ("unit", "source_values", "expected_mev"),
    [
        ("V", [-0.001, 0.0, 0.001], [-1.0, 0.0, 1.0]),
        ("mV", [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]),
        ("eV", [-0.001, 0.0, 0.001], [-1.0, 0.0, 1.0]),
        ("meV", [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]),
        ("uV", [-1000.0, 0.0, 1000.0], [-1.0, 0.0, 1.0]),
    ],
)
def test_plain_comma_export_and_energy_unit_conversion(
    tmp_path, unit, source_values, expected_mev
):
    source = tmp_path / f"raw files {unit}"
    source.mkdir()
    lines = ["energy,signal"] + [f"{energy},{index + 1}" for index, energy in enumerate(source_values)]
    (source / "site 1.txt").write_text("\n".join(lines), encoding="utf-8")
    recipe = _simple_recipe(
        tmp_path,
        source,
        unit=unit,
        output={"directory": str(tmp_path / f"output {unit}")},
    )
    result = import_text_measurement(recipe)
    np.testing.assert_allclose(result.measurement.axes["bias"], expected_mev)
    np.testing.assert_allclose(result.measurement.channels["didv"], [[1, 2, 3]])


def test_site_identifier_modes_are_explicit(tmp_path):
    source = tmp_path / "named"
    source.mkdir()
    for name in ("chain_site10.txt", "chain_site2.txt"):
        (source / name).write_text("energy,signal\n0,1\n1,2", encoding="utf-8")

    numeric = import_text_measurement(
        _simple_recipe(
            tmp_path,
            source,
            site={"mode": "filename_number"},
            output={"directory": str(tmp_path / "numeric")},
        )
    )
    np.testing.assert_array_equal(numeric.measurement.axes["site"], [2, 10])

    filenames = import_text_measurement(
        _simple_recipe(
            tmp_path,
            source,
            site={"mode": "filename"},
            output={"directory": str(tmp_path / "filenames")},
        )
    )
    np.testing.assert_array_equal(
        filenames.measurement.axes["site"], ["chain_site2", "chain_site10"]
    )


def test_missing_data_policies_error_keep_and_drop_common(tmp_path):
    source = tmp_path / "missing"
    source.mkdir()
    (source / "site1.txt").write_text(
        "energy,signal\n0,1\n1,nan\n2,3", encoding="utf-8"
    )
    (source / "site2.txt").write_text(
        "energy,signal\n0,4\n1,5\n2,6", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing values"):
        import_text_measurement(
            _simple_recipe(tmp_path, source, output={"directory": str(tmp_path / "strict")})
        )

    keep = import_text_measurement(
        _simple_recipe(
            tmp_path,
            source,
            processing={"energy_order": "ascending", "grid": "require_equal", "missing": "keep"},
            output={"directory": str(tmp_path / "keep")},
        )
    )
    assert not keep.measurement.is_primary_complete
    assert "review_missing_data" in keep.report_path.read_text()
    with pytest.raises(ValueError, match="missing values"):
        keep.measurement.site_spectra()

    dropped = import_text_measurement(
        _simple_recipe(
            tmp_path,
            source,
            processing={
                "energy_order": "ascending",
                "grid": "require_equal",
                "missing": "drop_common",
            },
            output={"directory": str(tmp_path / "dropped")},
        )
    )
    np.testing.assert_allclose(dropped.measurement.axes["bias"], [0, 2])
    assert dropped.measurement.is_primary_complete


def test_grid_mismatch_is_rejected_or_explicitly_interpolated(tmp_path):
    source = tmp_path / "grids"
    source.mkdir()
    (source / "site1.txt").write_text(
        "energy,signal\n0,0\n1,1\n2,2", encoding="utf-8"
    )
    (source / "site2.txt").write_text(
        "energy,signal\n0,0\n0.5,1\n1.5,3\n2,4", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="energy grid"):
        import_text_measurement(
            _simple_recipe(tmp_path, source, output={"directory": str(tmp_path / "reject")})
        )

    interpolated = import_text_measurement(
        _simple_recipe(
            tmp_path,
            source,
            processing={"energy_order": "ascending", "grid": "interpolate", "missing": "error"},
            output={"directory": str(tmp_path / "interpolated")},
        )
    )
    np.testing.assert_allclose(interpolated.measurement.channels["didv"][1], [0, 2, 4])
    assert any("interpolated" in warning for warning in interpolated.warnings)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("energy,other\n0,1\n1,2", "column 'signal' not found"),
        ("energy,signal\n0,1\n0,2", "duplicates"),
        ("energy,signal\n0,hello\n1,2", "non-numeric"),
        ("energy,signal", "contains no rows"),
        ("energy,signal\n0,1,extra", "fields; expected"),
    ],
)
def test_corrupt_or_ambiguous_exports_fail_with_actionable_errors(tmp_path, contents, message):
    source = tmp_path / "bad"
    source.mkdir()
    (source / "site1.txt").write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        import_text_measurement(_simple_recipe(tmp_path, source))


def test_import_outputs_are_protected_unless_overwrite_is_explicit(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "site1.txt").write_text("energy,signal\n0,1\n1,2", encoding="utf-8")
    recipe = _simple_recipe(tmp_path, source)
    first = import_text_measurement(recipe)
    before = first.csv_path.read_bytes()
    with pytest.raises(FileExistsError, match="output.overwrite"):
        import_text_measurement(recipe)
    assert first.csv_path.read_bytes() == before

    overwrite_payload = {
        "input": {"directory": str(source), "pattern": "*.txt"},
        "format": {"delimiter": ","},
        "columns": {
            "energy": {"column": "energy", "unit": "mV"},
            "signals": {"didv": {"column": "signal", "unit": "A"}},
        },
        "output": {"directory": str(tmp_path / "simple-output"), "overwrite": True},
    }
    overwritten = import_text_measurement(TextImportRecipe.from_mapping(overwrite_payload, base=tmp_path))
    assert overwritten.measurement.is_primary_complete


def test_channel_plot_contains_primary_and_auxiliary_panels(tmp_path):
    archive = tmp_path / "plot.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr("site1.dat", _instrument_file(1))
    result = import_text_measurement(_recipe(tmp_path, archive))
    figure = result.measurement.plot_spectroscopy()
    titles = [axis.get_title() for axis in figure.axes if axis.get_title()]
    assert any("didv (inference)" in title for title in titles)
    assert any("d2idv2 (plot/QC only)" in title for title in titles)


def test_import_command_line_workflow(tmp_path, capsys):
    source = tmp_path / "CLI raw files"
    source.mkdir()
    (source / "site1.txt").write_text("energy,signal\n0,1\n1,2", encoding="utf-8")
    payload = {
        "input": {"directory": str(source), "pattern": "*.txt"},
        "format": {"delimiter": ","},
        "columns": {
            "energy": {"column": "energy", "unit": "mV"},
            "signals": {"didv": {"column": "signal", "unit": "A"}},
        },
        "output": {"directory": str(tmp_path / "CLI output")},
    }
    recipe_path = tmp_path / "recipe with spaces.yaml"
    recipe_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert project_cli_main(["import-measurement", str(recipe_path)]) == 0
    output = capsys.readouterr().out
    assert "structurally ready for model compatibility checks" in output
    assert (tmp_path / "CLI output" / "spectroscopy.csv").exists()
