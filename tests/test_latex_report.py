"""The LaTeX summary: the artifact people send to a collaborator.

The HTML report is for reading where it was made. This one leaves the
machine, so what matters is that it compiles somewhere else, that its formulas
match the Hamiltonian the simulator actually builds, and that nothing in a
file path or a warning can break the document.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hamlet.experimental import latex_report


def test_every_supported_system_has_a_hamiltonian():
    """A report is worthless if it shows the wrong Hamiltonian."""
    # Every system type the package can generate. Listed against the
    # validator's own set below, so a new system cannot be added without a
    # Hamiltonian to print for it.
    supported = {
        "inhomogeneous_heisenberg",
        "homogeneous_heisenberg",
        "homogeneous_xxz_j1j2j3",
        "homogeneous_xxz_j1j2j3_dmi",
        "homogeneous_xxz_j1j2j3_dmi_impurity",
    }
    from hamlet.project import DatasetGenerationConfig
    import inspect

    declared = inspect.getsource(DatasetGenerationConfig.__post_init__)
    for name in supported:
        assert name in declared, f"{name} is no longer a supported system type"
    assert supported <= set(latex_report.HAMILTONIANS), (
        "a system type can be inferred but has no Hamiltonian to print"
    )
    for name, entry in latex_report.HAMILTONIANS.items():
        assert entry["equation"].strip(), name
        assert entry["reading"].strip(), name


def test_an_unknown_system_type_does_not_lose_the_report():
    """The numbers are already computed; a missing field must not discard them."""
    fallback = latex_report.hamiltonian_for("something-new")
    assert fallback["equation"]
    assert "placeholder" in fallback["reading"]
    assert latex_report.hamiltonian_for(None)["equation"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("C:\\Users\\tiago\\data_set.dat", r"\textbackslash{}"),
        ("100% of the signal", r"\%"),
        ("a_b_c", r"\_"),
        ("cost $5 & up", r"\&"),
        ("x^2 ~ y", r"\textasciicircum{}"),
    ],
)
def test_tex_escaping_survives_the_things_users_actually_have(raw, expected):
    """A Windows path alone carries enough backslashes to break a document."""
    escaped = latex_report.tex_escape(raw)
    assert expected in escaped


class _Diagnostics:
    status = "ok"
    warnings = ("a 100% mismatch in file C:\\data\\run_1.dat",)
    aggregation_method = "mean"
    max_ensemble_std = 0.01
    n_members = 3


class _Result:
    """The smallest thing shaped like a chain result."""

    source = "/tmp/some_path/measurement_1.csv"
    n_sites = 4
    n_bonds = 3
    coupling_unit = "meV"
    coupling_mean = (30.0, 31.5, 29.25)
    coupling_std = (0.1, 0.2, 0.15)
    diagnostics = _Diagnostics()


def test_the_document_contains_the_method_the_formula_and_the_numbers():
    source = latex_report.build_latex_document(
        _Result(),
        title="Chain S1",
        manifest={"system_type": "inhomogeneous_heisenberg", "model_name": "ridge"},
    )
    assert r"\documentclass" in source and r"\end{document}" in source
    assert "Chain S1" in source
    # The physics.
    assert r"\hat{\mathbf{S}}" in source
    assert "Bond-inhomogeneous Heisenberg chain" in source
    # The numbers.
    assert "30.0000" in source and "31.5000" in source
    # The caveat that must travel with every estimate.
    assert "not a calibrated confidence interval" in source
    # And the warning, escaped rather than dropped.
    assert r"100\%" in source
    assert r"\textbackslash{}" in source


def test_the_provenance_table_shows_values_not_python_reprs():
    """Nested manifest fields would otherwise print a line of Python."""
    source = latex_report.build_latex_document(
        _Result(),
        title="t",
        manifest={
            "system_type": "inhomogeneous_heisenberg",
            "training_preset": {"name": "quick", "epochs": 20, "seeds": [42]},
            "ensemble_aggregation": {"method": "mean", "weights": None},
            "energy_convention": {"dmrgpy_energy_unit_mev": 10.0},
        },
    )
    assert "quick" in source
    assert "'epochs'" not in source, "a Python repr reached the document"
    assert "weights" not in source
    assert "1 DMRGPy energy unit = 10 meV" in source


def test_a_missing_field_is_reported_not_omitted():
    """A reader deciding whether to trust a number must see an empty field."""
    source = latex_report.build_latex_document(
        _Result(), title="t", manifest={"system_type": "inhomogeneous_heisenberg"}
    )
    assert "not recorded" in source


def test_no_latex_installed_still_writes_the_source(tmp_path, monkeypatch):
    """The .tex is the durable artifact; Overleaf needs nothing installed here."""
    monkeypatch.setattr(latex_report, "find_latex_compiler", lambda: None)
    outcome = latex_report.save_latex_report(
        _Result(), tmp_path / "report.tex",
        artifact_manifest={"system_type": "inhomogeneous_heisenberg"},
    )
    assert Path(outcome["tex_path"]).exists()
    assert outcome["compiled"] is False
    assert outcome["pdf_path"] is None
    assert "Overleaf" in outcome["detail"]


latex_available = pytest.mark.skipif(
    latex_report.find_latex_compiler() is None,
    reason="no LaTeX toolchain on this machine",
)


@latex_available
def test_the_document_actually_compiles(tmp_path):
    """The only check that matters for a document: does LaTeX accept it."""
    outcome = latex_report.save_latex_report(
        _Result(), tmp_path / "report.tex",
        title="Chain S1: 100% & _underscored_",
        artifact_manifest={"system_type": "homogeneous_xxz_j1j2j3_dmi_impurity"},
    )
    assert outcome["compiled"], outcome["detail"]
    pdf = Path(outcome["pdf_path"])
    assert pdf.exists() and pdf.stat().st_size > 1000
    # And it leaves the results directory clean.
    leftovers = sorted(p.name for p in tmp_path.iterdir() if p.suffix in
                       (".aux", ".log", ".fls", ".fdb_latexmk", ".out"))
    assert not leftovers, f"build clutter left in the analysis folder: {leftovers}"


@latex_available
@pytest.mark.parametrize("system_type", sorted(latex_report.HAMILTONIANS))
def test_every_hamiltonian_compiles(tmp_path, system_type):
    """A formula with one unbalanced brace breaks only that system's report."""
    outcome = latex_report.save_latex_report(
        _Result(), tmp_path / f"{system_type}.tex",
        artifact_manifest={"system_type": system_type},
    )
    assert outcome["compiled"], f"{system_type}: {outcome['detail']}"
