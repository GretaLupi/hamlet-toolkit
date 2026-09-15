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


def _manifest_with(**overrides):
    """A manifest shaped like a real artifact's, for the report blocks."""
    manifest = {
        "model_name": "ridge",
        "system_type": "homogeneous_heisenberg",
        "target_names": ["J1", "J2"],
        "coupling_unit": "meV",
        "preprocessing": {"bias_cutoff_mev": 50.0, "output_points": 61},
        "dataset_metadata": {
            "generation_recipe": {
                "coupling_ranges_mev": [[30.0, 45.0], [0.0, 10.0]],
                "dynamics_mode": "ED",
            }
        },
        "metrics": {
            "split": {"test_groups": 300},
            "test": {
                "ensemble": {
                    "mae": 0.11395712,
                    "rmse": 0.16384,
                    "correlation_fidelity": 0.99987,
                    "skill": 0.62,
                },
                "per_target": [
                    {"name": "J1", "mae": 0.09, "correlation_fidelity": 0.999, "skill": 0.7},
                    {"name": "J2", "mae": 0.14, "correlation_fidelity": 0.97, "skill": 0.4},
                ],
            },
        },
    }
    manifest.update(overrides)
    return manifest


def test_the_report_says_how_accurate_the_model_is():
    """An inferred number a reader cannot weigh is not a result.

    It is the first question anyone asks of an inference, and the answer is
    already in the manifest, so leaving it out of the document that gets sent
    to a colleague was the omission worth fixing.
    """
    from hamlet.experimental.latex_report import _accuracy_block

    block = _accuracy_block(_manifest_with())
    assert "300 simulated chains" in block
    # Four significant figures, not whatever the float happened to be.
    assert "0.114 meV" in block and "0.113957" not in block
    assert "0.9999" in block, "fidelity needs enough digits to distinguish models"
    # The per-parameter breakdown, because an overall figure hides a coupling
    # the model never learned.
    assert "J1" in block and "J2" in block
    assert "sigma_\\text{pred}" in block or "\\sigma_\\text{pred}" in block
    assert "blind to a constant offset" in block
    assert "simulated" in block


def test_an_artifact_without_an_evaluation_says_so_rather_than_implying_one():
    from hamlet.experimental.latex_report import _accuracy_block

    block = _accuracy_block(_manifest_with(metrics={}))
    assert "no held-out evaluation" in block
    assert "model card" in block


def test_an_estimate_outside_the_trained_range_is_flagged_as_extrapolation():
    """A regression model returns a number everywhere, trained or not."""
    import numpy as np

    from hamlet.experimental import ExperimentalGlobalResult
    from hamlet.experimental.latex_report import _validity_block

    class Diagnostics:
        warnings = ()
        status = "ok"
        aggregation_method = "median"
        n_members = 3

    def result_for(values):
        bias = np.linspace(0, 50, 5)
        return ExperimentalGlobalResult(
            source="demo.csv", raw_bias_mev=bias, raw_spectra=np.zeros((4, 5)),
            processed_bias_mev=bias, processed_spectra=np.zeros((4, 5)),
            parameter_names=("J1", "J2"),
            coupling_mean=np.asarray(values), coupling_std=np.asarray([0.5, 0.5]),
            per_model_couplings=np.zeros((3, 2)), coupling_unit="meV",
            diagnostics=Diagnostics(),
        )

    inside = _validity_block(result_for([36.0, 4.0]), _manifest_with())
    assert "none of them is an extrapolation" in inside
    assert "outside" not in inside.replace("\\textbf{outside}", "")

    beyond = _validity_block(result_for([36.0, 900.0]), _manifest_with())
    assert "\\textbf{outside}" in beyond
    assert "J2" in beyond
    assert "indicative at best" in beyond


def test_an_unrecorded_range_is_admitted_rather_than_assumed_safe():
    import numpy as np

    from hamlet.experimental import ExperimentalGlobalResult
    from hamlet.experimental.latex_report import _validity_block

    class Diagnostics:
        warnings = ()
        status = "ok"
        aggregation_method = "median"
        n_members = 1

    bias = np.linspace(0, 50, 5)
    result = ExperimentalGlobalResult(
        source="demo.csv", raw_bias_mev=bias, raw_spectra=np.zeros((4, 5)),
        processed_bias_mev=bias, processed_spectra=np.zeros((4, 5)),
        parameter_names=("J1", "J2"),
        coupling_mean=np.asarray([36.0, 4.0]), coupling_std=np.asarray([0.5, 0.5]),
        per_model_couplings=np.zeros((1, 2)), coupling_unit="meV",
        diagnostics=Diagnostics(),
    )
    block = _validity_block(result, _manifest_with(dataset_metadata={}))
    assert "not recorded" in block
    assert "cannot be checked automatically" in block


def test_the_impurity_hamiltonian_matches_the_one_that_is_simulated():
    """The report's formula and the code had drifted apart.

    The in-plane anisotropy axes sit at an angle the design can choose, and
    the transverse field is an independent way to break the same symmetry.
    A report that omits both describes a special case as if it were the model.
    """
    from hamlet.experimental.latex_report import hamiltonian_for

    block = hamiltonian_for("homogeneous_xxz_j1j2j3_dmi_impurity")
    equation = block["equation"]
    assert r"\cos 2\varphi" in equation and r"\sin 2\varphi" in equation
    assert "B_{x}" in equation, "the optional transverse field is part of it"
    assert "Two impurities at distinct sites" in block["reading"]
