"""A LaTeX summary of one analysis, and a PDF when a TeX toolchain is present.

The HTML report is for reading on the machine that produced it. This is for
the other thing people do with a result: send it to a collaborator, put it in
a group meeting, paste a paragraph of it into a draft. That audience needs the
Hamiltonian written out, the method stated in a few sentences, and a table
they can read without the interface -- not a page of collapsible diagnostics.

Every formula here is transcribed from what
``hamlet.simulation.dmrgpy.DmrgpySimulator`` actually builds, so the document
describes the model the numbers came from rather than a textbook convention
that resembles it.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

from .analysis import ExperimentalChainResult, ExperimentalGlobalResult

# Escapes for the ten characters TeX treats specially. Applied to every value
# that reaches the document from a path, a model name, or a warning -- a
# Windows path alone carries enough backslashes and underscores to turn a
# report into a compile error.
_TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(value: Any) -> str:
    """Escape TeX's special characters in one pass over the input.

    One pass, not a chain of `str.replace`: a chain escapes its own output.
    Replacing the backslash first yields `\textbackslash{}`, and the later
    rules for `{` and `}` then mangle the braces it just introduced --
    `C:\data` came out as `C:\textbackslash\{\}data`. Reading each input
    character exactly once makes that impossible rather than merely fixed.
    """
    return "".join(_TEX_ESCAPES.get(character, character) for character in str(value))


# One entry per system this package can infer. ``equation`` is the Hamiltonian
# as the simulator assembles it; ``reading`` says in words what the reader
# should take from it.
HAMILTONIANS: dict[str, dict[str, str]] = {
    "inhomogeneous_heisenberg": {
        "title": "Bond-inhomogeneous Heisenberg chain",
        "equation": r"\hat{H} = \sum_{i=1}^{N-1} J_{i}\,"
                    r"\hat{\mathbf{S}}_{i}\cdot\hat{\mathbf{S}}_{i+1}",
        "reading": (
            "Isotropic nearest-neighbour exchange with one independent coupling "
            "per bond. The inferred quantity is the list $J_i$, one number for "
            "each of the $N-1$ bonds."
        ),
    },
    "homogeneous_heisenberg": {
        "title": "Homogeneous Heisenberg chain",
        "equation": r"\hat{H} = \sum_{d} J_{d} \sum_{i=1}^{N-d}\,"
                    r"\hat{\mathbf{S}}_{i}\cdot\hat{\mathbf{S}}_{i+d}",
        "reading": (
            "Isotropic exchange shared by the whole chain, with one coupling "
            "per interaction distance $d$. The inferred quantity is $J_d$."
        ),
    },
    "homogeneous_xxz_j1j2j3": {
        "title": "Homogeneous XXZ chain with longer-range isotropic exchange",
        "equation": r"""\hat{H} = \sum_{i=1}^{N-1} \Big[
    J_{1}^{xy}\big(\hat{S}^{x}_{i}\hat{S}^{x}_{i+1}
                 + \hat{S}^{y}_{i}\hat{S}^{y}_{i+1}\big)
  + J^{z}\,\hat{S}^{z}_{i}\hat{S}^{z}_{i+1} \Big]
  + \sum_{d=2,3} J_{d} \sum_{i=1}^{N-d}
    \hat{\mathbf{S}}_{i}\cdot\hat{\mathbf{S}}_{i+d}""",
        "reading": (
            "Nearest neighbours are anisotropic: the in-plane coupling "
            "$J_1^{xy}$ and the axial coupling $J^{z}$ are independent. The "
            "second- and third-neighbour terms are isotropic."
        ),
    },
    "homogeneous_xxz_j1j2j3_dmi": {
        "title": "XXZ chain with uniform Dzyaloshinskii--Moriya interaction",
        "equation": r"""\hat{H} = \sum_{i=1}^{N-1} \Big[
    J_{1}^{xy}\big(\hat{S}^{x}_{i}\hat{S}^{x}_{i+1}
                 + \hat{S}^{y}_{i}\hat{S}^{y}_{i+1}\big)
  + J^{z}\,\hat{S}^{z}_{i}\hat{S}^{z}_{i+1}
  + D_{z}\big(\hat{S}^{x}_{i}\hat{S}^{y}_{i+1}
            - \hat{S}^{y}_{i}\hat{S}^{x}_{i+1}\big) \Big]
  + \sum_{d=2,3} J_{d} \sum_{i=1}^{N-d}
    \hat{\mathbf{S}}_{i}\cdot\hat{\mathbf{S}}_{i+d}""",
        "reading": (
            "The $D_z$ term is the $z$ component of a Dzyaloshinskii--Moriya "
            "vector. In a chain with no other symmetry breaking it can be "
            "removed by a site-dependent rotation about $z$ and is therefore "
            "\\emph{not} identifiable from an on-site autocorrelator; see the "
            "DMI experiment specification shipped with this package."
        ),
    },
    "homogeneous_xxz_j1j2j3_dmi_impurity": {
        "title": "XXZ chain with DMI, exposed by anisotropic impurities",
        "equation": r"""\hat{H} = \sum_{i=1}^{N-1} \Big[
    J_{1}^{xy}\big(\hat{S}^{x}_{i}\hat{S}^{x}_{i+1}
                 + \hat{S}^{y}_{i}\hat{S}^{y}_{i+1}\big)
  + J^{z}\,\hat{S}^{z}_{i}\hat{S}^{z}_{i+1}
  + D_{z}\big(\hat{S}^{x}_{i}\hat{S}^{y}_{i+1}
            - \hat{S}^{y}_{i}\hat{S}^{x}_{i+1}\big) \Big]
  + \sum_{d=2,3} J_{d} \sum_{i=1}^{N-d}
    \hat{\mathbf{S}}_{i}\cdot\hat{\mathbf{S}}_{i+d}
  + \sum_{a \in \text{imp}} \Big[
      D^{a}_{\parallel}\big(\hat{S}^{z}_{a}\big)^{2}
    + E^{a}\big(\cos 2\varphi_{a}\,
        \big[(\hat{S}^{x}_{a})^{2} - (\hat{S}^{y}_{a})^{2}\big]
      + \sin 2\varphi_{a}\,
        \big[\hat{S}^{x}_{a}\hat{S}^{y}_{a} + \hat{S}^{y}_{a}\hat{S}^{x}_{a}\big]
      \big) \Big]
  - B_{x}\sum_{i=1}^{N}\hat{S}^{x}_{i}""",
        "reading": (
            "The impurity terms are what make $D_z$ measurable. The axial term "
            "$D_\\parallel$ commutes with the total $\\hat{S}^z$ and cannot "
            "expose it; the transverse term $E$, whose in-plane axes sit at "
            "$\\varphi_a$ from $x$, changes $S^z$ by two and breaks the U(1) "
            "symmetry that would otherwise let a collinear DM vector be "
            "rotated away. Two impurities at distinct sites are required: a "
            "single one has its in-plane axis turned by one angle, which a "
            "global rotation about $z$ undoes. $B_x$ is an optional transverse "
            "field, an independent way to break the same symmetry; it is zero "
            "unless the design asks for it."
        ),
    },
}

_FALLBACK_HAMILTONIAN = {
    "title": "Spin chain",
    "equation": r"\hat{H} = \sum_{i} J_{i}\,"
                r"\hat{\mathbf{S}}_{i}\cdot\hat{\mathbf{S}}_{i+1}",
    "reading": (
        "The system type was not recorded in the model manifest, so the "
        "nearest-neighbour Heisenberg form is shown as a placeholder. Check "
        "the model card for the exact Hamiltonian this model was trained on."
    ),
}


def hamiltonian_for(system_type: str | None) -> dict[str, str]:
    """The Hamiltonian block for a system type, never raising on an unknown one.

    A report that refuses to build because a manifest is missing one field is
    worse than a report that says which field is missing: the numbers are
    already computed either way.
    """
    return HAMILTONIANS.get(str(system_type or ""), _FALLBACK_HAMILTONIAN)


def _result_rows(
    result: ExperimentalChainResult | ExperimentalGlobalResult,
) -> tuple[str, str, str]:
    """The results table: its column spec, header, and body."""
    unit = tex_escape(result.coupling_unit)
    if isinstance(result, ExperimentalGlobalResult):
        header = (
            r"Parameter & Distance $d$ & Estimate (" + unit + r") & "
            r"Ensemble $\sigma$ \\"
        )
        body = "".join(
            f"${tex_escape(name)}$ & {distance} & {float(mean):.4f} & "
            f"{float(std):.4f} \\\\\n"
            for distance, (name, mean, std) in enumerate(
                zip(result.parameter_names, result.coupling_mean, result.coupling_std),
                start=1,
            )
        )
        return "l r r r", header, body
    header = (
        r"Bond & Sites & Estimate (" + unit + r") & Ensemble $\sigma$ \\"
    )
    body = "".join(
        f"$J_{{{index + 1}}}$ & {index}--{index + 1} & {float(mean):.4f} & "
        f"{float(std):.4f} \\\\\n"
        for index, (mean, std) in enumerate(
            zip(result.coupling_mean, result.coupling_std)
        )
    )
    return "l c r r", header, body


def _provenance_rows(manifest: Mapping[str, Any]) -> str:
    """What produced the numbers, as a two-column table.

    Blank entries are shown as "not recorded" rather than omitted: a reader
    deciding whether to trust a number needs to see that the field was empty,
    not to wonder whether the row was simply left out.
    """
    metrics = manifest.get("metrics") or {}
    test = (metrics.get("test") or {}).get("ensemble") or {}
    preprocessing = manifest.get("preprocessing") or {}
    # Several of these fields are nested mappings in the manifest. Printing
    # their repr puts a line of Python into a physics document and runs off
    # the page, so each is reduced to the one value a reader is looking for.
    entries = [
        ("Model", manifest.get("model_name")),
        ("System type", manifest.get("system_type")),
        ("Training preset", _named(manifest.get("training_preset"))),
        ("Bias cutoff", _with_unit(preprocessing.get("bias_cutoff_mev"), "meV")),
        ("Output points", preprocessing.get("output_points")),
        ("Observable", preprocessing.get("observable")),
        (
            "Ensemble aggregation",
            _field(manifest.get("ensemble_aggregation"), "method"),
        ),
        ("Held-out MAE", _metric(test.get("mae"))),
        ("Simulation solver", _solver(manifest)),
        ("Energy convention", _energy_convention(manifest.get("energy_convention"))),
    ]
    return "".join(
        f"{tex_escape(label)} & {tex_escape(value if value not in (None, '') else 'not recorded')} \\\\\n"
        for label, value in entries
    )


def _solver(manifest: Mapping[str, Any]) -> str:
    """Which solver produced the training spectra, said as a word.

    ED and DMRG are not the same calculation, and a reader comparing two
    models should not have to open the dataset to find out which one they are
    looking at.
    """
    recipe = (manifest.get("dataset_metadata") or {}).get("generation_recipe") or {}
    mode = str(recipe.get("dynamics_mode") or "")
    return {
        "ED": "exact diagonalisation",
        "DMRG": "DMRG",
        "auto": "chosen automatically from the Hilbert space size",
    }.get(mode, mode)


def _named(value: Any) -> Any:
    """A preset is either a name or a mapping that contains one."""
    if isinstance(value, Mapping):
        return value.get("name") or ""
    return value


def _field(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key) or ""
    return value


def _energy_convention(value: Any) -> str:
    """The simulator's unit, as the sentence it actually means."""
    if isinstance(value, Mapping):
        mev = value.get("dmrgpy_energy_unit_mev")
        if mev is not None:
            return f"1 DMRGPy energy unit = {_with_unit(mev, 'meV')}"
        return ""
    return "" if value is None else str(value)


def _with_unit(value: Any, unit: str) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):g} {unit}"
    except (TypeError, ValueError):
        return str(value)


def _trained_ranges(manifest: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """The sampled range of each learned parameter, keyed by name.

    Read from the generation recipe, because that is the statement of what the
    model was actually shown. Outside it a prediction is an extrapolation, and
    saying so is the difference between a number and a number a reader can act
    on.
    """
    recipe = (manifest.get("dataset_metadata") or {}).get("generation_recipe") or {}
    ranges = recipe.get("coupling_ranges_mev") or []
    single = _shared_range(manifest)
    names = list(manifest.get("target_names") or [])
    out: dict[str, tuple[float, float]] = {}
    for index, name in enumerate(names):
        span = None
        if index < len(ranges) and isinstance(ranges[index], (list, tuple)):
            span = ranges[index]
        elif isinstance(single, (list, tuple)) and len(single) == 2:
            # One range shared by every bond, as the inhomogeneous family
            # records it.
            span = single
        if span and len(span) == 2:
            try:
                out[str(name)] = (float(span[0]), float(span[1]))
            except (TypeError, ValueError):
                continue
    return out


def _shared_range(manifest: Mapping[str, Any]) -> tuple[float, float] | None:
    """One range covering every bond, as the inhomogeneous family records it.

    A bond-resolved model learns many couplings from a single sampled range,
    so there is nothing to key by name: the same interval applies to all of
    them.
    """
    recipe = (manifest.get("dataset_metadata") or {}).get("generation_recipe") or {}
    span = recipe.get("coupling_range_mev")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        try:
            return float(span[0]), float(span[1])
        except (TypeError, ValueError):
            return None
    return None


def _accuracy_block(manifest: Mapping[str, Any]) -> str:
    """How well the model recovered couplings it had never seen.

    Without this a reader has the estimate and no way to weigh it. It is the
    first question anyone sensible asks of an inferred number, and it is
    already in the manifest.
    """
    metrics = manifest.get("metrics") or {}
    test = metrics.get("test") or {}
    ensemble = test.get("ensemble") or {}
    if not ensemble:
        return (
            "This artifact records no held-out evaluation, so the accuracy of "
            "the inverse map it implements cannot be quoted here. Check the "
            "model card.\n"
        )

    split = metrics.get("split") or {}
    chains = split.get("test_groups") or test.get("n_examples")
    scope = (
        f"on {tex_escape(chains)} simulated chains withheld from both fitting "
        "and model selection"
        if chains
        else "on simulated data withheld from both fitting and model selection"
    )

    summary = [
        ("Mean absolute error", _metric(ensemble.get("mae"))),
        ("Root mean square error", _metric(ensemble.get("rmse"))),
        ("Fidelity", _rounded(ensemble.get("correlation_fidelity"))),
        ("Skill over the training mean", _rounded(ensemble.get("skill"))),
    ]
    rows = "".join(
        f"{tex_escape(label)} & {tex_escape(value or 'not recorded')} \\\\\n"
        for label, value in summary
    )

    per_target = test.get("per_target") or []
    if per_target:
        body = "".join(
            f"\\texttt{{{tex_escape(entry.get('name'))}}} & "
            f"{tex_escape(_metric(entry.get('mae'), ''))} & "
            f"{tex_escape(_rounded(entry.get('correlation_fidelity')))} & "
            f"{tex_escape(_rounded(entry.get('skill')))} \\\\\n"
            for entry in per_target
        )
        detail = (
            "\n\\begin{center}\n\\begin{tabular}{l r r r}\n\\hline\n"
            "Parameter & MAE (meV) & Fidelity & Skill \\\\\n\\hline\n"
            f"{body}\\hline\n\\end{{tabular}}\n\\end{{center}}\n"
            "\nA model can recover one coupling well and another hardly at "
            "all, which an overall figure hides; the per-parameter breakdown "
            "is the one to read.\n"
        )
    else:
        detail = ""

    return (
        f"Measured {scope}:\n\n"
        "\\begin{center}\n\\begin{tabular}{l r}\n\\hline\n"
        f"{rows}\\hline\n\\end{{tabular}}\n\\end{{center}}\n"
        f"{detail}"
        "\nFidelity is the absolute Pearson correlation between predicted and "
        "true values,\n"
        "\\begin{equation*}\n"
        "F = \\frac{\\big|\\langle (J_\\text{pred} - \\langle J_\\text{pred}\\rangle)"
        "(J_\\text{true} - \\langle J_\\text{true}\\rangle)\\rangle\\big|}"
        "{\\sigma_\\text{pred}\\,\\sigma_\\text{true}},\n"
        "\\end{equation*}\n"
        "running from 0 to 1. Being a correlation it is blind to a constant "
        "offset and to a scale factor, so it should be read next to the mean "
        "absolute error rather than instead of it. Skill is "
        "$1 - \\text{MAE}/\\text{MAE}_\\text{baseline}$ against always "
        "returning the training mean: zero means the spectrum contributed "
        "nothing.\n\n"
        "These scores describe \\emph{simulated} spectra from the same "
        "generator that produced the training set. They quantify the inverse "
        "problem, not whether the model Hamiltonian describes this material.\n"
    )


def _validity_block(
    result: ExperimentalChainResult | ExperimentalGlobalResult,
    manifest: Mapping[str, Any],
) -> str:
    """Whether each estimate landed inside the range the model was trained on.

    A regression model returns a number everywhere, including well outside the
    data it saw. Checking it here means a reader does not have to hold the
    training range in their head while looking at the results table.
    """
    ranges = _trained_ranges(manifest)
    shared = _shared_range(manifest)
    if not ranges and shared is None:
        return (
            "The sampled parameter range is not recorded in this artifact's "
            "manifest, so whether the estimates below are interpolations "
            "cannot be checked automatically. The model card states the range "
            "the model was trained for.\n"
        )

    rows, outside = [], []
    for name, estimate in _estimates(result):
        span = ranges.get(name, shared)
        if span is None:
            continue
        low, high = min(span), max(span)
        beyond = estimate < low or estimate > high
        if beyond:
            outside.append(name)
        # Built outside the f-string: a backslash in an f-string expression
        # is a syntax error before Python 3.12, and this package supports 3.10.
        verdict_cell = "\\textbf{outside}" if beyond else "inside"
        rows.append(
            f"\\texttt{{{tex_escape(name)}}} & {estimate:.3g} & "
            f"[{low:.3g}, {high:.3g}] & {verdict_cell} \\\\\n"
        )
    if not rows:
        return ""

    verdict = (
        "Every estimate lies inside the range the model was trained on, so "
        "none of them is an extrapolation."
        if not outside
        else "\\textbf{"
        + tex_escape(", ".join(outside))
        + "} lies outside the range the model was trained on. A regression "
        "model returns a number there too, but it was never shown data of "
        "that kind, and the accuracy above does not cover it. Treat those "
        "entries as indicative at best."
    )
    return (
        "\\begin{center}\n\\begin{tabular}{l r c l}\n\\hline\n"
        "Parameter & Estimate (meV) & Trained range (meV) & \\\\\n\\hline\n"
        + "".join(rows)
        + "\\hline\n\\end{tabular}\n\\end{center}\n\n"
        + verdict
        + "\n"
    )


def _estimates(
    result: ExperimentalChainResult | ExperimentalGlobalResult,
) -> list[tuple[str, float]]:
    """(name, value) for whatever shape of result this is."""
    pairs: list[tuple[str, float]] = []
    if isinstance(result, ExperimentalGlobalResult):
        for name, value in zip(result.parameter_names, result.coupling_mean):
            try:
                pairs.append((str(name), float(value)))
            except (TypeError, ValueError):
                continue
        return pairs

    # A chain result is one estimate per bond of the same coupling, so the
    # range check is about the extremes: if the softest and the stiffest bond
    # both sit inside the sampled range, so does everything between them.
    values = [float(value) for value in result.coupling_mean]
    if not values:
        return pairs
    names = list(getattr(result, "parameter_names", None) or ())
    name = str(names[0]) if names else "J"
    pairs.append((name, min(values)))
    if max(values) != min(values):
        pairs.append((name, max(values)))
    return pairs


def _rounded(value: Any) -> str:
    """A bounded score, to four decimals.

    Fidelity sits close to one for a model worth using, so three decimals
    round several genuinely different models to "1.000".
    """
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _metric(value: Any, unit: str = "meV") -> str:
    """An error in physical units, to four significant figures.

    `%g` prints whatever the float happens to be -- 0.113957 meV states a
    precision the number does not have and that nobody asked for.
    """
    if value is None:
        return ""
    try:
        number = f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)
    return f"{number} {unit}".strip()


def build_latex_document(
    result: ExperimentalChainResult | ExperimentalGlobalResult,
    *,
    title: str,
    manifest: Mapping[str, Any],
    figure_name: str | None = None,
) -> str:
    """The .tex source, complete and standalone.

    Only amsmath, graphicx and geometry are required, so it compiles on a
    minimal TeX install. A collaborator who wants to reuse a paragraph should
    not first have to install a package they have never heard of.
    """
    system_type = manifest.get("system_type")
    physics = hamiltonian_for(system_type)
    column_spec, header, body = _result_rows(result)
    warnings = result.diagnostics.warnings
    if warnings:
        warning_block = (
            "\\begin{itemize}\n"
            + "".join(f"  \\item {tex_escape(item)}\n" for item in warnings)
            + "\\end{itemize}\n"
        )
    else:
        warning_block = "No automatic check raised a warning.\n"
    status = result.diagnostics.status
    status_sentence = (
        "Every automatic quality gate passed."
        if status == "ok"
        else "At least one automatic gate asks for inspection; see the checks below. "
        "The numerical inference itself completed."
    )
    figure_block = (
        "\\begin{figure}[h]\n"
        "\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{{figure_name}}}\n"
        "\\caption{Quality-control overview: measured spectra, the inferred "
        "couplings, and the ensemble spread across trained members.}\n"
        "\\end{figure}\n"
        if figure_name
        else ""
    )
    accuracy_block = _accuracy_block(manifest)
    validity_block = _validity_block(result, manifest)
    n_members = getattr(result.diagnostics, "n_members", None)
    members_sentence = (
        f"The estimate is the {tex_escape(result.diagnostics.aggregation_method)} "
        f"over {int(n_members)} independently trained members."
        if isinstance(n_members, int) and n_members > 0
        else f"Estimates are aggregated across the trained members by "
             f"{tex_escape(result.diagnostics.aggregation_method)}."
    )
    return f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[margin=25mm]{{geometry}}
\\usepackage{{amsmath}}
\\usepackage{{graphicx}}
\\setlength{{\\parskip}}{{0.6em}}
\\setlength{{\\parindent}}{{0pt}}

\\title{{{tex_escape(title)}}}
\\author{{Prepared with HamLeT, the Hamiltonian Learning Toolkit}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\section*{{What was done}}

A site-resolved scanning-tunnelling spectroscopy measurement was compared
against a supervised model trained on simulated spectra, and the exchange
couplings of the underlying spin chain were inferred from it.
{status_sentence}

\\textbf{{Measurement:}} \\texttt{{{tex_escape(result.source)}}}

\\section*{{The model that was inferred}}

{tex_escape(physics['title'])}, on {result.n_sites} sites:

\\begin{{equation*}}
{physics['equation']}
\\end{{equation*}}

{physics['reading']}

\\section*{{Method}}

The inference is supervised rather than variational. Spectra were simulated
for spin chains with known couplings drawn from a stated parameter range,
using density-matrix renormalisation group and exact diagonalisation; a
regression model was trained to invert that map; and the measured spectrum was
passed through it. The model therefore carries a contract -- chain length,
bias window, observable, preprocessing -- and the measurement is refused if it
does not match, rather than extrapolated.

{members_sentence} The ensemble spread quoted below measures disagreement
between those members. It is not a calibrated confidence interval, and a
narrow spread is not by itself evidence that the estimate is correct.

\\section*{{Result}}

\\begin{{center}}
\\begin{{tabular}}{{{column_spec}}}
\\hline
{header}
\\hline
{body}\\hline
\\end{{tabular}}
\\end{{center}}

{figure_block}
\\section*{{Is this inside what the model knows?}}

{validity_block}
\\section*{{How accurate is this model?}}

{accuracy_block}
\\section*{{Automatic checks}}

{warning_block}
\\section*{{Provenance}}

\\begin{{center}}
\\begin{{tabular}}{{l p{{0.62\\linewidth}}}}
\\hline
Field & Value \\\\
\\hline
{_provenance_rows(manifest)}\\hline
\\end{{tabular}}
\\end{{center}}

\\section*{{Citing this}}

If this analysis appears in published work, please cite the HamLeT package
together with the reference given in the model card of the artifact used
above. The package implements the Hamiltonian families and measurement logic
of the published impurity-tomography and nanographene spin-chain work; a model
you trained yourself is your own, but the method is theirs.

\\vfill
\\footnotesize
Energies are in meV. Scores quoted here describe held-out simulated data
unless the model card says otherwise. A measurement must satisfy the artifact
contract before its predictions are physically interpretable. The Hamiltonians,
the correlator-to-dI/dV relation, and the metric definitions are set out in the
package's theory notes.

\\end{{document}}
"""


# The compilers worth trying, in order. `tectonic` is first because it fetches
# what a document needs and so succeeds on a machine with no full TeX Live;
# `latexmk` and `pdflatex` cover the ordinary installations.
_COMPILERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tectonic", ("--keep-logs", "--print")),
    ("latexmk", ("-pdf", "-interaction=nonstopmode", "-halt-on-error")),
    ("pdflatex", ("-interaction=nonstopmode", "-halt-on-error")),
)


def find_latex_compiler() -> tuple[str, tuple[str, ...]] | None:
    for name, flags in _COMPILERS:
        found = shutil.which(name)
        if found:
            return found, flags
    return None


def save_latex_report(
    result: ExperimentalChainResult | ExperimentalGlobalResult,
    path: str | Path,
    *,
    title: str = "HamLeT analysis",
    artifact_manifest: str | Path | Mapping[str, Any] | None = None,
    compile_pdf: bool = True,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Write ``report.tex``, and a PDF beside it when a compiler is installed.

    Returns what happened rather than raising when LaTeX is absent or fails:
    the .tex is the durable artifact and is useful on its own -- a
    collaborator with Overleaf needs nothing else -- so a missing toolchain
    must not lose the analysis it describes.
    """
    from .report import _load_manifest, _summary_image

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(artifact_manifest)

    # The figure is written beside the .tex rather than embedded, because TeX
    # has no equivalent of a data URI and a relative \includegraphics is what
    # survives being sent to someone else as a folder or a zip.
    figure_name = None
    try:
        encoded = _summary_image(result)
    except Exception:  # noqa: BLE001 - a plot is not worth the whole report
        # Anything from a headless backend to a result that cannot draw
        # itself. The tables and the Hamiltonian are the substance; a missing
        # figure costs a paragraph of the document, not the analysis.
        encoded = None
    if encoded:
        import base64

        figure_name = destination.stem + "-summary.png"
        (destination.parent / figure_name).write_bytes(base64.b64decode(encoded))

    source = build_latex_document(
        result, title=title, manifest=manifest, figure_name=figure_name
    )
    destination.write_text(source, encoding="utf-8")

    outcome: dict[str, Any] = {
        "tex_path": str(destination),
        "figure_path": str(destination.parent / figure_name) if figure_name else None,
        "pdf_path": None,
        "compiled": False,
        "compiler": None,
        "detail": "",
    }
    if not compile_pdf:
        outcome["detail"] = "PDF compilation was not requested."
        return outcome

    compiler = find_latex_compiler()
    if compiler is None:
        outcome["detail"] = (
            "No LaTeX compiler was found, so only the .tex was written. Install "
            "tectonic, latexmk, or pdflatex to get a PDF -- or upload the .tex "
            "and its PNG to Overleaf, which needs nothing installed here."
        )
        return outcome

    executable, flags = compiler
    outcome["compiler"] = Path(executable).name
    try:
        completed = subprocess.run(
            [executable, *flags, destination.name],
            cwd=destination.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        outcome["detail"] = f"{type(exc).__name__}: {exc}"
        return outcome

    pdf = destination.with_suffix(".pdf")
    if pdf.exists():
        outcome["pdf_path"] = str(pdf)
        outcome["compiled"] = True
        outcome["detail"] = f"Compiled with {outcome['compiler']}."
        _remove_build_files(destination)
    else:
        # The log is far more useful than the exit code, and LaTeX puts the
        # real complaint in the last few lines of stdout.
        tail = "\n".join((completed.stdout or "").strip().splitlines()[-12:])
        outcome["detail"] = (
            f"{outcome['compiler']} did not produce a PDF. The .tex is still "
            f"here and compiles on Overleaf.\n{tail}"
        )
    return outcome


# What a TeX run leaves behind. Removed on success only: on failure the log is
# the one thing worth having, and the analysis directory is somewhere a user
# browses for their results -- five build files sitting next to report.pdf
# make it look like a scratch directory rather than an answer.
_BUILD_SUFFIXES = (".aux", ".fdb_latexmk", ".fls", ".log", ".out", ".synctex.gz")


def _remove_build_files(tex_path: Path) -> None:
    for suffix in _BUILD_SUFFIXES:
        candidate = tex_path.with_suffix(suffix)
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            # Tidiness is not worth failing a completed report over.
            pass


__all__ = [
    "HAMILTONIANS",
    "build_latex_document",
    "find_latex_compiler",
    "hamiltonian_for",
    "save_latex_report",
    "tex_escape",
]
