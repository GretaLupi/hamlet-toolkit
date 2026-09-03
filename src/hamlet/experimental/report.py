"""Self-contained HTML reports for experimental Hamiltonian inference."""

from __future__ import annotations

import base64
from dataclasses import asdict
from html import escape
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping

from .analysis import ExperimentalChainResult, ExperimentalGlobalResult


def save_html_report(
    result: ExperimentalChainResult | ExperimentalGlobalResult,
    path: str | Path,
    *,
    title: str = "HamLeT analysis",
    artifact_manifest: str | Path | Mapping[str, Any] | None = None,
) -> Path:
    """Write a portable report with plots, estimates, diagnostics, and provenance."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(artifact_manifest)
    image = _summary_image(result)
    warnings = result.diagnostics.warnings
    warning_html = (
        "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in warnings) + "</ul>"
        if warnings
        else "<p>No automatic warnings.</p>"
    )
    if isinstance(result, ExperimentalGlobalResult):
        rows = "".join(
            "<tr>"
            f"<td>{escape(name)}</td><td>{distance}</td>"
            f"<td>{float(mean):.4f}</td><td>{float(std):.4f}</td>"
            f"<td>{escape(result.coupling_unit)}</td></tr>"
            for distance, (name, mean, std) in enumerate(
                zip(result.parameter_names, result.coupling_mean, result.coupling_std),
                start=1,
            )
        )
        table_header = (
            "<tr><th>Parameter</th><th>Interaction distance</th><th>Estimate</th>"
            "<th>Member σ</th><th>Unit</th></tr>"
        )
        count_label = "Parameters"
        count_value = result.n_parameters
        result_heading = "Inferred homogeneous exchange parameters"
    else:
        rows = "".join(
            "<tr>"
            f"<td>J{index + 1}</td><td>{index}</td><td>{index + 1}</td>"
            f"<td>{float(mean):.4f}</td><td>{float(std):.4f}</td>"
            f"<td>{escape(result.coupling_unit)}</td></tr>"
            for index, (mean, std) in enumerate(
                zip(result.coupling_mean, result.coupling_std)
            )
        )
        table_header = (
            "<tr><th>Bond</th><th>Left site</th><th>Right site</th><th>Estimate</th>"
            "<th>Member σ</th><th>Unit</th></tr>"
        )
        count_label = "Bonds"
        count_value = result.n_bonds
        result_heading = "Inferred exchange couplings"
    diagnostics = asdict(result.diagnostics)
    key_metrics = {
        "Sites": result.n_sites,
        count_label: count_value,
        "Cutoff": _manifest_cutoff(manifest),
        "Aggregation": result.diagnostics.aggregation_method,
        "Max member σ": (
            f"{result.diagnostics.max_ensemble_std:.3f} {result.coupling_unit}"
        ),
    }
    cards = "".join(
        f"<div class='metric'><span>{escape(str(label))}</span><strong>{escape(str(value))}</strong></div>"
        for label, value in key_metrics.items()
    )
    image_html = (
        f"<img class='summary' alt='Analysis summary' src='data:image/png;base64,{image}'>"
        if image
        else "<p>Summary plot unavailable; numerical results remain complete.</p>"
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title>
<style>
:root{{--ink:#15222d;--muted:#607080;--paper:#f6f8fa;--card:#fff;--accent:#16697a;--ok:#237a57;--review:#a65f00}}
body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:32px 20px 60px}}
h1{{margin-bottom:4px}} .source{{color:var(--muted);word-break:break-all}}
.status{{display:inline-block;padding:5px 11px;border-radius:18px;color:white;background:var(--review);font-weight:700}}
.status.ok{{background:var(--ok)}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;margin:22px 0}}
.metric,section{{background:var(--card);border:1px solid #dde3e8;border-radius:10px;padding:16px}}
.metric span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase}}
.metric strong{{display:block;margin-top:4px;font-size:18px}}
section{{margin-top:16px}} .summary{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%}} th,td{{padding:9px;border-bottom:1px solid #e5e9ed;text-align:right}}
th:first-child,td:first-child{{text-align:left}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f6;padding:12px;border-radius:7px}}
footer{{margin-top:24px;color:var(--muted);font-size:13px}}
</style>
</head>
<body><main>
<span class="status {'ok' if result.diagnostics.status == 'ok' else ''}">{escape(result.diagnostics.status.upper())}</span>
<h1>{escape(title)}</h1>
<div class="source">{escape(result.source)}</div>
<div class="metrics">{cards}</div>
<section><h2>Quality-control overview</h2>{image_html}</section>
<section><h2>{result_heading}</h2>
<table><thead>{table_header}</thead><tbody>{rows}</tbody></table></section>
<section><h2>Automatic checks</h2>{warning_html}<pre>{escape(json.dumps(diagnostics, indent=2))}</pre></section>
<section><h2>Model and preprocessing provenance</h2><pre>{escape(json.dumps(_manifest_summary(manifest), indent=2))}</pre></section>
<footer>Generated by HamLeT — Hamiltonian Learning Toolkit. REVIEW means the numerical inference completed but at least one automatic quality gate requires inspection.</footer>
</main></body></html>"""
    destination.write_text(html, encoding="utf-8")
    return destination


def _summary_image(
    result: ExperimentalChainResult | ExperimentalGlobalResult,
) -> str | None:
    try:
        import matplotlib.pyplot as plt

        figure = result.plot_summary()
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=140)
        plt.close(figure)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except ImportError:  # pragma: no cover
        return None


def _load_manifest(
    source: str | Path | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_cutoff(manifest: Mapping[str, Any]) -> str:
    cutoff = manifest.get("preprocessing", {}).get("bias_cutoff_mev")
    return f"{float(cutoff):g} meV" if cutoff is not None else "not recorded"


def _manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not manifest:
        return {"available": False}
    return {
        "model_name": manifest.get("model_name"),
        "system_type": manifest.get("system_type"),
        "training_preset": manifest.get("training_preset"),
        "preprocessing": manifest.get("preprocessing"),
        "ensemble_aggregation": manifest.get("ensemble_aggregation"),
        "validation_metrics": manifest.get("metrics", {}).get("validation"),
        "test_metrics": manifest.get("metrics", {}).get("test"),
        "energy_convention": manifest.get("energy_convention"),
    }
