# Importing one spectrum file per site

The import layer converts laboratory-specific text or DAT exports into two
portable outputs:

- a long-form CSV compatible with the existing pipeline;
- a canonical NPZ measurement that also retains auxiliary channels, masks,
  units, selected metadata, and source provenance.

The instrument format is described in a YAML recipe. Nothing in the parser
depends on the example filenames or on one laboratory's header names.

```bash
cp examples/import_nanonis_like_dat.yaml my_import.yaml
# Edit the input path and exact column names, then choose the physics profile:
hamlet inspect-experiment my_import.yaml \
  --mode heisenberg --variant inhomogeneous \
  --output-dir experiments/my_chain
```

The guided command creates `spectroscopy.csv`, `measurement.npz`,
`import_report.json`, `import_preview.html`, `experiment_manifest.json`, and a
visual `experiment_inspection.html`. The JSON and
HTML reports list every file, site assignment, row and column counts, missing
values, axis reversals, extra columns, and whether the result is ready for
cutoff selection. “Structurally ready” means the primary channel
is complete and well aligned; it does not claim that the bias coverage or
distribution matches a particular trained artifact. That is checked later by
the project inspection/calibration stages.

The lower-level `hamlet import-measurement my_import.yaml` command remains
available when only format conversion is wanted. The guided command is the
preferred experimentalist entry point because it saves the physical mode and
mode-specific checks.

Existing outputs are protected by default. Choose a new output directory for a
new acquisition, or set `output.overwrite: true` only when intentionally
regenerating the same import products from the same recipe.

## Channel mapping

For the supplied example export, the intended mapping is:

| Canonical channel | Source column | Role |
|---|---|---|
| `didv` | `LI Demod 1 X (A)` | inverse-model input |
| `d2idv2` | `LI Demod 2 X (A)` | plotting and quality control only |

These names are examples, not package assumptions. Each lab edits
`columns.energy` and `columns.signals` to match its exact headers. Selection is
exact, so a new `[filt]` column cannot silently replace the requested raw
channel.

The CSV begins with the stable columns
`site,bias_meV,didv_A`. When `include_auxiliary_csv` is true it also contains
`d2idv2_A`; current Hamiltonian inference deliberately ignores auxiliary
columns. The richer NPZ is preferable for archiving because it records channel
roles and units explicitly.

## Units and ordering

Input bias may be declared as `V`, `mV`, `eV`, `meV`, `uV`, or `ueV`; it is
converted to meV. This is the electron spectroscopy convention in which 1 mV
of sample bias corresponds numerically to 1 meV. The importer can reverse each
file to ascending bias and naturally sorts numbered filenames before assigning
sequential chain sites.

Use `site.mode: filename_number` when acquisition identifiers should be kept,
or `site.mode: filename` for complete filename stems. The order in the
canonical CSV is physical chain order; inspect it before inference.

## Missing data and mismatched grids

The recipe requires an explicit policy:

- `missing: error` stops on any missing dI/dV point;
- `missing: drop_common` retains only bias points valid for dI/dV at every
  site;
- `missing: keep` preserves masks but marks the result as not inference-ready.

Grid mismatches stop by default. `grid: interpolate` is available, but should
only be enabled when interpolation onto the first spectrum's grid is
scientifically justified. The report always records that interpolation was
used.

## Python interface

```python
from hamlet.io import import_text_measurement

result = import_text_measurement("my_import.yaml")
measurement = result.measurement

# Plot dI/dV and optional second derivative side by side.
measurement.plot_spectroscopy().savefig("raw_channels.png", dpi=180)

# Inference receives only measurement.primary_channel (normally dI/dV).
analysis = analyzer.analyze_measurement(measurement)
```

The `Measurement` type itself is not Heisenberg-specific: it stores named axes
and aligned named channels, so future QPI or other systems can use different
axis sets without changing the text-file parser contract.
