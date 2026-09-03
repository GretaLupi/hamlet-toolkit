import json

import numpy as np
import pandas as pd

from hamlet.experimental import ExperimentalChainAnalyzer
from hamlet.measurements import Measurement
from hamlet.preprocessing import SpectralPreprocessor
from hamlet.training import EnsembleAggregation


class ConstantWindowModel:
    def __init__(self, left, right):
        self.values = np.array([left, right], dtype=np.float32)

    def predict(self, windows, **kwargs):
        return np.tile(self.values, (len(windows), 1))


def write_experiment(path, n_sites=5):
    bias = np.linspace(0, 50, 101)
    rows = []
    for site in range(n_sites):
        for x, value in zip(bias, 1 + site + np.sin(bias / 10)):
            rows.append({"site": site + 1, "bias_meV": x, "didv_A": value})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_high_level_analyzer_loads_predicts_diagnoses_and_exports(tmp_path):
    input_path = tmp_path / "chain.csv"
    write_experiment(input_path)
    analyzer = ExperimentalChainAnalyzer(
        [ConstantWindowModel(0.2, 0.4), ConstantWindowModel(0.3, 0.5)],
        SpectralPreprocessor(output_points=40),
        output_range=(3.0, 4.0),
        coupling_unit="meV",
    )
    result = analyzer.analyze_csv(input_path)
    assert result.n_sites == 5
    assert result.coupling_mean.shape == (4,)
    assert result.coupling_std.shape == (4,)
    assert result.processed_spectra.shape == (5, 40)
    assert result.diagnostics.ensemble_size == 2
    assert result.diagnostics.status == "review"

    csv_path = tmp_path / "couplings.csv"
    json_path = tmp_path / "report.json"
    result.save_couplings_csv(csv_path)
    result.save_report_json(json_path)
    assert "uncertainty" in csv_path.read_text()
    report = json.loads(json_path.read_text())
    assert report["n_bonds"] == 4
    assert len(report["coupling_mean"]) == 4


def test_single_model_is_flagged_as_missing_uncertainty():
    bias = np.linspace(0, 50, 100)
    spectra = np.stack([bias + site for site in range(4)])
    analyzer = ExperimentalChainAnalyzer(
        [ConstantWindowModel(0.2, 0.2)], SpectralPreprocessor(output_points=20)
    )
    result = analyzer.analyze(spectra, bias)
    assert result.diagnostics.status == "review"
    assert "one model" in result.diagnostics.warnings[0]


def test_analyzer_uses_fixed_aggregation_but_keeps_member_spread():
    bias = np.linspace(0, 50, 100)
    spectra = np.stack([bias + site for site in range(4)])
    analyzer = ExperimentalChainAnalyzer(
        [
            ConstantWindowModel(0.0, 0.0),
            ConstantWindowModel(0.4, 0.4),
            ConstantWindowModel(1.0, 1.0),
        ],
        SpectralPreprocessor(output_points=20),
        aggregation=EnsembleAggregation("median"),
    )
    result = analyzer.analyze(spectra, bias)
    np.testing.assert_allclose(result.coupling_mean, 0.4)
    assert np.all(result.coupling_std > 0)
    assert result.diagnostics.aggregation_method == "median"


def test_analyzer_uses_only_declared_primary_measurement_channel():
    bias = np.linspace(0, 50, 100)
    didv = np.stack([bias + site for site in range(4)])
    measurement = Measurement(
        axes={"site": np.arange(4), "bias": bias},
        channels={"didv": didv, "d2idv2": np.full_like(didv, 1e30)},
        axis_units={"bias": "meV"},
        primary_channel="didv",
    )
    analyzer = ExperimentalChainAnalyzer(
        [ConstantWindowModel(0.2, 0.2)], SpectralPreprocessor(output_points=20)
    )
    canonical = analyzer.analyze_measurement(measurement)
    direct = analyzer.analyze(didv, bias)
    np.testing.assert_allclose(canonical.coupling_mean, direct.coupling_mean)
