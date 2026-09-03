"""High-level, diagnostics-first interface for experimental chain analysis."""

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..branding import brand_manifest
from ..io import load_measurement_csv
from ..measurements import Measurement
from ..preprocessing import SpectralPreprocessor, make_local_windows, reconstruct_chain
from ..training.scaling import MinMaxTargetScaler
from ..training.ensemble import EnsembleAggregation


@dataclass(frozen=True)
class ExperimentalDiagnostics:
    status: str
    warnings: tuple[str, ...]
    ensemble_size: int
    mean_overlap_disagreement: float
    max_ensemble_std: float
    fraction_predictions_outside_training_range: float
    aggregation_method: str = "mean"


@dataclass(frozen=True)
class ExperimentalChainResult:
    source: str
    site_labels: NDArray[Any]
    raw_bias_mev: NDArray[np.float64]
    raw_spectra: NDArray[np.float64]
    processed_bias_mev: NDArray[np.float64]
    processed_spectra: NDArray[np.float32]
    coupling_mean: NDArray[np.float32]
    coupling_std: NDArray[np.float32]
    per_model_couplings: NDArray[np.float32]
    per_model_window_predictions: NDArray[np.float32]
    overlap_disagreement: NDArray[np.float32]
    coupling_unit: str
    diagnostics: ExperimentalDiagnostics

    @property
    def n_sites(self) -> int:
        return int(self.raw_spectra.shape[0])

    @property
    def n_bonds(self) -> int:
        return self.n_sites - 1

    def save_couplings_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["bond", "left_site", "right_site", "coupling", "uncertainty", "unit"]
            )
            for bond, (mean, std) in enumerate(zip(self.coupling_mean, self.coupling_std)):
                writer.writerow(
                    [
                        bond,
                        self.site_labels[bond],
                        self.site_labels[bond + 1],
                        float(mean),
                        float(std),
                        self.coupling_unit,
                    ]
                )

    def save_report_json(self, path: str | Path) -> None:
        report = {
            "toolkit": brand_manifest(),
            "source": self.source,
            "n_sites": self.n_sites,
            "site_labels": self.site_labels.tolist(),
            "n_bonds": self.n_bonds,
            "coupling_unit": self.coupling_unit,
            "coupling_mean": self.coupling_mean.astype(float).tolist(),
            "coupling_std": self.coupling_std.astype(float).tolist(),
            "diagnostics": asdict(self.diagnostics),
        }
        Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    def save_html_report(
        self,
        path: str | Path,
        *,
        title: str = "HamLeT analysis",
        artifact_manifest: str | Path | dict[str, Any] | None = None,
    ) -> Path:
        """Save a self-contained, shareable analysis report."""
        from .report import save_html_report

        return save_html_report(
            self,
            path,
            title=title,
            artifact_manifest=artifact_manifest,
        )

    def plot_summary(self):
        """Return a four-panel quality-control figure."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("plotting requires matplotlib") from exc

        figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        raw = axes[0, 0].pcolormesh(
            self.raw_bias_mev,
            np.arange(self.n_sites),
            self.raw_spectra,
            shading="auto",
        )
        axes[0, 0].set(title="Raw experimental map", xlabel="bias [meV]", ylabel="site")
        figure.colorbar(raw, ax=axes[0, 0], label="dI/dV [a.u.]")

        processed = axes[0, 1].pcolormesh(
            self.processed_bias_mev,
            np.arange(self.n_sites),
            self.processed_spectra,
            shading="auto",
        )
        axes[0, 1].set(title="Model-ready map", xlabel="bias [meV]", ylabel="site")
        figure.colorbar(processed, ax=axes[0, 1], label="normalized response")

        bonds = np.arange(self.n_bonds)
        axes[1, 0].errorbar(
            bonds,
            self.coupling_mean,
            yerr=self.coupling_std,
            fmt="o-",
            capsize=4,
        )
        for model_values in self.per_model_couplings:
            axes[1, 0].plot(bonds, model_values, color="0.7", alpha=0.4, linewidth=1)
        axes[1, 0].set(
            title="Reconstructed couplings",
            xlabel="bond",
            ylabel=f"coupling [{self.coupling_unit}]",
        )

        interior_bonds = np.arange(1, self.n_bonds - 1)
        if self.overlap_disagreement.size:
            axes[1, 1].bar(interior_bonds, self.overlap_disagreement)
        axes[1, 1].set(
            title=f"Local consistency — {self.diagnostics.status.upper()}",
            xlabel="interior bond",
            ylabel=f"window disagreement [{self.coupling_unit}]",
        )
        if self.diagnostics.warnings:
            axes[1, 1].text(
                0.02,
                0.98,
                "\n".join(f"• {warning}" for warning in self.diagnostics.warnings),
                transform=axes[1, 1].transAxes,
                va="top",
                fontsize=8,
            )
        return figure


@dataclass(frozen=True)
class ExperimentalGlobalResult:
    """Inference result for homogeneous J1, J2, ... Hamiltonians."""

    source: str
    raw_bias_mev: NDArray[np.float64]
    raw_spectra: NDArray[np.float64]
    processed_bias_mev: NDArray[np.float64]
    processed_spectra: NDArray[np.float32]
    parameter_names: tuple[str, ...]
    coupling_mean: NDArray[np.float32]
    coupling_std: NDArray[np.float32]
    per_model_couplings: NDArray[np.float32]
    coupling_unit: str
    diagnostics: ExperimentalDiagnostics

    @property
    def n_sites(self) -> int:
        return int(self.raw_spectra.shape[0])

    @property
    def n_bonds(self) -> int:
        """Compatibility alias: global reports contain parameters, not bonds."""
        return len(self.parameter_names)

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names)

    def save_couplings_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            # Parameter order is not generally an interaction distance: for
            # example J1_xy and Jz are both nearest-neighbour terms, while D_z
            # may follow J3 in the target vector. Preserve the exact physical
            # names instead of exporting a misleading inferred distance.
            writer.writerow(["parameter", "coupling", "uncertainty", "unit"])
            for name, mean, std in zip(
                self.parameter_names, self.coupling_mean, self.coupling_std
            ):
                writer.writerow([name, float(mean), float(std), self.coupling_unit])

    def save_report_json(self, path: str | Path) -> None:
        report = {
            "toolkit": brand_manifest(),
            "source": self.source,
            "view": "global",
            "n_sites": self.n_sites,
            "n_parameters": self.n_parameters,
            "parameter_names": list(self.parameter_names),
            "coupling_unit": self.coupling_unit,
            "coupling_mean": self.coupling_mean.astype(float).tolist(),
            "coupling_std": self.coupling_std.astype(float).tolist(),
            "diagnostics": asdict(self.diagnostics),
        }
        Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    def save_html_report(
        self,
        path: str | Path,
        *,
        title: str = "HamLeT homogeneous analysis",
        artifact_manifest: str | Path | dict[str, Any] | None = None,
    ) -> Path:
        from .report import save_html_report

        return save_html_report(
            self, path, title=title, artifact_manifest=artifact_manifest
        )

    def plot_summary(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("plotting requires matplotlib") from exc

        figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        raw = axes[0, 0].pcolormesh(
            self.raw_bias_mev,
            np.arange(self.n_sites),
            self.raw_spectra,
            shading="auto",
        )
        axes[0, 0].set(title="Raw experimental map", xlabel="bias [meV]", ylabel="site")
        figure.colorbar(raw, ax=axes[0, 0], label="dI/dV [a.u.]")

        processed = axes[0, 1].pcolormesh(
            self.processed_bias_mev,
            np.arange(self.n_sites),
            self.processed_spectra,
            shading="auto",
        )
        axes[0, 1].set(title="Model-ready global map", xlabel="bias [meV]", ylabel="site")
        figure.colorbar(processed, ax=axes[0, 1], label="normalized response")

        x = np.arange(self.n_parameters)
        axes[1, 0].errorbar(
            x, self.coupling_mean, yerr=self.coupling_std, fmt="o-", capsize=4
        )
        for values in self.per_model_couplings:
            axes[1, 0].plot(x, values, color="0.7", alpha=0.5, linewidth=1)
        axes[1, 0].set(
            title="Homogeneous exchange parameters",
            xlabel="parameter",
            ylabel=f"coupling [{self.coupling_unit}]",
            xticks=x,
            xticklabels=self.parameter_names,
        )

        axes[1, 1].axis("off")
        axes[1, 1].set_title(f"Model checks — {self.diagnostics.status.upper()}")
        message = (
            "\n".join(f"• {warning}" for warning in self.diagnostics.warnings)
            if self.diagnostics.warnings
            else "No automatic warnings."
        )
        axes[1, 1].text(0.02, 0.96, message, va="top", fontsize=10)
        return figure


class ExperimentalGlobalAnalyzer:
    """Infer homogeneous shared-distance couplings from a complete chain map."""

    def __init__(
        self,
        models: Sequence[Any],
        preprocessor: SpectralPreprocessor,
        *,
        n_sites: int,
        parameter_names: Sequence[str],
        target_scaler: MinMaxTargetScaler,
        coupling_unit: str = "meV",
        model_names: Sequence[str] | None = None,
        disagreement_warning_fraction: float = 0.1,
        aggregation: EnsembleAggregation | None = None,
    ) -> None:
        if not models:
            raise ValueError("at least one trained model is required")
        if n_sites < 2:
            raise ValueError("homogeneous inference requires at least two sites")
        if not parameter_names:
            raise ValueError("parameter_names cannot be empty")
        self.models = tuple(models)
        self.preprocessor = preprocessor
        self.n_sites = int(n_sites)
        self.parameter_names = tuple(parameter_names)
        self.target_scaler = target_scaler
        self.coupling_unit = coupling_unit
        self.model_names = tuple(model_names or [f"model_{i}" for i in range(len(models))])
        if len(self.model_names) != len(self.models):
            raise ValueError("model_names must match models")
        self.disagreement_warning_fraction = disagreement_warning_fraction
        self.aggregation = aggregation or EnsembleAggregation()

    def analyze_measurement(
        self, measurement: Measurement, *, source: str = "canonical measurement"
    ) -> ExperimentalGlobalResult:
        bias, spectra = measurement.site_spectra()
        unit = measurement.axis_units.get("bias", "meV")
        if unit != "meV":
            raise ValueError(f"inference requires a canonical meV bias axis, got {unit!r}")
        return self.analyze(spectra, bias, source=source)

    def analyze(
        self,
        spectra: ArrayLike,
        bias_mev: ArrayLike,
        source: str = "in-memory",
    ) -> ExperimentalGlobalResult:
        raw_spectra = np.asarray(spectra, dtype=float)
        raw_bias = np.asarray(bias_mev, dtype=float)
        if raw_spectra.ndim != 2 or raw_spectra.shape[0] != self.n_sites:
            actual = raw_spectra.shape[0] if raw_spectra.ndim >= 1 else 0
            raise ValueError(
                f"homogeneous model requires exactly {self.n_sites} sites; got {actual}"
            )
        processed = self.preprocessor.transform_map(raw_spectra, raw_bias)
        model_input = processed.reshape(1, -1)
        raw_predictions = []
        physical_predictions = []
        expected_shape = (1, len(self.parameter_names))
        for model in self.models:
            try:
                prediction = model.predict(model_input, verbose=0)
            except TypeError:
                prediction = model.predict(model_input)
            prediction = np.asarray(prediction, dtype=np.float32)
            if prediction.shape != expected_shape:
                raise ValueError(
                    f"model returned {prediction.shape}; expected {expected_shape}"
                )
            raw_predictions.append(prediction[0])
            physical_predictions.append(self.target_scaler.inverse_transform(prediction)[0])

        raw_stack = np.stack(raw_predictions)
        physical_stack = np.stack(physical_predictions)
        coupling_mean = self.aggregation.aggregate(physical_stack)
        coupling_std = (
            physical_stack.std(axis=0, ddof=1)
            if physical_stack.shape[0] > 1
            else np.zeros(len(self.parameter_names), dtype=np.float32)
        )
        outside = float(np.mean((raw_stack < 0) | (raw_stack > 1)))
        span = float(np.max(self.target_scaler.maximum - self.target_scaler.minimum))
        max_std = float(np.max(coupling_std))
        warnings = []
        if len(self.models) == 1:
            warnings.append("one model cannot estimate ensemble uncertainty")
        if outside > 0:
            warnings.append(
                f"{outside:.1%} of global predictions lie outside the training range"
            )
        if max_std > self.disagreement_warning_fraction * span:
            warnings.append("ensemble members disagree strongly on at least one parameter")
        diagnostics = ExperimentalDiagnostics(
            status="review" if warnings else "ok",
            warnings=tuple(warnings),
            ensemble_size=len(self.models),
            mean_overlap_disagreement=0.0,
            max_ensemble_std=max_std,
            fraction_predictions_outside_training_range=outside,
            aggregation_method=self.aggregation.method,
        )
        return ExperimentalGlobalResult(
            source=source,
            raw_bias_mev=raw_bias,
            raw_spectra=raw_spectra,
            processed_bias_mev=self.preprocessor.output_bias_mev,
            processed_spectra=processed,
            parameter_names=self.parameter_names,
            coupling_mean=np.asarray(coupling_mean, dtype=np.float32),
            coupling_std=np.asarray(coupling_std, dtype=np.float32),
            per_model_couplings=physical_stack.astype(np.float32),
            coupling_unit=self.coupling_unit,
            diagnostics=diagnostics,
        )


class ExperimentalChainAnalyzer:
    """Load, preprocess, infer, diagnose, plot, and export one chain."""

    def __init__(
        self,
        models: Sequence[Any],
        preprocessor: SpectralPreprocessor,
        output_range: tuple[float, float] | None = None,
        target_scaler: MinMaxTargetScaler | None = None,
        coupling_unit: str = "model units",
        model_names: Sequence[str] | None = None,
        disagreement_warning_fraction: float = 0.1,
        flatten_inputs: bool = False,
        aggregation: EnsembleAggregation | None = None,
    ) -> None:
        if not models:
            raise ValueError("at least one trained model is required")
        if output_range is not None and not output_range[0] < output_range[1]:
            raise ValueError("output_range must satisfy low < high")
        if output_range is not None and target_scaler is not None:
            raise ValueError("pass output_range or target_scaler, not both")
        if disagreement_warning_fraction <= 0:
            raise ValueError("disagreement_warning_fraction must be positive")
        self.models = tuple(models)
        self.preprocessor = preprocessor
        self.output_range = output_range
        self.target_scaler = target_scaler
        self.coupling_unit = coupling_unit
        self.model_names = tuple(model_names or [f"model_{i}" for i in range(len(models))])
        if len(self.model_names) != len(self.models):
            raise ValueError("model_names must match models")
        self.disagreement_warning_fraction = disagreement_warning_fraction
        self.flatten_inputs = flatten_inputs
        self.aggregation = aggregation or EnsembleAggregation()
        if (
            self.aggregation.weights is not None
            and len(self.aggregation.weights) != len(self.models)
        ):
            raise ValueError("aggregation weights must match models")

    @classmethod
    def from_keras_paths(
        cls,
        paths: Sequence[str | Path],
        preprocessor: SpectralPreprocessor,
        **kwargs: Any,
    ) -> "ExperimentalChainAnalyzer":
        try:
            import keras
        except ImportError as exc:  # pragma: no cover
            raise ImportError("loading Keras artifacts requires the ml optional dependency") from exc
        resolved = [Path(path) for path in paths]
        models = [keras.saving.load_model(path) for path in resolved]
        return cls(
            models,
            preprocessor,
            model_names=[path.parent.name for path in resolved],
            **kwargs,
        )

    def analyze_csv(self, path: str | Path) -> ExperimentalChainResult:
        return self.analyze_measurement(load_measurement_csv(path), source=str(path))

    def analyze_measurement(
        self,
        measurement: Measurement,
        *,
        source: str = "canonical measurement",
    ) -> ExperimentalChainResult:
        """Analyze the primary dI/dV channel; auxiliaries remain diagnostic-only."""
        bias, spectra = measurement.site_spectra()
        unit = measurement.axis_units.get("bias", "meV")
        if unit != "meV":
            raise ValueError(f"inference requires a canonical meV bias axis, got {unit!r}")
        return self.analyze(
            spectra,
            bias,
            source=source,
            site_labels=np.asarray(measurement.axes["site"]),
        )

    def analyze(
        self,
        spectra: ArrayLike,
        bias_mev: ArrayLike,
        source: str = "in-memory",
        site_labels: ArrayLike | None = None,
    ) -> ExperimentalChainResult:
        raw_spectra = np.asarray(spectra, dtype=float)
        raw_bias = np.asarray(bias_mev, dtype=float)
        resolved_site_labels = (
            np.arange(raw_spectra.shape[0])
            if site_labels is None
            else np.asarray(site_labels)
        )
        if resolved_site_labels.ndim != 1 or len(resolved_site_labels) != raw_spectra.shape[0]:
            raise ValueError("site_labels must contain one label per spectrum")
        processed = self.preprocessor.transform_map(raw_spectra, raw_bias)
        windows = make_local_windows(processed)
        model_inputs = windows.reshape(windows.shape[0], -1) if self.flatten_inputs else windows

        raw_predictions = []
        physical_predictions = []
        for model in self.models:
            try:
                prediction = model.predict(model_inputs, verbose=0)
            except TypeError:
                prediction = model.predict(model_inputs)
            prediction = np.asarray(prediction, dtype=np.float32)
            if prediction.shape != (windows.shape[0], 2):
                raise ValueError(
                    f"model returned {prediction.shape}; expected {(windows.shape[0], 2)}"
                )
            raw_predictions.append(prediction)
            physical_predictions.append(self._to_output_units(prediction))

        raw_stack = np.stack(raw_predictions)
        window_stack = np.stack(physical_predictions)
        chain_stack = np.stack([reconstruct_chain(item) for item in window_stack])
        coupling_mean = self.aggregation.aggregate(chain_stack)
        coupling_std = (
            chain_stack.std(axis=0, ddof=1)
            if chain_stack.shape[0] > 1
            else np.zeros(chain_stack.shape[1], dtype=np.float32)
        )
        overlap_by_model = np.abs(window_stack[:, :-1, 1] - window_stack[:, 1:, 0])
        overlap = overlap_by_model.mean(axis=0)
        diagnostics = self._diagnose(raw_stack, coupling_std, overlap)
        return ExperimentalChainResult(
            source=source,
            site_labels=resolved_site_labels,
            raw_bias_mev=raw_bias,
            raw_spectra=raw_spectra,
            processed_bias_mev=self.preprocessor.output_bias_mev,
            processed_spectra=processed,
            coupling_mean=coupling_mean.astype(np.float32),
            coupling_std=coupling_std.astype(np.float32),
            per_model_couplings=chain_stack.astype(np.float32),
            per_model_window_predictions=window_stack.astype(np.float32),
            overlap_disagreement=overlap.astype(np.float32),
            coupling_unit=self.coupling_unit,
            diagnostics=diagnostics,
        )

    def _to_output_units(self, prediction: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.target_scaler is not None:
            return self.target_scaler.inverse_transform(prediction)
        if self.output_range is None:
            return prediction
        low, high = self.output_range
        return prediction * (high - low) + low

    def _diagnose(
        self,
        raw_predictions: NDArray[np.float32],
        coupling_std: NDArray[np.float32],
        overlap: NDArray[np.float32],
    ) -> ExperimentalDiagnostics:
        warnings = []
        if len(self.models) == 1:
            warnings.append("one model cannot estimate ensemble uncertainty")
        outside = (
            float(np.mean((raw_predictions < 0) | (raw_predictions > 1)))
            if self.output_range is not None or self.target_scaler is not None
            else 0.0
        )
        if outside > 0:
            warnings.append(f"{outside:.1%} of local predictions lie outside the training range")
        if self.target_scaler is not None:
            span = float(np.max(self.target_scaler.maximum - self.target_scaler.minimum))
        elif self.output_range is not None:
            span = self.output_range[1] - self.output_range[0]
        else:
            span = max(float(np.ptp(raw_predictions)), 1e-12)
        max_std = float(np.max(coupling_std))
        mean_overlap = float(np.mean(overlap)) if overlap.size else 0.0
        threshold = self.disagreement_warning_fraction * span
        if max_std > threshold:
            warnings.append("ensemble members disagree strongly on at least one bond")
        if mean_overlap > threshold:
            warnings.append("overlapping windows give inconsistent interior-bond estimates")
        return ExperimentalDiagnostics(
            status="review" if warnings else "ok",
            warnings=tuple(warnings),
            ensemble_size=len(self.models),
            mean_overlap_disagreement=mean_overlap,
            max_ensemble_std=max_std,
            fraction_predictions_outside_training_range=outside,
            aggregation_method=self.aggregation.method,
        )
