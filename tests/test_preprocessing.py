import numpy as np
import pytest

from hamlet.preprocessing import SpectralPreprocessor


def test_preprocessing_resamples_and_normalizes_each_site():
    bias = np.linspace(0, 50, 501)
    spectra = np.stack([0.2 + bias, 2.0 + 2.0 * bias])
    processor = SpectralPreprocessor(output_points=200)
    output = processor.transform_map(spectra, bias)
    assert output.shape == (2, 200)
    scale_mask = (processor.output_bias_mev >= 40) & (processor.output_bias_mev <= 50)
    np.testing.assert_allclose(np.mean(np.abs(output[:, scale_mask]), axis=1), 1.0, rtol=1e-5)


def test_preprocessing_requires_full_bias_coverage():
    processor = SpectralPreprocessor()
    with pytest.raises(ValueError, match="does not cover"):
        processor.transform_spectrum(np.ones(20), np.linspace(1, 49, 20))


def test_preprocessing_is_invariant_to_ampere_scale_signal_units():
    bias = np.linspace(0.0, 50.0, 501)
    shape = 0.2 + bias + 0.3 * np.sin(bias)
    processor = SpectralPreprocessor(output_points=200)

    ordinary = processor.transform_spectrum(shape, bias)
    lock_in_amperes = processor.transform_spectrum(shape * 1e-12, bias)

    np.testing.assert_allclose(lock_in_amperes, ordinary, rtol=2e-5, atol=2e-6)
    scale_mask = (
        (processor.output_bias_mev >= 40.0)
        & (processor.output_bias_mev <= 50.0)
    )
    assert np.mean(np.abs(lock_in_amperes[scale_mask])) == pytest.approx(1.0)


def test_preprocessing_rejects_an_undefined_flat_signal_scale():
    bias = np.linspace(0.0, 50.0, 501)
    with pytest.raises(ValueError, match="normalization scale is effectively zero"):
        SpectralPreprocessor().transform_spectrum(np.ones_like(bias) * 1e-12, bias)
