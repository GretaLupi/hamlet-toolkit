import numpy as np
import pytest

from hamlet.inference import LocalChainEstimator
from hamlet.preprocessing import make_local_windows, reconstruct_chain


class WindowIndexModel:
    def predict(self, windows, **kwargs):
        index = np.arange(len(windows), dtype=np.float32)
        return np.column_stack([index, index + 2])


def test_windows_support_arbitrary_chain_length():
    spectra = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
    windows = make_local_windows(spectra)
    assert windows.shape == (4, 15)
    np.testing.assert_array_equal(windows[0], spectra[:3].ravel())
    np.testing.assert_array_equal(windows[-1], spectra[-3:].ravel())


def test_overlapping_bonds_are_averaged():
    predictions = np.array([[1, 3], [2, 4], [3, 5]], dtype=np.float32)
    np.testing.assert_allclose(reconstruct_chain(predictions), [1, 2.5, 3.5, 5])


@pytest.mark.parametrize("n_sites", [3, 4, 6, 10, 16])
def test_estimator_returns_one_coupling_per_bond(n_sites):
    spectra = np.zeros((n_sites, 200), dtype=np.float32)
    result = LocalChainEstimator(WindowIndexModel()).predict(spectra)
    assert result.couplings_mev.shape == (n_sites - 1,)
    assert result.window_predictions.shape == (n_sites - 2, 2)


def test_normalized_predictions_are_converted_to_mev():
    class ConstantModel:
        def predict(self, windows, **kwargs):
            return np.full((len(windows), 2), 0.5)

    result = LocalChainEstimator(ConstantModel(), (30.0, 40.0)).predict(np.zeros((5, 8)))
    np.testing.assert_allclose(result.couplings_mev, 35.0)

