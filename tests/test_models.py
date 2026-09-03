import numpy as np
import pytest

from hamlet.models import (
    CNNConfig,
    MLPConfig,
    available_supervised_models,
    create_supervised_model,
)


def test_ridge_model_can_fit_multioutput_supervised_data():
    pytest.importorskip("sklearn")
    inputs = np.arange(30, dtype=float).reshape(10, 3)
    targets = np.column_stack([inputs[:, 0], inputs[:, 1]])
    model = create_supervised_model("ridge", 3, 2, alpha=1e-8)
    model.fit(inputs, targets)
    assert model.predict(inputs).shape == targets.shape


def test_unknown_model_lists_choices():
    with pytest.raises(ValueError, match="choose one of"):
        create_supervised_model("quantum_oracle", 3, 2)
    assert "keras_mlp" in available_supervised_models()
    assert "keras_cnn" in available_supervised_models()


def test_mlp_configuration_validation():
    config = MLPConfig(hidden_units=(64, 32), dropout=0.1, learning_rate=1e-3)
    assert config.hidden_units == (64, 32)
    with pytest.raises(ValueError, match="dropout"):
        MLPConfig(dropout=1.0)


def test_cnn_accepts_flattened_local_spectra():
    pytest.importorskip("tensorflow")
    config = CNNConfig(filters=(4,), kernel_size=3, dense_units=(8,), dropout=0.0)
    model = create_supervised_model(
        "keras_cnn",
        input_dim=3 * 40,
        output_dim=2,
        spectrum_shape=(3, 40),
        config=config,
    )
    assert model.input_shape == (None, 120)
    assert model.output_shape == (None, 2)
    assert model(np.zeros((2, 120), dtype=np.float32)).shape == (2, 2)


def test_cnn_requires_matching_spectrum_shape():
    with pytest.raises(ValueError, match="match input_dim"):
        create_supervised_model(
            "keras_cnn", 120, 2, spectrum_shape=(3, 39), config=CNNConfig()
        )
