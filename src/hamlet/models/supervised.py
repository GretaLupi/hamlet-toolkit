"""Typed configuration and registry for supervised regression models."""

from dataclasses import dataclass
from math import prod
from typing import Any


@dataclass(frozen=True)
class MLPConfig:
    """Hyperparameters for the dense spectroscopy regressor."""

    hidden_units: tuple[int, ...] = (512, 256, 128)
    activation: str = "relu"
    dropout: float = 0.25
    l2: float = 3e-4
    learning_rate: float = 3e-4
    batch_normalization: bool = True
    huber_delta: float = 0.02

    def __post_init__(self) -> None:
        if not self.hidden_units or any(units < 1 for units in self.hidden_units):
            raise ValueError("hidden_units must contain positive layer widths")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.l2 < 0 or self.learning_rate <= 0 or self.huber_delta <= 0:
            raise ValueError("l2 must be non-negative; learning_rate and huber_delta positive")


@dataclass(frozen=True)
class CNNConfig:
    """Hyperparameters for convolution along the spectroscopy bias axis."""

    filters: tuple[int, ...] = (32, 64)
    kernel_size: int = 7
    dense_units: tuple[int, ...] = (128,)
    activation: str = "relu"
    dropout: float = 0.2
    l2: float = 3e-4
    learning_rate: float = 3e-4
    batch_normalization: bool = True
    huber_delta: float = 0.02

    def __post_init__(self) -> None:
        if not self.filters or any(value < 1 for value in self.filters):
            raise ValueError("filters must contain positive channel counts")
        if self.kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        if any(units < 1 for units in self.dense_units):
            raise ValueError("dense_units must contain positive layer widths")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.l2 < 0 or self.learning_rate <= 0 or self.huber_delta <= 0:
            raise ValueError("l2 must be non-negative; learning_rate and huber_delta positive")


def available_supervised_models() -> tuple[str, ...]:
    return ("keras_mlp", "keras_cnn", "ridge", "random_forest")


def create_supervised_model(
    name: str,
    input_dim: int,
    output_dim: int,
    **kwargs: Any,
) -> Any:
    """Construct a compatible regressor without importing unused ML stacks.

    Keras models accept the flattened inputs produced by ``as_supervised``.
    ``keras_cnn`` additionally requires ``spectrum_shape=(sites, bias_points)``
    and reshapes internally before convolving along the bias axis.
    """
    if input_dim < 1 or output_dim < 1:
        raise ValueError("input_dim and output_dim must be positive")
    if name == "ridge":
        try:
            from sklearn.linear_model import Ridge
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ridge requires: pip install 'hamiltonian-learning[ml]'") from exc
        return Ridge(**kwargs)
    if name == "random_forest":
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("random_forest requires: pip install 'hamiltonian-learning[ml]'") from exc
        defaults = {"n_estimators": 200, "n_jobs": -1, "random_state": 42}
        defaults.update(kwargs)
        return RandomForestRegressor(**defaults)
    if name == "keras_mlp":
        config = _extract_config(kwargs, MLPConfig)
        return _build_keras_mlp(input_dim, output_dim, config)
    if name == "keras_cnn":
        spectrum_shape = kwargs.pop("spectrum_shape", None)
        if spectrum_shape is None:
            raise TypeError("keras_cnn requires spectrum_shape=(sites, bias_points)")
        spectrum_shape = tuple(int(value) for value in spectrum_shape)
        if len(spectrum_shape) != 2 or prod(spectrum_shape) != input_dim:
            raise ValueError("spectrum_shape must be (sites, bias_points) and match input_dim")
        config = _extract_config(kwargs, CNNConfig)
        return _build_keras_cnn(input_dim, output_dim, spectrum_shape, config)
    raise ValueError(
        f"unknown model {name!r}; choose one of {available_supervised_models()}"
    )


def _extract_config(options: dict[str, Any], config_type: type[Any]) -> Any:
    explicit = options.pop("config", None)
    if explicit is not None:
        if not isinstance(explicit, config_type):
            raise TypeError(f"config must be {config_type.__name__}")
        if options:
            raise TypeError("pass either config or individual hyperparameters, not both")
        return explicit
    try:
        return config_type(**options)
    except TypeError as exc:
        raise TypeError(f"invalid {config_type.__name__} options: {exc}") from exc


def _keras_modules():
    try:
        import keras
        from keras import layers, regularizers
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Keras models require: pip install 'hamiltonian-learning[ml]'") from exc
    return keras, layers, regularizers


def _build_keras_mlp(input_dim: int, output_dim: int, config: MLPConfig):
    keras, layers, regularizers = _keras_modules()
    model = keras.Sequential(name="spectroscopy_mlp")
    model.add(layers.Input((input_dim,)))
    for index, units in enumerate(config.hidden_units):
        model.add(
            layers.Dense(
                units,
                activation=config.activation,
                kernel_regularizer=regularizers.l2(config.l2),
            )
        )
        if index < len(config.hidden_units) - 1:
            if config.batch_normalization:
                model.add(layers.BatchNormalization())
            if config.dropout:
                model.add(layers.Dropout(config.dropout))
    model.add(layers.Dense(output_dim, activation="linear"))
    return _compile_keras(model, keras, config.learning_rate, config.huber_delta)


def _build_keras_cnn(
    input_dim: int,
    output_dim: int,
    spectrum_shape: tuple[int, int],
    config: CNNConfig,
):
    keras, layers, regularizers = _keras_modules()
    inputs = layers.Input((input_dim,), name="flattened_spectra")
    x = layers.Reshape(spectrum_shape)(inputs)
    x = layers.Permute((2, 1), name="bias_by_site")(x)
    for filters in config.filters:
        x = layers.Conv1D(
            filters,
            config.kernel_size,
            padding="same",
            activation=config.activation,
            kernel_regularizer=regularizers.l2(config.l2),
        )(x)
        if config.batch_normalization:
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    for units in config.dense_units:
        x = layers.Dense(
            units,
            activation=config.activation,
            kernel_regularizer=regularizers.l2(config.l2),
        )(x)
        if config.dropout:
            x = layers.Dropout(config.dropout)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    model = keras.Model(inputs, outputs, name="spectroscopy_cnn")
    return _compile_keras(model, keras, config.learning_rate, config.huber_delta)


def _compile_keras(model: Any, keras: Any, learning_rate: float, huber_delta: float):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(delta=huber_delta),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )
    return model

