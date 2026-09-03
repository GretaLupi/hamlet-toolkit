# Model selection and MLP tuning

## Which model should a user start with?

Use the same grouped train/validation/test split for every candidate and compare
physical-unit validation metrics after converting predictions back to meV.

| Model | Good first use | Main limitation |
|---|---|---|
| `ridge` | Sanity check and linear baseline | Cannot learn nonlinear spectral relationships |
| `random_forest` | Small or medium tabular datasets | Large models; does not exploit the ordered bias axis |
| `keras_mlp` | General nonlinear baseline matching the reference project | Parameter-heavy and ignores explicit spectral locality |
| `keras_cnn` | Peaks and local patterns along the bias axis | More architectural choices and training time |

A more complex model should only replace a simpler one when it improves held-out
validation performance consistently across several dataset seeds and noise
conditions.

## Configure an MLP

```python
from hamlet.models import MLPConfig, create_supervised_model

config = MLPConfig(
    hidden_units=(512, 256, 128),
    activation="relu",
    dropout=0.25,
    l2=3e-4,
    learning_rate=3e-4,
    batch_normalization=True,
    huber_delta=0.02,
)

model = create_supervised_model(
    "keras_mlp",
    input_dim=training.inputs.shape[1],
    output_dim=training.targets.shape[1],
    config=config,
)
```

`huber_delta=0.02` assumes targets have been scaled approximately to `[0, 1]`.
If training directly in meV, the loss scale must be reconsidered.

## Configure a CNN

The package still accepts flattened supervised inputs. `spectrum_shape` tells
the model how to recover the site and bias axes.

```python
from hamlet.models import CNNConfig, create_supervised_model

config = CNNConfig(
    filters=(32, 64),
    kernel_size=7,
    dense_units=(128,),
    dropout=0.2,
    l2=3e-4,
    learning_rate=3e-4,
)

# Local three-site windows with 200 bias points per site.
model = create_supervised_model(
    "keras_cnn",
    input_dim=training.inputs.shape[1],
    output_dim=2,
    spectrum_shape=(3, 200),
    config=config,
)

# For a global length-10 model, use spectrum_shape=(10, 200).
```

The convolution runs along the bias axis; sites are treated as measurement
channels. Global average pooling makes the CNN tolerant to a different number
of bias points, but the current flattened input layer still fixes the declared
input shape when the model is built.

## Recommended MLP tuning order

Do not tune every parameter simultaneously. A stable sequence is:

1. Establish `ridge` and default-MLP baselines.
2. Tune learning rate: `1e-4`, `3e-4`, `1e-3`.
3. Tune capacity: `(256, 128)`, `(512, 256, 128)`, `(512, 512, 256)`.
4. Tune regularization: dropout `0.0–0.4` and L2 `1e-5–1e-3`.
5. Only then compare activation, batch normalization, or loss settings.

Typical interpretation:

- high train and validation error: increase capacity or improve input features;
- low train error but high validation error: increase regularization or dataset diversity;
- unstable validation loss: lower the learning rate;
- good synthetic performance but poor experiment performance: improve the noise and
  broadening model rather than merely enlarging the network.

## Small validation search

```python
import keras
from hamlet.benchmarks import mae
from hamlet.models import MLPConfig, create_supervised_model

candidates = [
    MLPConfig((256, 128), learning_rate=1e-3, dropout=0.1),
    MLPConfig((512, 256, 128), learning_rate=3e-4, dropout=0.25),
    MLPConfig((512, 512, 256), learning_rate=1e-4, dropout=0.35),
]

results = []
for config in candidates:
    keras.utils.set_random_seed(42)
    model = create_supervised_model(
        "keras_mlp",
        split.train.inputs.shape[1],
        split.train.targets.shape[1],
        config=config,
    )
    model.fit(
        split.train.inputs,
        scaler.transform(split.train.targets),
        validation_data=(
            split.validation.inputs,
            scaler.transform(split.validation.targets),
        ),
        epochs=100,
        batch_size=128,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=12, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
            ),
        ],
        verbose=0,
    )
    validation_prediction = scaler.inverse_transform(
        model.predict(split.validation.inputs, verbose=0)
    )
    results.append((mae(validation_prediction, split.validation.targets), config, model))

best_validation_mae, best_config, best_model = min(results, key=lambda item: item[0])
```

After choosing the configuration, evaluate it once on the untouched test set.
For research-quality comparison, repeat training with several random model seeds
and report the mean and spread rather than selecting one unusually good run.

