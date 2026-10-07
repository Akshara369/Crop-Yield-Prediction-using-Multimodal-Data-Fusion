"""Train and evaluate a small image-only RGB+NDVI CNN regression baseline."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
MODELS = ROOT / "models"
TRAIN_PATH = DATASETS / "cnn_train.csv"
VALIDATION_PATH = DATASETS / "cnn_validation.csv"
TEST_PATH = DATASETS / "cnn_test.csv"
MODEL_PATH = MODELS / "cnn_image_baseline.keras"
METRICS_PATH = MODELS / "cnn_image_baseline_metrics.json"
PREDICTIONS_PATH = MODELS / "cnn_image_baseline_test_predictions.csv"
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 16
MAX_EPOCHS = 50
RANDOM_SEED = 42


def load_image(path: str) -> np.ndarray:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Image path escapes repository root: {path}")
    with Image.open(resolved) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def make_dataset(
    frame: pd.DataFrame,
    target_mean: float,
    target_std: float,
    shuffle: bool,
) -> tf.data.Dataset:
    images = np.stack(
        [
            np.concatenate((load_image(row.rgb), load_image(row.ndvi)), axis=-1)
            for row in frame.itertuples(index=False)
        ]
    )
    targets = (
        frame["yield"].to_numpy(dtype=np.float32) - target_mean
    ) / target_std
    dataset = tf.data.Dataset.from_tensor_slices((images, targets))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(frame), seed=RANDOM_SEED, reshuffle_each_iteration=True
        )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 6))
    x = inputs
    for filters in (16, 32, 64):
        x = tf.keras.layers.Conv2D(
            filters, kernel_size=3, padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
    }


def predict(
    model: tf.keras.Model,
    frame: pd.DataFrame,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    scaled = model.predict(
        make_dataset(frame, target_mean, target_std, shuffle=False), verbose=0
    ).reshape(-1)
    return scaled * target_std + target_mean


def main() -> None:
    if tf is None:
        print(
            "TensorFlow is not installed, so the CNN baseline cannot be trained in "
            "this environment."
        )
        print(
            "Running the lightweight image-feature baseline instead. To train the "
            "CNN later, install TensorFlow with: pip install tensorflow"
        )
        from run_image_feature_baselines import main as run_feature_baseline

        run_feature_baseline()
        return

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    MODELS.mkdir(exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    test = pd.read_csv(TEST_PATH)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "Train, validation, and test CSVs must all have samples; "
            f"got {len(train)}, {len(validation)}, {len(test)}."
        )

    target_mean = float(train["yield"].mean())
    target_std = float(train["yield"].std(ddof=0))
    if not np.isfinite(target_std) or target_std == 0:
        raise ValueError("Training yield has no finite variation.")

    model = build_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    history = model.fit(
        make_dataset(train, target_mean, target_std, shuffle=True).repeat(),
        validation_data=make_dataset(
            validation, target_mean, target_std, shuffle=False
        ),
        epochs=MAX_EPOCHS,
        steps_per_epoch=math.ceil(len(train) / BATCH_SIZE),
        callbacks=[early_stopping],
        shuffle=False,
        verbose=2,
    )

    test_predictions = predict(model, test, target_mean, target_std)
    actual = test["yield"].to_numpy(dtype=np.float64)
    cnn_metrics = regression_metrics(actual, test_predictions)
    tabular_metrics = regression_metrics(
        actual, test["tabular_predicted_yield"].to_numpy(dtype=np.float64)
    )

    predictions = test[
        ["feature_key", "state", "crop", "year", "season", "yield"]
    ].copy()
    predictions["cnn_predicted_yield"] = test_predictions
    predictions["tabular_predicted_yield"] = test["tabular_predicted_yield"]
    predictions["cnn_absolute_error"] = np.abs(actual - test_predictions)
    predictions["tabular_absolute_error"] = np.abs(
        actual - predictions["tabular_predicted_yield"].to_numpy(dtype=np.float64)
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    model.save(MODEL_PATH)

    metrics = {
        "model": "small_cnn_rgb_ndvi_six_channel",
        "input": {
            "modalities": ["rgb", "ndvi"],
            "channels": 6,
            "image_size": list(IMAGE_SIZE),
            "yield_scaled_using_training_mean_and_std": {
                "mean": target_mean,
                "std": target_std,
            },
        },
        "split_years": {
            "train": f"<=2016 ({len(train)} samples)",
            "validation": f"2017-2018 ({len(validation)} samples)",
            "test": f">=2019 ({len(test)} samples)",
        },
        "epochs_trained": len(history.history["loss"]),
        "best_validation_loss_scaled": float(min(history.history["val_loss"])),
        "test": {
            "cnn_image_only": cnn_metrics,
            "tabular_gradient_boosting_same_samples": tabular_metrics,
            "sample_count": int(len(test)),
        },
        "limitations": [
            "The test set has few eligible image pairs; test metrics are exploratory.",
            "No eligible 2020 image pairs are present, so this is a 2019-only test.",
            "The CNN is trained from scratch because no pretrained image weights are bundled.",
        ],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"CNN test metrics (n={len(test)}): {cnn_metrics}")
    print(f"Tabular same-sample test metrics (n={len(test)}): {tabular_metrics}")
    print(f"Saved model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved metrics: {METRICS_PATH.relative_to(ROOT)}")
    print(f"Saved predictions: {PREDICTIONS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
