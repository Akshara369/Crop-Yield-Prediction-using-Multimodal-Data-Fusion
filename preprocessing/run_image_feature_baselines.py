"""Compare RGB, rendered-NDVI, and combined image-summary baselines."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
MODELS = ROOT / "models"
TRAIN_PATH = DATASETS / "cnn_train.csv"
VALIDATION_PATH = DATASETS / "cnn_validation.csv"
TEST_PATH = DATASETS / "cnn_test.csv"
CNN_METRICS_PATH = MODELS / "cnn_image_baseline_metrics.json"
METRICS_PATH = MODELS / "image_feature_baseline_metrics.json"
COMPARISON_PATH = MODELS / "image_feature_baseline_comparison.csv"
SELECTION_PATH = MODELS / "image_feature_baseline_model_selection.csv"
PREDICTIONS_PATH = MODELS / "image_feature_baseline_test_predictions.csv"
MODEL_PATH = MODELS / "image_feature_baseline_best.joblib"
IMAGE_SIZE = (96, 96)
GRID_SIZE = 4
HISTOGRAM_BINS = 8
RANDOM_STATE = 42


def load_image(path: str) -> np.ndarray:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Image path escapes repository root: {path}")
    with Image.open(resolved) as image:
        image = image.convert("RGB").resize(
            IMAGE_SIZE, Image.Resampling.BILINEAR
        )
        return np.asarray(image, dtype=np.float32) / 255.0


def summarize_image(image: np.ndarray) -> np.ndarray:
    features: list[float] = []
    for channel in range(image.shape[-1]):
        values = image[:, :, channel]
        features.extend(
            (
                float(values.mean()),
                float(values.std()),
                float(np.quantile(values, 0.1)),
                float(np.quantile(values, 0.9)),
            )
        )
        histogram, _ = np.histogram(
            values, bins=HISTOGRAM_BINS, range=(0.0, 1.0), density=False
        )
        features.extend((histogram / values.size).tolist())
        for row in np.array_split(values, GRID_SIZE, axis=0):
            for cell in np.array_split(row, GRID_SIZE, axis=1):
                features.append(float(cell.mean()))
    return np.asarray(features, dtype=np.float32)


def extract_features(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    rgb_rows: list[np.ndarray] = []
    ndvi_rows: list[np.ndarray] = []
    for row in frame.itertuples(index=False):
        rgb_rows.append(summarize_image(load_image(row.rgb)))
        ndvi_rows.append(summarize_image(load_image(row.ndvi)))
    rgb = np.stack(rgb_rows)
    ndvi = np.stack(ndvi_rows)
    return {
        "rgb_only": rgb,
        "ndvi_rendered_png_only": ndvi,
        "rgb_plus_ndvi_rendered_png": np.concatenate((rgb, ndvi), axis=1),
    }


def build_candidates() -> dict[str, object]:
    return {
        "mean_dummy": DummyRegressor(strategy="mean"),
        "ridge_alpha_1": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)
        ),
        "ridge_alpha_10": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)
        ),
        "ridge_alpha_100": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0)
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=1,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
    }


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
    }


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    test = pd.read_csv(TEST_PATH)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Train, validation, and test CSVs must all be non-empty.")

    train_features = extract_features(train)
    validation_features = extract_features(validation)
    test_features = extract_features(test)
    train_target = train["yield"].to_numpy(dtype=np.float64)
    validation_target = validation["yield"].to_numpy(dtype=np.float64)
    test_target = test["yield"].to_numpy(dtype=np.float64)

    rows: list[dict[str, object]] = []
    winner_by_view: dict[str, dict[str, object]] = {}
    candidates_by_view: dict[str, dict[str, object]] = {}
    for view in train_features:
        candidates_by_view[view] = build_candidates()
        for name, estimator in candidates_by_view[view].items():
            estimator.fit(train_features[view], train_target)
            validation_predictions = estimator.predict(validation_features[view])
            validation_scores = regression_metrics(
                validation_target, validation_predictions
            )
            rows.append(
                {
                    "input_view": view,
                    "model": name,
                    "validation_rmse": validation_scores["rmse"],
                    "validation_mae": validation_scores["mae"],
                    "validation_r2": validation_scores["r2"],
                }
            )

        winner = min(
            [row for row in rows if row["input_view"] == view],
            key=lambda row: (float(row["validation_mae"]), str(row["model"])),
        )
        winner_by_view[view] = winner
        for row in rows:
            if row["input_view"] == view:
                row["selected_by_validation"] = row is winner
        winner_estimator = candidates_by_view[view][str(winner["model"])]
        winner_predictions = winner_estimator.predict(test_features[view])
        test_scores = regression_metrics(test_target, winner_predictions)
        winner["test_rmse"] = test_scores["rmse"]
        winner["test_mae"] = test_scores["mae"]
        winner["test_r2"] = test_scores["r2"]
        winner["test_sample_count"] = len(test)
        winner["test_predictions"] = winner_predictions

    pd.DataFrame(rows).drop(columns=["test_predictions"], errors="ignore").to_csv(
        SELECTION_PATH, index=False
    )

    comparison_rows = []
    prediction_frame = test[
        ["feature_key", "state", "crop", "year", "season", "yield"]
    ].copy()
    for view, winner in winner_by_view.items():
        comparison_row = {
            key: value
            for key, value in winner.items()
            if key != "test_predictions"
        }
        comparison_row["selected_by"] = "validation_mae"
        comparison_rows.append(comparison_row)
        column = f"{view}_predicted_yield"
        prediction_frame[column] = winner["test_predictions"]
        joblib.dump(
            candidates_by_view[view][str(winner["model"])],
            MODELS / f"image_feature_baseline_{view}.joblib",
        )

    tabular_predictions = test["tabular_predicted_yield"].to_numpy(dtype=np.float64)
    tabular_scores = regression_metrics(test_target, tabular_predictions)
    comparison_rows.append(
        {
            "input_view": "tabular_reference",
            "model": "gradient_boosting_existing",
            "validation_rmse": np.nan,
            "validation_mae": np.nan,
            "validation_r2": np.nan,
            "test_rmse": tabular_scores["rmse"],
            "test_mae": tabular_scores["mae"],
            "test_r2": tabular_scores["r2"],
            "test_sample_count": len(test),
            "selected_by": "existing_tabular_test_predictions",
        }
    )
    if CNN_METRICS_PATH.is_file():
        cnn_metrics = json.loads(CNN_METRICS_PATH.read_text(encoding="utf-8"))
        cnn_test = cnn_metrics["test"]["cnn_image_only"]
        comparison_rows.append(
            {
                "input_view": "rgb_plus_ndvi_cnn",
                "model": "small_cnn_from_scratch",
                "validation_rmse": np.nan,
                "validation_mae": np.nan,
                "validation_r2": np.nan,
                "test_rmse": cnn_test["rmse"],
                "test_mae": cnn_test["mae"],
                "test_r2": cnn_test["r2"],
                "test_sample_count": cnn_metrics["test"]["sample_count"],
                "selected_by": "existing_cnn_run",
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(COMPARISON_PATH, index=False)
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)

    best_image_view = min(
        winner_by_view,
        key=lambda view: (
            float(winner_by_view[view]["validation_mae"]),
            view,
        ),
    )
    best_row = winner_by_view[best_image_view]
    best_estimator = candidates_by_view[best_image_view][str(best_row["model"])]
    joblib.dump(
        {
            "input_view": best_image_view,
            "model_name": best_row["model"],
            "estimator": best_estimator,
            "feature_definition": {
                "image_size": list(IMAGE_SIZE),
                "per_channel": [
                    "mean",
                    "standard_deviation",
                    "10th_percentile",
                    "90th_percentile",
                    f"{HISTOGRAM_BINS}-bin_histogram",
                    f"{GRID_SIZE}x{GRID_SIZE}_spatial_means",
                ],
                "ndvi_note": (
                    "NDVI values are summarized from the rendered PNG RGB colors; "
                    "the source is not decoded as raw NDVI."
                ),
            },
        },
        MODEL_PATH,
    )
    metrics = {
        "experiment": "image_summary_features_single_modalities",
        "sample_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "split_years": {
            "train": "through 2016",
            "validation": "2017-2018",
            "test": sorted(test["year"].astype(int).unique().tolist()),
        },
        "selection": (
            "Each image view selects its estimator using validation MAE; test "
            "data is not used for model or hyperparameter selection."
        ),
        "image_features": {
            "resize": list(IMAGE_SIZE),
            "per_channel": [
                "mean",
                "standard_deviation",
                "10th_percentile",
                "90th_percentile",
                f"{HISTOGRAM_BINS}-bin_histogram",
                f"{GRID_SIZE}x{GRID_SIZE}_spatial_means",
            ],
            "ndvi_note": (
                "NDVI values are summarized from the rendered PNG RGB colors; "
                "the source is not decoded as raw NDVI."
            ),
        },
        "selected_per_view": [
            {
                key: value
                for key, value in winner.items()
                if key != "test_predictions"
            }
            for winner in winner_by_view.values()
        ],
        "best_image_view_by_validation_mae": best_image_view,
        "best_image_model": str(best_row["model"]),
        "comparison_csv": str(COMPARISON_PATH.relative_to(ROOT)),
        "candidate_validation_csv": str(SELECTION_PATH.relative_to(ROOT)),
        "test_predictions_csv": str(PREDICTIONS_PATH.relative_to(ROOT)),
        "limitations": [
            "The 2019 test set contains only 13 samples; test metrics are exploratory.",
            "NDVI PNGs are treated as rendered RGB images, not calibrated NDVI values.",
        ],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(comparison.to_string(index=False))
    print(f"Best image view by validation MAE: {best_image_view} / {best_row['model']}")
    print(f"Saved metrics: {METRICS_PATH.relative_to(ROOT)}")
    print(f"Saved comparison: {COMPARISON_PATH.relative_to(ROOT)}")
    print(f"Saved validation model selection: {SELECTION_PATH.relative_to(ROOT)}")
    print(f"Saved test predictions: {PREDICTIONS_PATH.relative_to(ROOT)}")
    print(f"Saved best image model: {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
