import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = MODELS_DIR / "reports"

INPUT_PATH = DATASETS_DIR / "modeling_dataset.csv"
MODEL_PATH = MODELS_DIR / "tabular_random_forest.joblib"
BEST_MODEL_PATH = MODELS_DIR / "tabular_best_model.joblib"
METRICS_PATH = MODELS_DIR / "tabular_baseline_metrics.json"
MODEL_COMPARISON_PATH = MODELS_DIR / "tabular_model_comparison.csv"
CROP_METRICS_PATH = MODELS_DIR / "tabular_crop_metrics.csv"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "tabular_feature_importance.csv"
PREDICTIONS_PATH = MODELS_DIR / "tabular_baseline_test_predictions.csv"
ACTUAL_VS_PREDICTED_PLOT_PATH = REPORTS_DIR / "actual_vs_predicted.png"
ERROR_DISTRIBUTION_PLOT_PATH = REPORTS_DIR / "error_distribution.png"
FEATURE_IMPORTANCE_PLOT_PATH = REPORTS_DIR / "feature_importance.png"

TARGET_COLUMN = "Yield"
TRAIN_END_YEAR = 2016
VALIDATION_END_YEAR = 2018
RANDOM_STATE = 42
MAX_REASONABLE_YIELD = 15.0

# Production is excluded because it is usually used to calculate yield
# and would leak target information into the model.
EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    "Production",
    "Start_Date",
    "End_Date",
    "RGB_Image",
    "NDVI_Image",
    "Satellite_Source",
}


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[column for column in EXCLUDED_COLUMNS if column in df])
    categorical_columns = [
        column
        for column in ["State", "Crop", "Season"]
        if column in feature_df.columns
    ]
    numeric_columns = [
        column
        for column in feature_df.columns
        if column not in categorical_columns and pd.api.types.is_numeric_dtype(feature_df[column])
    ]
    return numeric_columns, categorical_columns


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    area = enriched["Area"].replace(0, np.nan)
    enriched["Fertilizer_per_Area"] = enriched["Fertilizer"] / area
    enriched["Pesticide_per_Area"] = enriched["Pesticide"] / area
    return enriched


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def make_random_forest(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )


def make_gradient_boosting(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.04,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_hist_gradient_boosting(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.04,
                    max_iter=300,
                    l2_regularization=0.05,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_dummy_baseline() -> Pipeline:
    return Pipeline(
        steps=[
            ("model", DummyRegressor(strategy="mean")),
        ]
    )


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predictions = model.predict(x)
    return evaluate_predictions(y, predictions)


def evaluate_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, predictions)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)) if len(y_true) > 1 else float("nan"),
    }


def split_by_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["Year"] <= TRAIN_END_YEAR].copy()
    validation = df[
        (df["Year"] > TRAIN_END_YEAR) & (df["Year"] <= VALIDATION_END_YEAR)
    ].copy()
    test = df[df["Year"] > VALIDATION_END_YEAR].copy()
    return train, validation, test


def build_model_candidates(numeric_columns: list[str], categorical_columns: list[str]) -> dict[str, Pipeline]:
    return {
        "dummy_mean": make_dummy_baseline(),
        "random_forest": make_random_forest(make_preprocessor(numeric_columns, categorical_columns)),
        "gradient_boosting": make_gradient_boosting(make_preprocessor(numeric_columns, categorical_columns)),
        "hist_gradient_boosting": make_hist_gradient_boosting(make_preprocessor(numeric_columns, categorical_columns)),
    }


def make_predictions_frame(source: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    predictions_df = source[["State", "Crop", "Year", "Season", TARGET_COLUMN]].copy()
    predictions_df["Predicted_Yield"] = predictions
    predictions_df["Error"] = predictions_df[TARGET_COLUMN] - predictions_df["Predicted_Yield"]
    predictions_df["Absolute_Error"] = predictions_df["Error"].abs()
    return predictions_df


def build_crop_metrics(predictions_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for crop, crop_df in predictions_df.groupby("Crop"):
        metric_values = evaluate_predictions(crop_df[TARGET_COLUMN], crop_df["Predicted_Yield"].to_numpy())
        rows.append(
            {
                "model": model_name,
                "crop": crop,
                "rows": len(crop_df),
                **metric_values,
            }
        )
    return pd.DataFrame(rows)


def get_feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out().tolist()


def build_feature_importance(model: Pipeline) -> pd.DataFrame:
    regressor = model.named_steps["model"]
    if not hasattr(regressor, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    feature_names = get_feature_names(model)
    cleaned_names = [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in feature_names
    ]
    return (
        pd.DataFrame(
            {
                "feature": cleaned_names,
                "importance": regressor.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_actual_vs_predicted(predictions_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for crop, crop_df in predictions_df.groupby("Crop"):
        ax.scatter(crop_df[TARGET_COLUMN], crop_df["Predicted_Yield"], label=crop, alpha=0.75)

    min_value = min(predictions_df[TARGET_COLUMN].min(), predictions_df["Predicted_Yield"].min())
    max_value = max(predictions_df[TARGET_COLUMN].max(), predictions_df["Predicted_Yield"].max())
    ax.plot([min_value, max_value], [min_value, max_value], color="#334155", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual yield")
    ax.set_ylabel("Predicted yield")
    ax.set_title("Actual vs Predicted Yield")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_error_distribution(predictions_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(predictions_df["Error"], bins=24, color="#2563eb", alpha=0.78, edgecolor="white")
    ax.axvline(0, color="#334155", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual - predicted yield")
    ax.set_ylabel("Rows")
    ax.set_title("Prediction Error Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(feature_importance: pd.DataFrame, output_path: Path) -> None:
    top_features = feature_importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top_features["feature"], top_features["importance"], color="#16a34a")
    ax.set_xlabel("Importance")
    ax.set_title("Top Random Forest Feature Importances")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=[TARGET_COLUMN, "Year"]).copy()
    df = add_derived_features(df)
    original_rows = len(df)
    df = df[(df[TARGET_COLUMN] >= 0) & (df[TARGET_COLUMN] <= MAX_REASONABLE_YIELD)].copy()

    train, validation, test = split_by_year(df)
    numeric_columns, categorical_columns = get_feature_columns(df)
    feature_columns = numeric_columns + categorical_columns

    x_train = train[feature_columns]
    y_train = train[TARGET_COLUMN]
    x_validation = validation[feature_columns]
    y_validation = validation[TARGET_COLUMN]
    x_test = test[feature_columns]
    y_test = test[TARGET_COLUMN]

    models = build_model_candidates(numeric_columns, categorical_columns)
    for model in models.values():
        model.fit(x_train, y_train)

    model_comparison_rows = []
    model_metrics = {}
    for model_name, model in models.items():
        validation_metrics = evaluate(model, x_validation, y_validation)
        test_metrics = evaluate(model, x_test, y_test)
        model_metrics[model_name] = {
            "validation": validation_metrics,
            "test": test_metrics,
        }
        for split_name, split_metrics in [("validation", validation_metrics), ("test", test_metrics)]:
            model_comparison_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    **split_metrics,
                }
            )

    model_comparison = pd.DataFrame(model_comparison_rows)
    trainable_models = [name for name in models if name != "dummy_mean"]
    best_model_name = min(
        trainable_models,
        key=lambda name: model_metrics[name]["validation"]["rmse"],
    )
    best_model = models[best_model_name]
    best_test_predictions = make_predictions_frame(test, best_model.predict(x_test))
    crop_metrics = build_crop_metrics(best_test_predictions, best_model_name)

    random_forest = models["random_forest"]
    feature_importance = build_feature_importance(random_forest)

    metrics = {
        "dataset": {
            "rows_before_outlier_filter": int(original_rows),
            "rows": int(len(df)),
            "removed_outlier_rows": int(original_rows - len(df)),
            "max_reasonable_yield": MAX_REASONABLE_YIELD,
            "features": int(len(feature_columns)),
            "numeric_features": numeric_columns,
            "categorical_features": categorical_columns,
            "target": TARGET_COLUMN,
        },
        "split": {
            "train_years": f"{int(train['Year'].min())}-{TRAIN_END_YEAR}",
            "validation_years": f"{TRAIN_END_YEAR + 1}-{VALIDATION_END_YEAR}",
            "test_years": f"{VALIDATION_END_YEAR + 1}-{int(test['Year'].max())}",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
        },
        "models": model_metrics,
        "best_model": best_model_name,
        "crop_metrics": crop_metrics.to_dict(orient="records"),
        "notes": [
            "Production is excluded to avoid target leakage.",
            "Seven implausible target outliers above 15 tonnes/ha are filtered before training.",
            "The saved model is trained on rows through the validation period for stronger future predictions.",
        ],
    }

    final_train = df[df["Year"] <= VALIDATION_END_YEAR].copy()
    final_model_builders = build_model_candidates(numeric_columns, categorical_columns)
    final_model = final_model_builders[best_model_name]
    final_model.fit(final_train[feature_columns], final_train[TARGET_COLUMN])
    artifact = {
        "model": final_model,
        "model_name": best_model_name,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target_column": TARGET_COLUMN,
        "train_end_year": TRAIN_END_YEAR,
        "validation_end_year": VALIDATION_END_YEAR,
        "max_reasonable_yield": MAX_REASONABLE_YIELD,
    }
    joblib.dump(artifact, BEST_MODEL_PATH)

    final_random_forest = make_random_forest(make_preprocessor(numeric_columns, categorical_columns))
    final_random_forest.fit(final_train[feature_columns], final_train[TARGET_COLUMN])
    joblib.dump({**artifact, "model": final_random_forest, "model_name": "random_forest"}, MODEL_PATH)

    model_comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    crop_metrics.to_csv(CROP_METRICS_PATH, index=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    best_test_predictions.to_csv(PREDICTIONS_PATH, index=False)

    plot_actual_vs_predicted(best_test_predictions, ACTUAL_VS_PREDICTED_PLOT_PATH)
    plot_error_distribution(best_test_predictions, ERROR_DISTRIBUTION_PLOT_PATH)
    if not feature_importance.empty:
        plot_feature_importance(feature_importance, FEATURE_IMPORTANCE_PLOT_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    best_test = metrics["models"][best_model_name]["test"]
    print(f"Wrote {BEST_MODEL_PATH.relative_to(ROOT)}")
    print(f"Wrote {MODEL_PATH.relative_to(ROOT)}")
    print(f"Wrote {METRICS_PATH.relative_to(ROOT)}")
    print(f"Wrote {MODEL_COMPARISON_PATH.relative_to(ROOT)}")
    print(f"Wrote {CROP_METRICS_PATH.relative_to(ROOT)}")
    print(f"Wrote {FEATURE_IMPORTANCE_PATH.relative_to(ROOT)}")
    print(f"Wrote {PREDICTIONS_PATH.relative_to(ROOT)}")
    print(
        f"Best model: {best_model_name} | "
        f"test RMSE={best_test['rmse']:.4f}, MAE={best_test['mae']:.4f}, R2={best_test['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
