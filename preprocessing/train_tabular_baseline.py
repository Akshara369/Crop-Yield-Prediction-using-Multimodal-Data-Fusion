import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "models"

INPUT_PATH = DATASETS_DIR / "modeling_dataset.csv"
MODEL_PATH = MODELS_DIR / "tabular_random_forest.joblib"
METRICS_PATH = MODELS_DIR / "tabular_baseline_metrics.json"
PREDICTIONS_PATH = MODELS_DIR / "tabular_baseline_test_predictions.csv"

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
                    n_jobs=-1,
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
    return {
        "rmse": float(root_mean_squared_error(y, predictions)),
        "mae": float(mean_absolute_error(y, predictions)),
        "r2": float(r2_score(y, predictions)),
    }


def split_by_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["Year"] <= TRAIN_END_YEAR].copy()
    validation = df[
        (df["Year"] > TRAIN_END_YEAR) & (df["Year"] <= VALIDATION_END_YEAR)
    ].copy()
    test = df[df["Year"] > VALIDATION_END_YEAR].copy()
    return train, validation, test


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

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

    dummy = make_dummy_baseline()
    dummy.fit(x_train, y_train)

    random_forest = make_random_forest(make_preprocessor(numeric_columns, categorical_columns))
    random_forest.fit(x_train, y_train)

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
        "models": {
            "dummy_mean": {
                "validation": evaluate(dummy, x_validation, y_validation),
                "test": evaluate(dummy, x_test, y_test),
            },
            "random_forest": {
                "validation": evaluate(random_forest, x_validation, y_validation),
                "test": evaluate(random_forest, x_test, y_test),
            },
        },
        "notes": [
            "Production is excluded to avoid target leakage.",
            "The saved model is trained on rows through the validation period for stronger future predictions.",
        ],
    }

    final_train = df[df["Year"] <= VALIDATION_END_YEAR].copy()
    final_model = make_random_forest(make_preprocessor(numeric_columns, categorical_columns))
    final_model.fit(final_train[feature_columns], final_train[TARGET_COLUMN])
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "target_column": TARGET_COLUMN,
        },
        MODEL_PATH,
    )

    test_predictions = test[
        ["State", "Crop", "Year", "Season", TARGET_COLUMN]
    ].copy()
    test_predictions["Predicted_Yield"] = random_forest.predict(x_test)
    test_predictions["Absolute_Error"] = np.abs(
        test_predictions[TARGET_COLUMN] - test_predictions["Predicted_Yield"]
    )
    test_predictions.to_csv(PREDICTIONS_PATH, index=False)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    rf_test = metrics["models"]["random_forest"]["test"]
    print(f"Wrote {MODEL_PATH.relative_to(ROOT)}")
    print(f"Wrote {METRICS_PATH.relative_to(ROOT)}")
    print(f"Wrote {PREDICTIONS_PATH.relative_to(ROOT)}")
    print(
        "Random Forest test metrics: "
        f"RMSE={rf_test['rmse']:.4f}, MAE={rf_test['mae']:.4f}, R2={rf_test['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
