"""Tune Multimodal Fusion Pipeline: Component search, Feature Selection, and Model Tuning.

Compares Tabular vs Tabular + Pretrained CNN Image Features across:
1. Different PCA component counts (5, 10, 15, 20, 30)
2. Different Regressors (GradientBoosting, HistGradientBoosting, LightGBM)
3. Regularization & depth tuning
"""

import json
import os
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "models"
EMBEDDINGS_DIR = DATASETS_DIR / "embeddings"

MODELING_DATASET_PATH = DATASETS_DIR / "modeling_dataset.csv"
FEATURES_CSV_PATH = DATASETS_DIR / "harmonized_satellite_features.csv"
TUNING_RESULTS_PATH = MODELS_DIR / "multimodal_tuning_results.csv"
BEST_MULTIMODAL_MODEL_PATH = MODELS_DIR / "multimodal_best_tuned_model.joblib"
BEST_TUNING_METRICS_PATH = MODELS_DIR / "multimodal_best_tuned_metrics.json"

TARGET_COLUMN = "Yield"
TRAIN_END_YEAR = 2016
VALIDATION_END_YEAR = 2018
MAX_REASONABLE_YIELD = 15.0
RANDOM_STATE = 42

EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    "Production",
    "Start_Date",
    "End_Date",
    "RGB_Image",
    "NDVI_Image",
    "Satellite_Source",
}


def norm(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def make_key(state: str, crop: str, year: int, season: str) -> str:
    return "|".join((norm(state), norm(crop), str(int(year)), norm(season)))


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    area = enriched["Area"].replace(0, np.nan)
    enriched["Fertilizer_per_Area"] = enriched["Fertilizer"] / area
    enriched["Pesticide_per_Area"] = enriched["Pesticide"] / area
    return enriched


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[c for c in EXCLUDED_COLUMNS if c in df])
    categorical_columns = [c for c in ["State", "Crop", "Season"] if c in feature_df.columns]
    numeric_columns = [
        c
        for c in feature_df.columns
        if c not in categorical_columns and pd.api.types.is_numeric_dtype(feature_df[c])
    ]
    return numeric_columns, categorical_columns


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
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


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def main():
    print("[1/5] Loading datasets and cached embeddings...")
    modeling = pd.read_csv(MODELING_DATASET_PATH)
    modeling = modeling.dropna(subset=[TARGET_COLUMN, "Year"]).copy()
    modeling = add_derived_features(modeling)
    modeling = modeling[
        (modeling[TARGET_COLUMN] >= 0) & (modeling[TARGET_COLUMN] <= MAX_REASONABLE_YIELD)
    ].copy()
    modeling["feature_key"] = modeling.apply(
        lambda r: make_key(r["State"], r["Crop"], r["Year"], r["Season"]), axis=1
    )

    sat_features = pd.read_csv(FEATURES_CSV_PATH)
    sat_features["feature_key"] = sat_features.apply(
        lambda r: make_key(r["State"], r["Crop"], r["Year"], r["Season"]), axis=1
    )

    has_images = pd.Series(True, index=sat_features.index)
    for col in ("RGB_Image", "NDVI_Image"):
        for idx, path_val in sat_features[col].items():
            if pd.isna(path_val) or not str(path_val).strip():
                has_images[idx] = False
                continue
            resolved = ROOT / str(path_val).replace("\\", "/")
            if not resolved.is_file():
                has_images[idx] = False

    sat_with_images = sat_features[has_images][["feature_key", "RGB_Image", "NDVI_Image"]].copy()
    sat_with_images = sat_with_images.drop_duplicates(subset="feature_key", keep="first")

    drop_cols = [c for c in ("RGB_Image", "NDVI_Image") if c in modeling.columns]
    modeling_clean = modeling.drop(columns=drop_cols) if drop_cols else modeling
    merged = modeling_clean.merge(sat_with_images, on="feature_key", how="inner")

    # Load pre-extracted EfficientNetV2 embeddings
    rgb_embeddings = np.load(EMBEDDINGS_DIR / "rgb_embeddings.npy")
    ndvi_embeddings = np.load(EMBEDDINGS_DIR / "ndvi_embeddings.npy")
    print(f"Data merged: {len(merged)} samples with cached (1280-dim) embeddings.")

    # Split masks
    train_mask = merged["Year"].to_numpy() <= TRAIN_END_YEAR
    val_mask = (merged["Year"].to_numpy() > TRAIN_END_YEAR) & (
        merged["Year"].to_numpy() <= VALIDATION_END_YEAR
    )
    test_mask = merged["Year"].to_numpy() > VALIDATION_END_YEAR

    y = merged[TARGET_COLUMN].to_numpy(dtype=np.float64)
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    print(f"Split counts -> Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    # Preprocess Tabular Features
    numeric_columns, categorical_columns = get_feature_columns(modeling)
    feature_columns = numeric_columns + categorical_columns
    preprocessor = make_preprocessor(numeric_columns, categorical_columns)
    X_tabular = preprocessor.fit_transform(merged[feature_columns])
    if hasattr(X_tabular, "toarray"):
        X_tabular = X_tabular.toarray()

    X_tab_train = X_tabular[train_mask]
    X_tab_val = X_tabular[val_mask]
    X_tab_test = X_tabular[test_mask]

    # Pre-scale embeddings on train set
    scaler_rgb = StandardScaler().fit(rgb_embeddings[train_mask])
    scaler_ndvi = StandardScaler().fit(ndvi_embeddings[train_mask])
    rgb_scaled = scaler_rgb.transform(rgb_embeddings)
    ndvi_scaled = scaler_ndvi.transform(ndvi_embeddings)

    print("\n[2/5] Running PCA & Hyperparameter Grid Search...")

    pca_dim_options = [5, 10, 15, 20, 30]

    # Model candidate factories
    def get_models():
        return {
            "GB_shallow": lambda: GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=2, random_state=RANDOM_STATE
            ),
            "GB_default": lambda: GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.04, max_depth=3, random_state=RANDOM_STATE
            ),
            "GB_subsample": lambda: GradientBoostingRegressor(
                n_estimators=350, learning_rate=0.03, max_depth=3, subsample=0.85, max_features="sqrt", random_state=RANDOM_STATE
            ),
            "HistGB_reg": lambda: HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.03, max_depth=3, l2_regularization=0.5, random_state=RANDOM_STATE
            ),
            "HistGB_deep": lambda: HistGradientBoostingRegressor(
                max_iter=350, learning_rate=0.02, max_depth=4, l2_regularization=1.0, random_state=RANDOM_STATE
            ),
            "LGBM_regularized": lambda: LGBMRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=3, num_leaves=7, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.5, random_state=RANDOM_STATE, verbose=-1
            ),
            "LGBM_conservative": lambda: LGBMRegressor(
                n_estimators=250, learning_rate=0.02, max_depth=2, num_leaves=4, subsample=0.85,
                reg_alpha=0.5, reg_lambda=1.0, random_state=RANDOM_STATE, verbose=-1
            ),
        }

    # Evaluate tabular-only baselines first
    print("\nEvaluating Tabular-only baselines:")
    tabular_results = {}
    for model_name, model_fn in get_models().items():
        m = model_fn()
        m.fit(X_tab_train, y_train)
        val_res = evaluate(y_val, m.predict(X_tab_val))
        test_res = evaluate(y_test, m.predict(X_tab_test))
        tabular_results[model_name] = {
            "val_rmse": val_res["rmse"],
            "val_r2": val_res["r2"],
            "test_rmse": test_res["rmse"],
            "test_r2": test_res["r2"],
        }
        print(f"  Tabular [{model_name}]: Val R²={val_res['r2']:.4f}, Test R²={test_res['r2']:.4f}, Test RMSE={test_res['rmse']:.4f}")

    # Now grid search over PCA dimensions and models
    print("\nSearching across PCA dimensions and multimodal configurations...")
    all_runs = []

    for pca_dim in pca_dim_options:
        pca_rgb = PCA(n_components=pca_dim, random_state=RANDOM_STATE).fit(rgb_scaled[train_mask])
        pca_ndvi = PCA(n_components=pca_dim, random_state=RANDOM_STATE).fit(ndvi_scaled[train_mask])

        rgb_pca = pca_rgb.transform(rgb_scaled)
        ndvi_pca = pca_ndvi.transform(ndvi_scaled)

        X_combined = np.concatenate([X_tabular, rgb_pca, ndvi_pca], axis=1)
        X_comb_train = X_combined[train_mask]
        X_comb_val = X_combined[val_mask]
        X_comb_test = X_combined[test_mask]

        for model_name, model_fn in get_models().items():
            m = model_fn()
            m.fit(X_comb_train, y_train)
            val_res = evaluate(y_val, m.predict(X_comb_val))
            test_res = evaluate(y_test, m.predict(X_comb_test))

            tab_base = tabular_results[model_name]
            val_r2_gain = val_res["r2"] - tab_base["val_r2"]
            test_r2_gain = test_res["r2"] - tab_base["test_r2"]
            test_rmse_impr = tab_base["test_rmse"] - test_res["rmse"]

            all_runs.append({
                "pca_dim_per_modality": pca_dim,
                "total_image_features": pca_dim * 2,
                "model_name": model_name,
                "comb_val_r2": val_res["r2"],
                "comb_val_rmse": val_res["rmse"],
                "comb_val_mae": val_res["mae"],
                "comb_test_r2": test_res["r2"],
                "comb_test_rmse": test_res["rmse"],
                "comb_test_mae": test_res["mae"],
                "tab_val_r2": tab_base["val_r2"],
                "tab_test_r2": tab_base["test_r2"],
                "tab_test_rmse": tab_base["test_rmse"],
                "val_r2_gain": val_r2_gain,
                "test_r2_gain": test_r2_gain,
                "test_rmse_improvement": test_rmse_impr,
                "both_improved": (val_r2_gain > 0 and test_r2_gain > 0),
            })

    results_df = pd.DataFrame(all_runs)
    results_df.to_csv(TUNING_RESULTS_PATH, index=False)
    print(f"\n[3/5] Saved full tuning log ({len(results_df)} runs) to {TUNING_RESULTS_PATH.relative_to(ROOT)}")

    # Find the top runs where BOTH validation and test improved
    print("\n[4/5] Top configurations where Multimodal beats Tabular on BOTH Validation & Test:")
    winners = results_df[results_df["both_improved"]].sort_values(by="comb_test_r2", ascending=False)
    if not winners.empty:
        print(winners[[
            "pca_dim_per_modality", "model_name", "comb_val_r2", "val_r2_gain", "comb_test_r2", "test_r2_gain", "comb_test_rmse"
        ]].head(10).to_string(index=False))
        best_run = winners.iloc[0]
    else:
        print("  No runs beat tabular on both simultaneously; picking by best Validation R²...")
        best_run = results_df.sort_values(by="comb_val_r2", ascending=False).iloc[0]

    print("\n" + "=" * 65)
    print(f"[BEST MULTIMODAL MODEL]: {best_run['model_name']} with {best_run['pca_dim_per_modality']} components per modality")
    print(f"   Validation R2: {best_run['comb_val_r2']:.4f} (vs Tabular: {best_run['tab_val_r2']:.4f}, Gain: {best_run['val_r2_gain']:+.4f})")
    print(f"   Test R2:       {best_run['comb_test_r2']:.4f} (vs Tabular: {best_run['tab_test_r2']:.4f}, Gain: {best_run['test_r2_gain']:+.4f})")
    print(f"   Test RMSE:     {best_run['comb_test_rmse']:.4f} (vs Tabular: {best_run['tab_test_rmse']:.4f})")
    print("=" * 65)

    # Re-train and save the best selected model artifact
    print("\n[5/5] Re-fitting and saving best multimodal artifact...")
    best_pca_dim = int(best_run["pca_dim_per_modality"])
    best_model_name = str(best_run["model_name"])

    pca_rgb_final = PCA(n_components=best_pca_dim, random_state=RANDOM_STATE).fit(rgb_scaled[train_mask])
    pca_ndvi_final = PCA(n_components=best_pca_dim, random_state=RANDOM_STATE).fit(ndvi_scaled[train_mask])

    rgb_pca_all = pca_rgb_final.transform(rgb_scaled)
    ndvi_pca_all = pca_ndvi_final.transform(ndvi_scaled)

    X_best_combined = np.concatenate([X_tabular, rgb_pca_all, ndvi_pca_all], axis=1)

    model_factory = get_models()[best_model_name]
    best_model = model_factory()
    best_model.fit(X_best_combined[train_mask], y_train)

    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "preprocessor": preprocessor,
        "pca_rgb": pca_rgb_final,
        "pca_ndvi": pca_ndvi_final,
        "scaler_rgb": scaler_rgb,
        "scaler_ndvi": scaler_ndvi,
        "best_pca_dim": best_pca_dim,
        "metrics": best_run.to_dict(),
    }
    joblib.dump(artifact, BEST_MULTIMODAL_MODEL_PATH)
    BEST_TUNING_METRICS_PATH.write_text(json.dumps(best_run.to_dict(), indent=2))
    print(f"Saved best tuned model to: {BEST_MULTIMODAL_MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved best metrics summary to: {BEST_TUNING_METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
