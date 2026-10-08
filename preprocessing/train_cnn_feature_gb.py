"""Strategy 4: Pretrained CNN Feature Extractor + Gradient Boosting.

Extract embeddings from a pretrained EfficientNetV2B0 (no CNN training needed),
reduce dimensionality with PCA, combine with existing tabular features, and
train gradient boosting on the combined feature set.

Run AFTER:
  1. Fetching satellite images (fetch_single_state_satellite.py)
  2. Rebuilding the manifest (build_cnn_manifest.py)
  3. Preparing the CNN dataset (prepare_cnn_dataset.py)
  4. Training the tabular baseline (train_tabular_baseline.py)
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = MODELS_DIR / "reports"

# Input paths
MODELING_DATASET_PATH = DATASETS_DIR / "modeling_dataset.csv"
FEATURES_CSV_PATH = DATASETS_DIR / "harmonized_satellite_features.csv"

# Output paths
METRICS_PATH = MODELS_DIR / "cnn_feature_gb_metrics.json"
MODEL_PATH = MODELS_DIR / "cnn_feature_gb_model.joblib"
PREDICTIONS_PATH = MODELS_DIR / "cnn_feature_gb_test_predictions.csv"
EMBEDDINGS_DIR = DATASETS_DIR / "embeddings"

# Config
IMAGE_SIZE = (224, 224)
PCA_COMPONENTS_RGB = 30
PCA_COMPONENTS_NDVI = 30
RANDOM_STATE = 42
TARGET_COLUMN = "Yield"
TRAIN_END_YEAR = 2016
VALIDATION_END_YEAR = 2018
MAX_REASONABLE_YIELD = 15.0

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


def load_and_preprocess_image(path: str | Path, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load image, resize, and return as float32 array [0, 255] for EfficientNetV2."""
    resolved = ROOT / str(path).replace("\\", "/")
    with Image.open(resolved) as img:
        img = img.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.float32)


def extract_embeddings_batch(image_paths: list[str], backbone, batch_size: int = 16) -> np.ndarray:
    """Extract embeddings for a batch of images using the pretrained backbone."""
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

    all_embeddings = []
    total = len(image_paths)
    for i in range(0, total, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = np.stack([load_and_preprocess_image(p) for p in batch_paths])
        batch_images = preprocess_input(batch_images)
        embeddings = backbone.predict(batch_images, verbose=0)
        all_embeddings.append(embeddings)
        done = min(i + batch_size, total)
        if (i // batch_size) % 5 == 0:
            print(f"  Extracted {done}/{total} embeddings...")

    return np.vstack(all_embeddings)


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[c for c in EXCLUDED_COLUMNS if c in df])
    categorical_columns = [c for c in ["State", "Crop", "Season"] if c in feature_df.columns]
    numeric_columns = [
        c for c in feature_df.columns
        if c not in categorical_columns and pd.api.types.is_numeric_dtype(feature_df[c])
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


def main() -> None:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetV2B0

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # ── Step 1: Load the tabular modeling dataset ──────────────────────────
    print("Loading modeling dataset...")
    modeling = pd.read_csv(MODELING_DATASET_PATH)
    modeling = modeling.dropna(subset=[TARGET_COLUMN, "Year"]).copy()
    modeling = add_derived_features(modeling)
    modeling = modeling[
        (modeling[TARGET_COLUMN] >= 0) & (modeling[TARGET_COLUMN] <= MAX_REASONABLE_YIELD)
    ].copy()
    modeling["feature_key"] = modeling.apply(
        lambda r: make_key(r["State"], r["Crop"], r["Year"], r["Season"]), axis=1
    )

    # ── Step 2: Load image paths from harmonized features ────────────────
    print("Loading satellite feature paths...")
    sat_features = pd.read_csv(FEATURES_CSV_PATH)
    sat_features["feature_key"] = sat_features.apply(
        lambda r: make_key(r["State"], r["Crop"], r["Year"], r["Season"]), axis=1
    )

    # Keep only rows that have BOTH RGB and NDVI images on disk
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

    # ── Step 3: Merge tabular data with image paths ──────────────────────
    # Drop image path columns from modeling to avoid collision with sat_with_images
    # Drop image path columns from modeling to avoid collision with sat_with_images
    drop_cols = [c for c in ("RGB_Image", "NDVI_Image") if c in modeling.columns]
    modeling_clean = modeling.drop(columns=drop_cols) if drop_cols else modeling
    merged = modeling_clean.merge(sat_with_images, on="feature_key", how="inner")
    print(f"Tabular rows: {len(modeling)}, Rows with images: {len(sat_with_images)}, Merged: {len(merged)}")

    if merged.empty:
        raise ValueError("No modeling rows matched with available satellite images!")

    # ── Step 4: Extract CNN embeddings ───────────────────────────────────
    rgb_cache = EMBEDDINGS_DIR / "rgb_embeddings.npy"
    ndvi_cache = EMBEDDINGS_DIR / "ndvi_embeddings.npy"
    keys_cache = EMBEDDINGS_DIR / "embedding_keys.npy"

    use_cache = False
    if rgb_cache.is_file() and ndvi_cache.is_file() and keys_cache.is_file():
        cached_keys = np.load(keys_cache, allow_pickle=True)
        if np.array_equal(cached_keys, merged["feature_key"].to_numpy()):
            use_cache = True
            print("Using cached embeddings.")

    if use_cache:
        rgb_embeddings = np.load(rgb_cache)
        ndvi_embeddings = np.load(ndvi_cache)
    else:
        print("Loading pretrained EfficientNetV2B0 backbone...")
        backbone = EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(*IMAGE_SIZE, 3),
        )
        backbone.trainable = False

        print(f"Extracting RGB embeddings for {len(merged)} images...")
        rgb_embeddings = extract_embeddings_batch(
            merged["RGB_Image"].tolist(), backbone, batch_size=16
        )
        print(f"Extracting NDVI embeddings for {len(merged)} images...")
        ndvi_embeddings = extract_embeddings_batch(
            merged["NDVI_Image"].tolist(), backbone, batch_size=16
        )

        np.save(rgb_cache, rgb_embeddings)
        np.save(ndvi_cache, ndvi_embeddings)
        np.save(keys_cache, merged["feature_key"].to_numpy())
        print(f"Cached embeddings to {EMBEDDINGS_DIR.relative_to(ROOT)}/")

    print(f"RGB embedding shape: {rgb_embeddings.shape}, NDVI embedding shape: {ndvi_embeddings.shape}")

    # ── Step 5: PCA dimensionality reduction ─────────────────────────────
    print("Applying PCA dimensionality reduction...")
    train_mask = merged["Year"].to_numpy() <= TRAIN_END_YEAR
    val_mask = (merged["Year"].to_numpy() > TRAIN_END_YEAR) & (
        merged["Year"].to_numpy() <= VALIDATION_END_YEAR
    )
    test_mask = merged["Year"].to_numpy() > VALIDATION_END_YEAR

    n_rgb = min(PCA_COMPONENTS_RGB, int(train_mask.sum()) - 1, rgb_embeddings.shape[1])
    n_ndvi = min(PCA_COMPONENTS_NDVI, int(train_mask.sum()) - 1, ndvi_embeddings.shape[1])

    pca_rgb = PCA(n_components=n_rgb, random_state=RANDOM_STATE)
    pca_ndvi = PCA(n_components=n_ndvi, random_state=RANDOM_STATE)
    scaler_rgb = StandardScaler()
    scaler_ndvi = StandardScaler()

    # Fit on train only (prevent leakage)
    scaler_rgb.fit(rgb_embeddings[train_mask])
    scaler_ndvi.fit(ndvi_embeddings[train_mask])
    pca_rgb.fit(scaler_rgb.transform(rgb_embeddings[train_mask]))
    pca_ndvi.fit(scaler_ndvi.transform(ndvi_embeddings[train_mask]))

    rgb_pca_all = pca_rgb.transform(scaler_rgb.transform(rgb_embeddings))
    ndvi_pca_all = pca_ndvi.transform(scaler_ndvi.transform(ndvi_embeddings))

    print(
        f"PCA variance retained - RGB: {pca_rgb.explained_variance_ratio_.sum():.3f}, "
        f"NDVI: {pca_ndvi.explained_variance_ratio_.sum():.3f}"
    )

    # ── Step 6: Build combined feature matrix ────────────────────────────
    numeric_columns, categorical_columns = get_feature_columns(modeling)
    feature_columns = numeric_columns + categorical_columns

    preprocessor = make_preprocessor(numeric_columns, categorical_columns)
    X_tabular = preprocessor.fit_transform(merged[feature_columns])
    if hasattr(X_tabular, "toarray"):
        X_tabular = X_tabular.toarray()

    rgb_pca_cols = [f"rgb_pca_{i}" for i in range(rgb_pca_all.shape[1])]
    ndvi_pca_cols = [f"ndvi_pca_{i}" for i in range(ndvi_pca_all.shape[1])]

    X_combined = np.concatenate([X_tabular, rgb_pca_all, ndvi_pca_all], axis=1)
    y = merged[TARGET_COLUMN].to_numpy(dtype=np.float64)

    print(
        f"Combined feature matrix: {X_combined.shape} "
        f"(tabular: {X_tabular.shape[1]} + RGB PCA: {rgb_pca_all.shape[1]} + NDVI PCA: {ndvi_pca_all.shape[1]})"
    )

    # ── Step 7: Train and evaluate ──────────────────────────────────────
    X_train, y_train = X_combined[train_mask], y[train_mask]
    X_val, y_val = X_combined[val_mask], y[val_mask]
    X_test, y_test = X_combined[test_mask], y[test_mask]

    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    gb_model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.04, max_depth=3, random_state=RANDOM_STATE,
    )
    print("Training Gradient Boosting on combined features...")
    gb_model.fit(X_train, y_train)

    # Also train tabular-only on same subset for fair comparison
    X_tab_train = X_tabular[train_mask]
    X_tab_val = X_tabular[val_mask]
    X_tab_test = X_tabular[test_mask]

    gb_tabular_only = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.04, max_depth=3, random_state=RANDOM_STATE,
    )
    gb_tabular_only.fit(X_tab_train, y_train)

    results = {}
    for name, model, xv, xt in [
        ("combined_tabular_cnn", gb_model, X_val, X_test),
        ("tabular_only_same_samples", gb_tabular_only, X_tab_val, X_tab_test),
    ]:
        val_pred = model.predict(xv)
        test_pred = model.predict(xt)
        results[name] = {
            "validation": evaluate(y_val, val_pred),
            "test": evaluate(y_test, test_pred),
        }
        print(f"\n{name}:")
        print(f"  Validation: {results[name]['validation']}")
        print(f"  Test:       {results[name]['test']}")

    # ── Step 8: Save outputs ────────────────────────────────────────────
    test_rows = merged[test_mask][
        ["feature_key", "State", "Crop", "Year", "Season", TARGET_COLUMN]
    ].copy()
    test_rows["combined_predicted_yield"] = gb_model.predict(X_test)
    test_rows["tabular_only_predicted_yield"] = gb_tabular_only.predict(X_tab_test)
    test_rows["combined_abs_error"] = np.abs(
        test_rows[TARGET_COLUMN] - test_rows["combined_predicted_yield"]
    )
    test_rows["tabular_only_abs_error"] = np.abs(
        test_rows[TARGET_COLUMN] - test_rows["tabular_only_predicted_yield"]
    )
    test_rows.to_csv(PREDICTIONS_PATH, index=False)

    tabular_feature_names = preprocessor.get_feature_names_out().tolist()
    all_feature_names = tabular_feature_names + rgb_pca_cols + ndvi_pca_cols
    importances = pd.DataFrame(
        {"feature": all_feature_names, "importance": gb_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    metrics = {
        "strategy": "pretrained_cnn_feature_extractor_plus_gradient_boosting",
        "backbone": "EfficientNetV2B0_imagenet",
        "image_size": list(IMAGE_SIZE),
        "pca_components": {
            "rgb": int(rgb_pca_all.shape[1]),
            "ndvi": int(ndvi_pca_all.shape[1]),
            "rgb_variance_retained": float(pca_rgb.explained_variance_ratio_.sum()),
            "ndvi_variance_retained": float(pca_ndvi.explained_variance_ratio_.sum()),
        },
        "feature_dimensions": {
            "tabular": int(X_tabular.shape[1]),
            "rgb_pca": int(rgb_pca_all.shape[1]),
            "ndvi_pca": int(ndvi_pca_all.shape[1]),
            "total_combined": int(X_combined.shape[1]),
        },
        "sample_counts": {
            "total_with_images": int(len(merged)),
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "results": results,
        "top_10_features": importances.head(10).to_dict(orient="records"),
        "top_5_image_features": importances[
            importances["feature"].str.startswith(("rgb_pca", "ndvi_pca"))
        ]
        .head(5)
        .to_dict(orient="records"),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    artifact = {
        "gb_model": gb_model,
        "preprocessor": preprocessor,
        "pca_rgb": pca_rgb,
        "pca_ndvi": pca_ndvi,
        "scaler_rgb": scaler_rgb,
        "scaler_ndvi": scaler_ndvi,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "image_size": IMAGE_SIZE,
    }
    joblib.dump(artifact, MODEL_PATH)

    print(f"\nSaved metrics: {METRICS_PATH.relative_to(ROOT)}")
    print(f"Saved model:   {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved predictions: {PREDICTIONS_PATH.relative_to(ROOT)}")

    combined_test = results["combined_tabular_cnn"]["test"]
    tabular_test = results["tabular_only_same_samples"]["test"]
    improvement = tabular_test["rmse"] - combined_test["rmse"]
    print(f"\n{'='*60}")
    print(
        f"Combined (Tabular+CNN):  RMSE={combined_test['rmse']:.4f}  "
        f"MAE={combined_test['mae']:.4f}  R²={combined_test['r2']:.4f}"
    )
    print(
        f"Tabular Only (same N):   RMSE={tabular_test['rmse']:.4f}  "
        f"MAE={tabular_test['mae']:.4f}  R²={tabular_test['r2']:.4f}"
    )
    print(f"RMSE improvement: {improvement:+.4f} ({'better' if improvement > 0 else 'worse'})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
