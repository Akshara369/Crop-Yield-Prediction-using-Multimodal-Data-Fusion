from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "datasets" / "rice_data.csv"
CLEANED_PATH = ROOT / "datasets" / "rice_data_cleaned.csv"
PREPROCESSED_PATH = ROOT / "datasets" / "rice_data_preprocessed.csv"

NUMERIC_COLUMNS = [
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield",
]


def clean_rice_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean labels, enforce types, and add agronomic ratio features."""
    cleaned = df.copy()

    cleaned.columns = cleaned.columns.str.strip()
    for column in ["Crop", "Season", "State"]:
        cleaned[column] = cleaned[column].astype(str).str.strip()

    cleaned["Crop_Year"] = pd.to_numeric(cleaned["Crop_Year"], errors="coerce").astype("Int64")
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna().drop_duplicates()
    cleaned = cleaned[cleaned["Crop"].str.casefold() == "rice"]
    cleaned = cleaned[
        (cleaned["Crop_Year"].between(1997, 2020))
        & (cleaned[NUMERIC_COLUMNS] > 0).all(axis=1)
    ].copy()

    cleaned["Fertilizer_per_Area"] = cleaned["Fertilizer"] / cleaned["Area"]
    cleaned["Pesticide_per_Area"] = cleaned["Pesticide"] / cleaned["Area"]
    cleaned["Production_per_Area"] = cleaned["Production"] / cleaned["Area"]

    ordered_columns = [
        "Crop",
        "Crop_Year",
        "Season",
        "State",
        "Area",
        "Production",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide",
        "Fertilizer_per_Area",
        "Pesticide_per_Area",
        "Production_per_Area",
        "Yield",
    ]
    return cleaned[ordered_columns].sort_values(
        ["Crop_Year", "State", "Season"], ignore_index=True
    )


def make_model_ready(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Create one-hot encoded and standardized features for ML experiments."""
    model_ready = cleaned.drop(columns=["Crop"]).copy()
    model_ready = pd.get_dummies(model_ready, columns=["Season", "State"], dtype=int)

    feature_columns = [column for column in model_ready.columns if column != "Yield"]
    numeric_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(model_ready[column])
        and model_ready[column].nunique(dropna=True) > 2
    ]

    for column in numeric_features:
        mean = model_ready[column].mean()
        std = model_ready[column].std(ddof=0)
        if std:
            model_ready[f"{column}_scaled"] = (model_ready[column] - mean) / std

    target = model_ready.pop("Yield")
    model_ready["Yield"] = target
    return model_ready


def main() -> None:
    raw = pd.read_csv(INPUT_PATH)
    cleaned = clean_rice_data(raw)
    model_ready = make_model_ready(cleaned)

    cleaned.to_csv(CLEANED_PATH, index=False)
    model_ready.to_csv(PREPROCESSED_PATH, index=False)

    print(f"Raw rows: {len(raw)}")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Cleaned file: {CLEANED_PATH}")
    print(f"Preprocessed file: {PREPROCESSED_PATH}")
    print(f"Preprocessed shape: {model_ready.shape}")


if __name__ == "__main__":
    main()
