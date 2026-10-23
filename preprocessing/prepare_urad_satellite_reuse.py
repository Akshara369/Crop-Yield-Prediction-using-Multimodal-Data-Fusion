from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"

CROP_YIELD_PATH = DATASETS_DIR / "crop_yield.csv"
SATELLITE_FEATURES_PATH = DATASETS_DIR / "harmonized_satellite_features.csv"

URAD_DATA_PATH = DATASETS_DIR / "urad_data.csv"
REUSED_MATCHES_PATH = DATASETS_DIR / "urad_satellite_reused_matches.csv"
MISSING_TARGETS_PATH = DATASETS_DIR / "urad_missing_satellite_targets.csv"
MODEL_READY_REUSED_PATH = DATASETS_DIR / "urad_modeling_dataset_reused_satellite.csv"

TARGET_CROP = "Urad"
ORIGINAL_YIELD_COLUMNS = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield",
]
REUSE_KEYS = ["State_key", "Year", "Season_key"]
IMAGE_COLUMNS = {"RGB_Image", "NDVI_Image"}
SATELLITE_METADATA_COLUMNS = {
    "Start_Date",
    "End_Date",
    "Satellite_Source",
    "Cloud_Score_Threshold",
    "Buffer_Degree",
}


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def add_yield_keys(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    keyed.columns = keyed.columns.str.strip()
    keyed["State"] = normalize_text(keyed["State"])
    keyed["Crop"] = normalize_text(keyed["Crop"])
    keyed["Season"] = normalize_text(keyed["Season"])
    keyed["Year"] = pd.to_numeric(keyed["Crop_Year"], errors="coerce").astype("Int64")
    keyed["State_key"] = keyed["State"].str.casefold()
    keyed["Season_key"] = keyed["Season"].str.casefold()
    return keyed


def add_satellite_keys(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    keyed.columns = keyed.columns.str.strip()
    keyed["State"] = normalize_text(keyed["State"])
    keyed["Crop"] = normalize_text(keyed["Crop"])
    keyed["Season"] = normalize_text(keyed["Season"])
    keyed["Year"] = pd.to_numeric(keyed["Year"], errors="coerce").astype("Int64")
    keyed["State_key"] = keyed["State"].str.casefold()
    keyed["Season_key"] = keyed["Season"].str.casefold()
    return keyed


def first_non_null(values: pd.Series):
    non_null = values.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def combine_unique(values: pd.Series) -> str:
    unique_values = [str(value) for value in values.dropna().unique()]
    return "; ".join(unique_values)


def collapse_satellite_group(group: pd.DataFrame) -> pd.Series:
    collapsed = {}

    for column in group.columns:
        if column in REUSE_KEYS:
            collapsed[column] = group[column].iloc[0]
        elif column == "Crop":
            collapsed["Satellite_Source_Crop"] = combine_unique(group[column])
            collapsed[column] = first_non_null(group[column])
        elif column in {"State", "Season"}:
            collapsed[column] = group[column].iloc[0]
        elif column in IMAGE_COLUMNS:
            collapsed[column] = first_non_null(group[column])
        elif column == "Image_Count":
            collapsed[column] = group[column].max()
        elif column == "Satellite_Source":
            collapsed[column] = combine_unique(group[column])
        elif column in SATELLITE_METADATA_COLUMNS:
            collapsed[column] = first_non_null(group[column])
        elif pd.api.types.is_numeric_dtype(group[column]):
            collapsed[column] = group[column].mean()
        else:
            collapsed[column] = first_non_null(group[column])

    return pd.Series(collapsed)


def main() -> None:
    crop_yield = pd.read_csv(CROP_YIELD_PATH)
    crop_yield = add_yield_keys(crop_yield)

    urad = crop_yield[crop_yield["Crop"].str.casefold() == TARGET_CROP.casefold()].copy()
    urad[ORIGINAL_YIELD_COLUMNS].to_csv(URAD_DATA_PATH, index=False)

    satellite = pd.read_csv(SATELLITE_FEATURES_PATH)
    satellite = add_satellite_keys(satellite)
    satellite_reused = pd.DataFrame(
        [
            collapse_satellite_group(group)
            for _, group in satellite.groupby(REUSE_KEYS, sort=False)
        ]
    ).reset_index(drop=True)

    urad_targets = urad[["State", "Crop", "Crop_Year", "Season", *REUSE_KEYS]].drop_duplicates()
    matches = urad_targets.merge(
        satellite_reused,
        on=REUSE_KEYS,
        how="left",
        suffixes=("", "_satellite"),
        indicator=True,
    )

    reused = matches[matches["_merge"] == "both"].copy()
    missing = matches[matches["_merge"] == "left_only"].copy()

    reused.to_csv(REUSED_MATCHES_PATH, index=False)
    missing[["State", "Crop", "Crop_Year", "Season"]].sort_values(
        ["State", "Crop_Year", "Season"]
    ).to_csv(MISSING_TARGETS_PATH, index=False)

    model_ready = urad.merge(
        satellite_reused,
        on=REUSE_KEYS,
        how="inner",
        suffixes=("", "_satellite"),
    )
    model_ready = model_ready.drop(
        columns=[
            "State_key",
            "Season_key",
            "State_satellite",
            "Crop_satellite",
            "Season_satellite",
            "Crop_Year",
            "_merge",
        ],
        errors="ignore",
    )
    ordered_columns = [
        "State",
        "Crop",
        "Year",
        "Season",
        "Area",
        "Production",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide",
        "Yield",
    ]
    remaining_columns = [column for column in model_ready.columns if column not in ordered_columns]
    model_ready = model_ready[ordered_columns + remaining_columns]
    model_ready.to_csv(MODEL_READY_REUSED_PATH, index=False)

    print(f"Urad crop rows: {len(urad)}")
    print(f"Unique Urad state-year-season targets: {len(urad_targets)}")
    print(f"Reusable existing satellite targets: {len(reused)}")
    print(f"Missing targets to fetch/process: {len(missing)}")
    print(f"Wrote {URAD_DATA_PATH.relative_to(ROOT)}")
    print(f"Wrote {REUSED_MATCHES_PATH.relative_to(ROOT)}")
    print(f"Wrote {MISSING_TARGETS_PATH.relative_to(ROOT)}")
    print(f"Wrote {MODEL_READY_REUSED_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
