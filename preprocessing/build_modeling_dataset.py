from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"

CROP_YIELD_PATH = DATASETS_DIR / "crop_yield.csv"
SATELLITE_FEATURES_PATH = DATASETS_DIR / "harmonized_satellite_features.csv"
OUTPUT_PATH = DATASETS_DIR / "modeling_dataset.csv"

TARGET_CROPS = {"rice", "maize"}
JOIN_KEYS = ["State_key", "Crop_key", "Year", "Season_key"]
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


def add_join_keys(df: pd.DataFrame, year_column: str) -> pd.DataFrame:
    keyed = df.copy()
    keyed["State"] = normalize_text(keyed["State"])
    keyed["Crop"] = normalize_text(keyed["Crop"])
    keyed["Season"] = normalize_text(keyed["Season"])
    keyed["Year"] = keyed[year_column].astype(int)
    keyed["State_key"] = keyed["State"].str.lower()
    keyed["Crop_key"] = keyed["Crop"].str.lower()
    keyed["Season_key"] = keyed["Season"].str.lower()
    return keyed


def first_non_null(values: pd.Series):
    non_null = values.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def combine_unique(values: pd.Series) -> str:
    unique_values = [str(value) for value in values.dropna().unique()]
    return "; ".join(unique_values)


def choose_crop_image_path(paths: pd.Series, crop: str):
    non_null_paths = [str(path) for path in paths.dropna()]
    if not non_null_paths:
        return pd.NA

    crop_token = crop.lower().replace(" ", "_")
    for path in non_null_paths:
        if crop_token in path.lower():
            return path

    return non_null_paths[0]


def collapse_satellite_group(group: pd.DataFrame) -> pd.Series:
    collapsed = {}
    crop = str(group["Crop"].iloc[0])

    for column in group.columns:
        if column in JOIN_KEYS:
            collapsed[column] = group[column].iloc[0]
        elif column in {"State", "Crop", "Season"}:
            collapsed[column] = group[column].iloc[0]
        elif column in IMAGE_COLUMNS:
            collapsed[column] = choose_crop_image_path(group[column], crop)
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


def build_modeling_dataset() -> tuple[pd.DataFrame, dict[str, int]]:
    crop_yield = pd.read_csv(CROP_YIELD_PATH)
    satellite = pd.read_csv(SATELLITE_FEATURES_PATH)

    crop_yield = add_join_keys(crop_yield, "Crop_Year")
    satellite = add_join_keys(satellite, "Year")

    crop_yield = crop_yield[crop_yield["Crop_key"].isin(TARGET_CROPS)].copy()
    satellite = satellite[satellite["Crop_key"].isin(TARGET_CROPS)].copy()

    raw_satellite_rows = len(satellite)
    satellite = pd.DataFrame(
        [
            collapse_satellite_group(group)
            for _, group in satellite.groupby(JOIN_KEYS, sort=False)
        ]
    ).reset_index(drop=True)

    merged = crop_yield.merge(
        satellite,
        on=JOIN_KEYS,
        how="inner",
        suffixes=("", "_satellite"),
    )

    merged = merged.drop(
        columns=[
            "State_satellite",
            "Crop_satellite",
            "Season_satellite",
            "Crop_Year",
            "State_key",
            "Crop_key",
            "Season_key",
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
    remaining_columns = [column for column in merged.columns if column not in ordered_columns]
    merged = merged[ordered_columns + remaining_columns]

    report = {
        "yield_rows_rice_maize": len(crop_yield),
        "satellite_rows_raw": raw_satellite_rows,
        "satellite_rows_deduplicated": len(satellite),
        "merged_rows": len(merged),
        "unmatched_yield_rows": len(crop_yield) - len(merged),
    }

    return merged, report


def main() -> None:
    modeling_dataset, report = build_modeling_dataset()
    modeling_dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
