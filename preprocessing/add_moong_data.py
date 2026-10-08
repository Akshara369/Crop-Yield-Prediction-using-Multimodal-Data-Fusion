"""Add Moong(Green Gram) into harmonized_satellite_features.csv and modeling_dataset.csv.

Maps existing state/year/season satellite spectral measurements (NDVI, EVI, NDWI, band statistics)
from identical regional seasonal composites to Moong(Green Gram) records without requiring
Earth Engine re-downloading.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
CROP_YIELD_PATH = DATASETS_DIR / "crop_yield.csv"
SATELLITE_FEATURES_PATH = DATASETS_DIR / "harmonized_satellite_features.csv"


def main():
    print("[1/3] Reading crop_yield.csv and harmonized_satellite_features.csv...")
    cy = pd.read_csv(CROP_YIELD_PATH)
    hf = pd.read_csv(SATELLITE_FEATURES_PATH)

    # Backup original harmonized_satellite_features.csv
    backup_path = SATELLITE_FEATURES_PATH.with_suffix(".csv.pre_moong_bak")
    if not backup_path.exists():
        hf.to_csv(backup_path, index=False)
        print(f"Backed up original satellite features to: {backup_path.name}")

    moong_cy = cy[cy["Crop"].str.contains("Moong", case=False, na=False)].copy()
    moong_cy["State_key"] = moong_cy["State"].str.lower().str.strip()
    moong_cy["Season_key"] = moong_cy["Season"].str.lower().str.strip()
    moong_cy["Year"] = moong_cy["Crop_Year"].astype(int)

    hf["State_key"] = hf["State"].str.lower().str.strip()
    hf["Season_key"] = hf["Season"].str.lower().str.strip()

    # Drop any existing Moong rows from hf to avoid duplicates
    hf = hf[~hf["Crop"].str.contains("Moong", case=False, na=False)].copy()

    # Get unique regional satellite seasonal records
    hf_seasonal = hf.drop_duplicates(subset=["State_key", "Year", "Season_key"]).copy()

    # Merge Moong target records with existing regional seasonal satellite data
    moong_sat = moong_cy[["State_key", "Year", "Season_key", "State", "Season"]].merge(
        hf_seasonal.drop(columns=["State", "Season", "Crop"]),
        on=["State_key", "Year", "Season_key"],
        how="inner",
    )
    moong_sat["Crop"] = "Moong(Green Gram)"
    moong_sat = moong_sat.drop(columns=["State_key", "Season_key"], errors="ignore")
    hf_clean = hf.drop(columns=["State_key", "Season_key"], errors="ignore")

    # Combine
    combined = pd.concat([hf_clean, moong_sat], ignore_index=True)
    combined = combined.drop_duplicates(subset=["State", "Crop", "Year", "Season"], keep="first")
    combined.to_csv(SATELLITE_FEATURES_PATH, index=False)

    print(f"[2/3] Added {len(moong_sat)} seasonal satellite records for Moong(Green Gram).")
    print(f"Total rows in harmonized_satellite_features.csv: {len(combined)}")
    print(f"Crops present in satellite dataset: {combined['Crop'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
