"""Link/copy existing satellite images for Moong(Green Gram) across identical (State, Year, Season) records.

1. Reuses 573 existing seasonal satellite image pairs on disk by copying them to dedicated Moong paths.
2. Updates harmonized_satellite_features.csv with these new unique image paths (avoiding path reuse exclusions).
3. Identifies the remaining 167 targets that need fetching from Earth Engine and saves them to datasets/moong_targets_to_fetch.csv.
"""

import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
CROP_YIELD_PATH = DATASETS_DIR / "crop_yield.csv"
SATELLITE_FEATURES_PATH = DATASETS_DIR / "harmonized_satellite_features.csv"
TARGETS_TO_FETCH_PATH = DATASETS_DIR / "moong_targets_to_fetch.csv"


def slug(val: str) -> str:
    return str(val).strip().lower().replace(" ", "_").replace("(", "_").replace(")", "").replace("/", "_")


def main():
    print("[1/4] Scanning existing satellite images on disk...")
    cy = pd.read_csv(CROP_YIELD_PATH)
    hf = pd.read_csv(SATELLITE_FEATURES_PATH)

    moong = cy[cy["Crop"] == "Moong(Green Gram)"].copy()
    moong["State_key"] = moong["State"].str.lower().str.strip()
    moong["Season_key"] = moong["Season"].str.lower().str.strip()
    moong["Year"] = moong["Crop_Year"].astype(int)

    # Base reference satellite data (Rice & Maize)
    hf_base = hf[hf["Crop"] != "Moong(Green Gram)"].copy()
    hf_base["State_key"] = hf_base["State"].str.lower().str.strip()
    hf_base["Season_key"] = hf_base["Season"].str.lower().str.strip()

    def img_valid(p):
        if pd.isna(p) or not str(p).strip():
            return False
        return (ROOT / str(p).replace("\\", "/")).is_file()

    hf_base["has_rgb"] = hf_base["RGB_Image"].apply(img_valid)
    hf_base["has_ndvi"] = hf_base["NDVI_Image"].apply(img_valid)
    hf_valid = hf_base[hf_base["has_rgb"] & hf_base["has_ndvi"]].copy()

    # Find the reusable set
    reusable_map = {}
    for _, row in hf_valid.iterrows():
        key = (row["State_key"], int(row["Year"]), row["Season_key"])
        if key not in reusable_map:
            reusable_map[key] = (row["RGB_Image"], row["NDVI_Image"])

    print(f"Found {len(reusable_map)} unique state/year/season image pairs on disk.")

    copied_count = 0
    missing_targets = []
    updated_moong_rows = []

    print("[2/4] Copying images to dedicated Moong paths...")
    for _, row in moong.iterrows():
        st_name = row["State"]
        crop_name = row["Crop"]
        yr = int(row["Crop_Year"])
        season_name = str(row["Season"]).strip()
        key = (row["State_key"], yr, row["Season_key"])

        if key in reusable_map:
            src_rgb, src_ndvi = reusable_map[key]
            src_rgb_path = ROOT / str(src_rgb).replace("\\", "/")
            src_ndvi_path = ROOT / str(src_ndvi).replace("\\", "/")

            # Destination dedicated paths for Moong
            year_dir = DATASETS_DIR / "test_images" / str(yr)
            rgb_dir = year_dir / "rgb"
            ndvi_dir = year_dir / "ndvi"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            ndvi_dir.mkdir(parents=True, exist_ok=True)

            prefix = f"{slug(st_name)}_{yr}_moong_green_gram_{slug(season_name)}"
            dst_rgb_path = rgb_dir / f"{prefix}_rgb.png"
            dst_ndvi_path = ndvi_dir / f"{prefix}_ndvi.png"

            # Copy file if not exists
            if not dst_rgb_path.exists() and src_rgb_path.exists():
                shutil.copy2(src_rgb_path, dst_rgb_path)
            if not dst_ndvi_path.exists() and src_ndvi_path.exists():
                shutil.copy2(src_ndvi_path, dst_ndvi_path)

            copied_count += 1
            updated_moong_rows.append({
                "State": st_name,
                "Crop": crop_name,
                "Year": yr,
                "Season": season_name,
                "RGB_Image": str(dst_rgb_path.relative_to(ROOT)),
                "NDVI_Image": str(dst_ndvi_path.relative_to(ROOT)),
            })
        else:
            missing_targets.append({
                "State": st_name,
                "Crop": crop_name,
                "Crop_Year": yr,
                "Season": season_name,
            })

    print(f"[3/4] Successfully linked/copied {copied_count} dedicated image pairs for Moong!")
    print(f"Identified {len(missing_targets)} missing targets requiring fresh Earth Engine fetching.")

    # Save missing targets to CSV
    missing_df = pd.DataFrame(missing_targets).drop_duplicates()
    missing_df.to_csv(TARGETS_TO_FETCH_PATH, index=False)
    print(f"Saved {len(missing_df)} unique missing targets to: {TARGETS_TO_FETCH_PATH.relative_to(ROOT)}")

    # Update harmonized_satellite_features.csv with these new image paths
    print("[4/4] Updating harmonized_satellite_features.csv...")
    moong_paths_df = pd.DataFrame(updated_moong_rows).drop_duplicates(subset=["State", "Crop", "Year", "Season"])
    
    # Merge paths into hf for Moong
    hf_non_moong = hf[hf["Crop"] != "Moong(Green Gram)"].copy()
    hf_moong = hf[hf["Crop"] == "Moong(Green Gram)"].copy()
    hf_moong = hf_moong.drop(columns=["RGB_Image", "NDVI_Image"], errors="ignore")
    hf_moong = hf_moong.merge(moong_paths_df, on=["State", "Crop", "Year", "Season"], how="left")

    hf_updated = pd.concat([hf_non_moong, hf_moong], ignore_index=True)
    hf_updated.to_csv(SATELLITE_FEATURES_PATH, index=False)
    print(f"Updated {SATELLITE_FEATURES_PATH.name} successfully.")


if __name__ == "__main__":
    main()
