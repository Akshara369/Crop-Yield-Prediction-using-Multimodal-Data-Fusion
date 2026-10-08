import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Iterable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
COORDS_PATH = ROOT / "datasets" / "state_coordinates.csv"
FEATURES_PATH = ROOT / "datasets" / "harmonized_satellite_features.csv"
FAILED_TARGETS_PATH = ROOT / "datasets" / "failed_satellite_targets.csv"
OUTPUT_DIR = ROOT / "datasets" / "test_images"
COMMON_BANDS = ["BLUE", "GREEN", "RED", "NIR"]


class EarthEngineSetupError(RuntimeError):
    """Raised when local Earth Engine credentials/project are not ready."""


def slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("/", "_")


def season_window(crop: str, season: str | None = None) -> tuple[int, int]:
    season_key = (season or "").strip().casefold()
    crop_key = crop.strip().casefold()

    if season_key == "rabi" or crop_key == "wheat":
        return (11, 4)
    if season_key in {"summer", "zaid"}:
        return (3, 6)
    if season_key == "whole year":
        return (1, 12)
    return (7, 10)


def date_range_for_season(year: int, months: tuple[int, int]) -> tuple[str, str]:
    start_month, end_month = months
    start_year = year
    end_year = year if end_month >= start_month else year + 1
    return f"{start_year}-{start_month:02d}-01", f"{end_year}-{end_month:02d}-28"


def initialize_earth_engine(project: str | None = None, authenticate: bool = False):
    try:
        import ee
    except ModuleNotFoundError as exc:
        raise EarthEngineSetupError(
            "Google Earth Engine is required for satellite extraction, but "
            "`earthengine-api` is not installed. Install it with:\n"
            "  pip install earthengine-api"
        ) from exc

    project_id = project or os.getenv("EE_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        return ee
    except Exception as exc:
        if authenticate:
            ee.Authenticate()
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
            return ee

        project_hint = (
            "Provide a Google Cloud project with `--ee-project YOUR_PROJECT_ID`, "
            "or set it once with `$env:EE_PROJECT='YOUR_PROJECT_ID'`."
            if not project_id
            else f"Earth Engine project being used: {project_id}"
        )
        raise EarthEngineSetupError(
            "Google Earth Engine is required for satellite extraction. "
            "Your local Earth Engine session is not initialized.\n\n"
            "Run these commands once in PowerShell:\n"
            "  earthengine authenticate\n"
            "  $env:EE_PROJECT='YOUR_PROJECT_ID'\n\n"
            f"{project_hint}\n\n"
            f"Original Earth Engine error: {exc}"
        ) from exc


def validate_earth_engine_setup(project: str | None = None) -> None:
    initialize_earth_engine(project, authenticate=False)


def add_cloud_score_mask(ee, s2_collection, region, start_date: str, end_date: str, threshold: float):
    cloud_score = (
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
    )

    joined = ee.Join.saveFirst("cloud_score").apply(
        primary=s2_collection,
        secondary=cloud_score,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    def mask_image(image):
        image = ee.Image(image)
        cs = ee.Image(image.get("cloud_score")).select(["cs", "cs_cdf"])
        return image.addBands(cs).updateMask(cs.select("cs_cdf").gte(threshold))

    return ee.ImageCollection(joined).map(mask_image)


def add_common_vegetation_indices(ee, image):
    blue = image.select("BLUE")
    green = image.select("GREEN")
    red = image.select("RED")
    nir = image.select("NIR")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    evi = (
        nir.subtract(red)
        .multiply(2.5)
        .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
        .rename("EVI")
    )
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
    return image.addBands([ndvi, evi, ndwi])


def mask_and_standardize_landsat(ee, image, source_bands: list[str]):
    qa = image.select("QA_PIXEL")
    clear = (
        qa.bitwiseAnd(1).eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )
    return (
        image.updateMask(clear)
        .updateMask(image.select("QA_RADSAT").eq(0))
        .select(source_bands)
        .multiply(0.0000275)
        .add(-0.2)
        .rename(COMMON_BANDS)
    )


def sentinel2_collection(ee, region, start_date: str, end_date: str, max_scene_cloud: int, cloud_score_threshold: float):
    raw_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_scene_cloud))
        .select(["B2", "B3", "B4", "B8"])
    )
    masked_collection = add_cloud_score_mask(
        ee, raw_collection, region, start_date, end_date, cloud_score_threshold
    )
    return raw_collection, masked_collection.map(
        lambda image: image.select(["B2", "B3", "B4", "B8"]).divide(10000).rename(COMMON_BANDS)
    ), "Sentinel-2 SR", 20


def landsat_collection(ee, year: int, region, start_date: str, end_date: str):
    def filtered(collection_id: str, source_bands: list[str]):
        return (
            ee.ImageCollection(collection_id)
            .filterDate(start_date, end_date)
            .filterBounds(region)
            .map(lambda image: mask_and_standardize_landsat(ee, image, source_bands))
        )

    if year <= 2012:
        bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4"]
        landsat5_t1 = filtered("LANDSAT/LT05/C02/T1_L2", bands)
        landsat5_t2 = filtered("LANDSAT/LT05/C02/T2_L2", bands)
        landsat7_t1 = filtered("LANDSAT/LE07/C02/T1_L2", bands)
        landsat7_t2 = filtered("LANDSAT/LE07/C02/T2_L2", bands)
        return (
            landsat5_t1.merge(landsat5_t2).merge(landsat7_t1).merge(landsat7_t2),
            "Landsat 5/7 SR T1/T2",
            30,
        )

    bands = ["SR_B2", "SR_B3", "SR_B4", "SR_B5"]
    landsat8_t1 = filtered("LANDSAT/LC08/C02/T1_L2", bands)
    landsat8_t2 = filtered("LANDSAT/LC08/C02/T2_L2", bands)
    return landsat8_t1.merge(landsat8_t2), "Landsat 8 SR T1/T2", 30


def download_file(url: str, path: Path) -> None:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(path, "wb") as file:
        file.write(resp.content)


def fetch_sentinel2_gee_features(
    state_name: str,
    lat: float,
    lon: float,
    crop: str,
    year: int,
    season: str | None = None,
    buffer_degree: float = 0.25,
    dimensions: int = 768,
    cloud_score_threshold: float = 0.6,
    max_scene_cloud: int = 70,
    ee_project: str | None = None,
) -> dict:
    print(f"[INFO] Initializing Earth Engine for {state_name}...")
    ee = initialize_earth_engine(ee_project)
    months = season_window(crop, season)
    start_date, end_date = date_range_for_season(year, months)
    print(f"[INFO] Searching satellite scenes from {start_date} to {end_date}...")

    region = ee.Geometry.Rectangle(
        [lon - buffer_degree, lat - buffer_degree, lon + buffer_degree, lat + buffer_degree]
    )

    using_sentinel = year >= 2017
    if using_sentinel:
        raw_collection, collection, satellite_source, scale = sentinel2_collection(
            ee, region, start_date, end_date, max_scene_cloud, cloud_score_threshold
        )
        image_count = raw_collection.size().getInfo()
        if image_count == 0:
            print(f"[INFO] No Sentinel-2 scenes; trying Landsat 8 for {state_name} {year}...")
            collection, satellite_source, scale = landsat_collection(
                ee, year, region, start_date, end_date
            )
            raw_collection = collection
            image_count = raw_collection.size().getInfo()
            using_sentinel = False
    else:
        collection, satellite_source, scale = landsat_collection(ee, year, region, start_date, end_date)
        raw_collection = collection
        image_count = raw_collection.size().getInfo()

    if image_count == 0:
        raise RuntimeError(f"No {satellite_source} scenes found for {state_name} in {start_date} to {end_date}")
    print(f"[INFO] Found {image_count} {satellite_source} scenes. Building cloud-masked composite...")

    composite = collection.map(lambda image: add_common_vegetation_indices(ee, image)).median().clip(region)

    stat_bands = ["NDVI", "EVI", "NDWI", "BLUE", "GREEN", "RED", "NIR"]
    reducer = ee.Reducer.mean().combine(
        reducer2=ee.Reducer.stdDev(), sharedInputs=True
    ).combine(
        reducer2=ee.Reducer.minMax(), sharedInputs=True
    )
    stats = composite.select(stat_bands).reduceRegion(
        reducer=reducer,
        geometry=region,
        scale=scale,
        bestEffort=True,
        maxPixels=1_000_000_000,
    ).getInfo()
    if using_sentinel and not any(value is not None for value in stats.values()):
        print(f"[INFO] Sentinel-2 composite has no valid pixels; trying Landsat 8 for {state_name} {year}...")
        collection, satellite_source, scale = landsat_collection(
            ee, year, region, start_date, end_date
        )
        raw_collection = collection
        image_count = raw_collection.size().getInfo()
        if image_count == 0:
            raise RuntimeError(f"No Landsat 8 scenes found for {state_name} in {start_date} to {end_date}")
        composite = collection.map(lambda image: add_common_vegetation_indices(ee, image)).median().clip(region)
        stats = composite.select(stat_bands).reduceRegion(
            reducer=reducer,
            geometry=region,
            scale=scale,
            bestEffort=True,
            maxPixels=1_000_000_000,
        ).getInfo()
    print("[INFO] Feature statistics calculated. Downloading RGB and NDVI previews...")

    prefix = f"{slug(state_name)}_{year}_{slug(crop)}_{slug(season or 'kharif')}"
    year_output_dir = OUTPUT_DIR / str(year)
    rgb_output_dir = year_output_dir / "rgb"
    ndvi_output_dir = year_output_dir / "ndvi"
    rgb_output_dir.mkdir(parents=True, exist_ok=True)
    ndvi_output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = rgb_output_dir / f"{prefix}_rgb.png"
    ndvi_path = ndvi_output_dir / f"{prefix}_ndvi.png"

    rgb_url = composite.getThumbURL(
        {
            "bands": ["RED", "GREEN", "BLUE"],
            "min": 0,
            "max": 0.3,
            "dimensions": dimensions,
            "region": region,
            "format": "png",
        }
    )
    ndvi_url = composite.getThumbURL(
        {
            "bands": ["NDVI"],
            "min": 0,
            "max": 0.9,
            "palette": ["8c510a", "d8b365", "f6e8c3", "c7eae5", "5ab4ac", "01665e"],
            "dimensions": dimensions,
            "region": region,
            "format": "png",
        }
    )

    download_file(rgb_url, rgb_path)
    download_file(ndvi_url, ndvi_path)

    row = {
        "State": state_name,
        "Crop": crop,
        "Year": year,
        "Season": season or "Kharif",
        "Latitude": lat,
        "Longitude": lon,
        "Start_Date": start_date,
        "End_Date": end_date,
        "Image_Count": image_count,
        "Satellite_Source": satellite_source,
        "Cloud_Score_Threshold": cloud_score_threshold,
        "Buffer_Degree": buffer_degree,
        "RGB_Image": str(rgb_path.relative_to(ROOT)),
        "NDVI_Image": str(ndvi_path.relative_to(ROOT)),
    }
    row.update({key: round(value, 6) if isinstance(value, (int, float)) else value for key, value in stats.items()})

    print(f"[SUCCESS] {state_name} {year} {crop}: RGB, NDVI, and features saved.")
    return row


def save_feature_rows(rows: list[dict], path: Path = FEATURES_PATH) -> None:
    if not rows:
        return

    new_df = pd.DataFrame(rows)
    if path.exists():
        old_df = pd.read_csv(path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["State", "Crop", "Year", "Season"], keep="last"
        ).sort_values(["State", "Crop", "Year"])
    else:
        merged = new_df.sort_values(["State", "Crop", "Year"])

    merged.to_csv(path, index=False)
    print(f"[SUCCESS] Harmonized satellite feature table saved to: {path}")


def target_key(state: str, crop: str, year: int, season: str | None) -> tuple[str, str, int, str]:
    if pd.isna(state) or pd.isna(crop) or pd.isna(year):
        raise ValueError("Target key fields State, Crop, and Year must not be empty.")

    return (
        str(state).strip().casefold(),
        str(crop).strip().casefold(),
        int(year),
        "" if pd.isna(season) else str(season).strip().casefold(),
    )


def target_group_key(state: str, crop: str, year: int) -> tuple[str, str, int]:
    if pd.isna(state) or pd.isna(crop) or pd.isna(year):
        raise ValueError("Target group key fields State, Crop, and Year must not be empty.")

    return (
        str(state).strip().casefold(),
        str(crop).strip().casefold(),
        int(year),
    )


def valid_target_key_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].dropna(subset=["State", "Crop", "Year"])


def completed_target_keys(path: Path = FEATURES_PATH) -> set[tuple[str, str, int, str]]:
    if not path.exists():
        return set()

    completed = pd.read_csv(path)
    required_columns = {"State", "Crop", "Year", "Season"}
    if not required_columns.issubset(completed.columns):
        return set()

    rows = valid_target_key_rows(completed, ["State", "Crop", "Year", "Season"])
    return {target_key(row.State, row.Crop, row.Year, row.Season) for row in rows.itertuples(index=False)}


def failed_target_keys(path: Path = FAILED_TARGETS_PATH) -> set[tuple[str, str, int, str]]:
    if not path.exists():
        return set()

    failed = pd.read_csv(path)
    required_columns = {"State", "Crop", "Year", "Season"}
    if not required_columns.issubset(failed.columns):
        return set()

    rows = valid_target_key_rows(failed, ["State", "Crop", "Year", "Season"])
    return {target_key(row.State, row.Crop, row.Year, row.Season) for row in rows.itertuples(index=False)}


def failed_target_group_keys(path: Path = FAILED_TARGETS_PATH) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()

    failed = pd.read_csv(path)
    required_columns = {"State", "Crop", "Year"}
    if not required_columns.issubset(failed.columns):
        return set()

    rows = valid_target_key_rows(failed, ["State", "Crop", "Year"])
    return {target_group_key(row.State, row.Crop, row.Year) for row in rows.itertuples(index=False)}


def save_failed_target(
    state: str,
    crop: str,
    year: int,
    season: str | None,
    reason: str,
    path: Path = FAILED_TARGETS_PATH,
) -> None:
    row = {
        "State": state,
        "Crop": crop,
        "Year": int(year),
        "Season": season or "Kharif",
        "Reason": str(reason).replace("\n", " ").strip(),
        "Failed_At": datetime.now().isoformat(timespec="seconds"),
    }
    new_df = pd.DataFrame([row])
    if path.exists():
        old_df = pd.read_csv(path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df

    merged = merged.drop_duplicates(
        subset=["State", "Crop", "Year", "Season"], keep="last"
    ).sort_values(["State", "Crop", "Year", "Season"])
    merged.to_csv(path, index=False)


def clear_failed_target(
    state: str,
    crop: str,
    year: int,
    season: str | None,
    path: Path = FAILED_TARGETS_PATH,
) -> None:
    if not path.exists():
        return

    failed = pd.read_csv(path)
    required_columns = {"State", "Crop", "Year", "Season"}
    if failed.empty or not required_columns.issubset(failed.columns):
        return

    key = target_key(state, crop, year, season)
    keep_mask = failed.apply(
        lambda row: target_key(row["State"], row["Crop"], row["Year"], row["Season"]) != key,
        axis=1,
    )
    failed.loc[keep_mask].to_csv(path, index=False)


def load_targets_from_crop_yield(
    crops: Iterable[str],
    years: Iterable[int] | None,
    seasons: Iterable[str] | None = None,
) -> pd.DataFrame:
    crop_yield_path = ROOT / "datasets" / "crop_yield.csv"
    df = pd.read_csv(crop_yield_path)
    df.columns = df.columns.str.strip()
    df["Crop"] = df["Crop"].str.strip()
    df["Season"] = df["Season"].str.strip()

    crop_set = {crop.casefold() for crop in crops}
    target = df[df["Crop"].str.casefold().isin(crop_set)].copy()
    if seasons:
        season_set = {season.strip().casefold() for season in seasons}
        target = target[target["Season"].str.casefold().isin(season_set)]
    if years:
        year_set = set(years)
        target = target[target["Crop_Year"].isin(year_set)]

    return target[["State", "Crop", "Crop_Year", "Season"]].drop_duplicates()


def load_targets_from_csv(path: Path) -> pd.DataFrame:
    target = pd.read_csv(path)
    target.columns = target.columns.str.strip()

    required_columns = ["State", "Crop", "Crop_Year", "Season"]
    missing_columns = [column for column in required_columns if column not in target.columns]
    if missing_columns:
        raise ValueError(
            f"{path} is missing required target columns: {', '.join(missing_columns)}"
        )

    for column in ["State", "Crop", "Season"]:
        target[column] = target[column].astype(str).str.strip()
    target["Crop_Year"] = pd.to_numeric(target["Crop_Year"], errors="coerce").astype("Int64")
    target = target.dropna(subset=required_columns)
    return target[required_columns].drop_duplicates()


def parse_years(years_arg: str | None) -> list[int] | None:
    if not years_arg:
        return None
    years: list[int] = []
    for part in years_arg.split(","):
        part = part.strip()
        if "-" in part:
            start, end = [int(value) for value in part.split("-", maxsplit=1)]
            years.extend(range(start, end + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch harmonized Landsat/Sentinel-2 composites and vegetation features."
    )
    parser.add_argument("--state", default="Punjab", help="State name, or ALL for every state.")
    parser.add_argument("--crop", default="Rice", help="Crop name.")
    parser.add_argument("--year", type=int, default=2020, help="Crop year.")
    parser.add_argument("--season", default=None, help="Season label such as Kharif, Rabi, or Whole Year.")
    parser.add_argument("--all-states", action="store_true", help="Process every state in state_coordinates.csv.")
    parser.add_argument("--from-crop-yield", action="store_true", help="Use crop_yield.csv state/crop/year/season rows.")
    parser.add_argument("--targets-csv", default=None, help="CSV containing State,Crop,Crop_Year,Season targets to fetch.")
    parser.add_argument("--crops", default="Rice,Maize", help="Comma-separated crops for --from-crop-yield.")
    parser.add_argument("--seasons", default=None, help="Comma-separated seasons for --from-crop-yield, e.g. Kharif,Rabi.")
    parser.add_argument("--years", default=None, help="Years for --from-crop-yield, e.g. 2018-2020 or 2020.")
    parser.add_argument("--buffer-degree", type=float, default=0.25, help="Region radius around the centroid.")
    parser.add_argument("--dimensions", type=int, default=768, help="Output image size in pixels.")
    parser.add_argument("--cloud-score-threshold", type=float, default=0.6, help="Cloud Score+ cs_cdf mask threshold.")
    parser.add_argument("--ee-project", default=None, help="Google Cloud project ID registered for Earth Engine.")
    parser.add_argument("--resume", action="store_true", help="Skip targets already saved in harmonized_satellite_features.csv.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry targets listed in failed_satellite_targets.csv.")
    args = parser.parse_args()

    coords_df = pd.read_csv(COORDS_PATH)

    if args.targets_csv:
        targets = load_targets_from_csv(Path(args.targets_csv))
    elif args.from_crop_yield:
        crops = [crop.strip() for crop in args.crops.split(",") if crop.strip()]
        seasons = [season.strip() for season in args.seasons.split(",") if season.strip()] if args.seasons else None
        years = parse_years(args.years)
        targets = load_targets_from_crop_yield(crops, years, seasons)
    else:
        states = coords_df["State"].tolist() if args.all_states or args.state.casefold() == "all" else [args.state]
        targets = pd.DataFrame(
            [
                {
                    "State": state,
                    "Crop": args.crop,
                    "Crop_Year": args.year,
                    "Season": args.season or "Kharif",
                }
                for state in states
            ]
        )

    if args.resume:
        completed = completed_target_keys()
        pending = []
        for _, target in targets.iterrows():
            key = target_key(
                str(target["State"]),
                str(target["Crop"]),
                int(target["Crop_Year"]),
                str(target["Season"]) if pd.notna(target["Season"]) else None,
            )
            if key not in completed:
                pending.append(target)
        targets = pd.DataFrame(pending, columns=targets.columns)
        print(f"[INFO] Resume mode: {len(completed)} completed targets found; {len(targets)} remain.")

    if not args.retry_failed:
        failed = failed_target_keys()
        failed_groups = failed_target_group_keys()
        pending = []
        skipped_failed = 0
        for _, target in targets.iterrows():
            key = target_key(
                str(target["State"]),
                str(target["Crop"]),
                int(target["Crop_Year"]),
                str(target["Season"]) if pd.notna(target["Season"]) else None,
            )
            group_key = target_group_key(
                str(target["State"]),
                str(target["Crop"]),
                int(target["Crop_Year"]),
            )
            if key in failed or group_key in failed_groups:
                skipped_failed += 1
            else:
                pending.append(target)
        targets = pd.DataFrame(pending, columns=targets.columns)
        if skipped_failed:
            print(
                f"[INFO] Skipping {skipped_failed} known failed target/year groups from "
                f"{FAILED_TARGETS_PATH.name}. Use --retry-failed to try them again."
            )

    if targets.empty:
        print("[INFO] No targets to process.")
        return

    try:
        validate_earth_engine_setup(args.ee_project)
    except EarthEngineSetupError as exc:
        parser.exit(status=1, message=f"[EARTH_ENGINE_SETUP_ERROR]\n{exc}\n")

    failed = failed_target_keys() if not args.retry_failed else set()
    failed_groups = failed_target_group_keys() if not args.retry_failed else set()

    for _, target in targets.iterrows():
        state_name = str(target["State"]).strip()
        state_row = coords_df[coords_df["State"].str.casefold() == state_name.casefold()]
        if state_row.empty:
            print(f"[SKIP] Coordinates not found for {state_name}")
            continue

        lat = float(state_row["Latitude"].iloc[0])
        lon = float(state_row["Longitude"].iloc[0])
        crop = str(target["Crop"]).strip()
        year = int(target["Crop_Year"])
        season = str(target["Season"]).strip() if pd.notna(target["Season"]) else None
        key = target_key(state_name, crop, year, season)
        group_key = target_group_key(state_name, crop, year)

        if not args.retry_failed and (key in failed or group_key in failed_groups):
            print(f"[SKIP] Known failed target: {state_name} {year} {crop} {season or 'Kharif'}")
            continue

        try:
            row = fetch_sentinel2_gee_features(
                state_name=state_name,
                lat=lat,
                lon=lon,
                crop=crop,
                year=year,
                season=season,
                buffer_degree=args.buffer_degree,
                dimensions=args.dimensions,
                cloud_score_threshold=args.cloud_score_threshold,
                ee_project=args.ee_project,
            )
            save_feature_rows([row])
            clear_failed_target(state_name, crop, year, season)
        except Exception as exc:
            print(f"[FAILED] {state_name} {year} {crop} {season or 'Kharif'}: {exc}")
            save_failed_target(state_name, crop, year, season, str(exc))
            failed.add(key)
            failed_groups.add(group_key)

if __name__ == "__main__":
    main()
