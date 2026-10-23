"""Build an auditable image-to-yield manifest and chronological split.

Run from the repository root with:
    python preprocessing/build_cnn_manifest.py

The split matches the tabular baseline: through 2016 train, 2017-2018
validation, and 2019 onward test. The partition is assigned by year, which
keeps every state/crop/season record in a year together and prevents
same-year leakage. Rows with uncertain targets, missing images, or image
paths reused for different target keys remain in the manifest but are not
marked usable.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
FEATURES_PATH = DATASETS / "harmonized_satellite_features.csv"
TARGETS_PATH = DATASETS / "crop_yield.csv"
MANIFEST_PATH = DATASETS / "cnn_image_yield_manifest.csv"
GROUP_SUMMARY_PATH = DATASETS / "cnn_grouped_split_summary.csv"

KEY_COLUMNS = ["state_key", "crop_key", "year_key", "season_key"]


def norm(value: object) -> str:
    """Normalize case and punctuation without guessing aliases."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def assign_split(year: int) -> str:
    if year <= 2016:
        return "train"
    if year <= 2018:
        return "validation"
    return "test"


def resolve_image_path(value: object) -> Path | None:
    if pd.isna(value) or not str(value).strip():
        return None
    # Paths in the CSV are repository-relative Windows paths. Path accepts
    # these natively on Windows; replacing separators also keeps this portable.
    relative = str(value).strip().replace("\\", "/")
    return ROOT.joinpath(*Path(relative).parts)


def main() -> None:
    features = pd.read_csv(FEATURES_PATH)
    targets = pd.read_csv(TARGETS_PATH)
    targets = targets[targets["Crop"].map(norm).isin({"rice", "maize", "moong green gram", "urad"})].copy()

    for frame, crop_col, year_col in (
        (features, "Crop", "Year"),
        (targets, "Crop", "Crop_Year"),
    ):
        frame["state_key"] = frame["State"].map(norm)
        frame["crop_key"] = frame[crop_col].map(norm)
        frame["season_key"] = frame["Season"].map(norm)
        frame["year_key"] = pd.to_numeric(frame[year_col], errors="coerce").astype("Int64")

    target_lookup: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in targets[KEY_COLUMNS + ["Yield"]].itertuples(index=False, name=None):
        key, value = tuple(row[:4]), row[4]
        if pd.notna(value):
            target_lookup[key].append(float(value))

    # A path tied to more than one target key is unsafe: this commonly occurs
    # when a seasonless image was reused for multiple seasonal yield records.
    path_keys: dict[str, set[tuple[object, ...]]] = defaultdict(set)
    for row in features.itertuples(index=False):
        key = tuple(getattr(row, c) for c in KEY_COLUMNS)
        for column in ("RGB_Image", "NDVI_Image"):
            path = getattr(row, column, None)
            if pd.notna(path) and str(path).strip():
                path_keys[str(path).replace("\\", "/")].add(key)

    records: list[dict[str, object]] = []
    for feature_row, row in enumerate(features.itertuples(index=False), start=2):
        key = tuple(getattr(row, c) for c in KEY_COLUMNS)
        key_text = "|".join(map(str, key))
        raw_values = target_lookup.get(key, [])
        distinct_yields = sorted(set(raw_values))
        target_status = (
            "matched" if len(distinct_yields) == 1
            else "missing_yield_match" if not distinct_yields
            else "ambiguous_yield_rows"
        )
        yield_value = distinct_yields[0] if len(distinct_yields) == 1 else None
        year = int(row.year_key) if pd.notna(row.year_key) else None
        split = assign_split(year) if year is not None else "unassigned"
        state = str(row.State).strip()
        crop = str(row.Crop).strip()
        season = str(row.Season).strip()
        group_id = f"{norm(state)}|{year}" if year is not None else ""

        for modality, column in (("rgb", "RGB_Image"), ("ndvi", "NDVI_Image")):
            raw_path = getattr(row, column, None)
            path = str(raw_path).replace("\\", "/") if pd.notna(raw_path) else ""
            resolved = resolve_image_path(raw_path)
            exists = bool(resolved and resolved.is_file())
            reused = bool(path and len(path_keys[path]) > 1)
            reasons = []
            if target_status != "matched":
                reasons.append(target_status)
            if not exists:
                reasons.append("missing_image")
            if reused:
                reasons.append("image_path_reused_across_target_keys")
            records.append({
                "feature_csv_row": feature_row,
                "feature_key": key_text,
                "state": state,
                "crop": crop,
                "year": year,
                "season": season,
                "group_id_state_year": group_id,
                "modality": modality,
                "image_path": path,
                "image_exists": exists,
                "path_reused_across_target_keys": reused,
                "target_match_status": target_status,
                "yield": yield_value,
                "yield_unit": "as recorded in crop_yield.csv",
                "split": split,
                "cnn_eligible": not reasons,
                "exclusion_reason": ";".join(reasons),
            })

    manifest = pd.DataFrame(records)
    manifest.to_csv(MANIFEST_PATH, index=False)

    # Summary at the state-year group level, plus modality counts. The year
    # partition makes a leakage audit straightforward and reproducible.
    group_rows = []
    for (group_id, year, split), group in manifest.groupby(
        ["group_id_state_year", "year", "split"], dropna=False
    ):
        group_rows.append({
            "group_id_state_year": group_id,
            "year": year,
            "split": split,
            "image_records": len(group),
            "eligible_image_records": int(group["cnn_eligible"].sum()),
            "states": ",".join(sorted(group["state"].dropna().unique())),
        })
    groups = pd.DataFrame(group_rows)
    groups.to_csv(GROUP_SUMMARY_PATH, index=False)

    print(f"Feature rows: {len(features)}")
    print(f"Manifest image records: {len(manifest)}")
    print(f"Exact-key target matches: {int(manifest['target_match_status'].eq('matched').sum())} image records")
    print(f"Image records with files present: {int(manifest['image_exists'].sum())}")
    print(f"Image records with reused paths across targets: {int(manifest['path_reused_across_target_keys'].sum())}")
    print(f"CNN-eligible image records: {int(manifest['cnn_eligible'].sum())}")
    print("Eligibility exclusions:", dict(Counter(
        reason for value in manifest["exclusion_reason"] if value
        for reason in value.split(";")
    )))
    print("Split image-record totals:", manifest.groupby("split").size().to_dict())
    print("Split eligible totals:", manifest.groupby("split")["cnn_eligible"].sum().astype(int).to_dict())
    print(f"State-year groups: {len(groups)}; groups per split: {groups.groupby('split').size().to_dict()}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"Group summary: {GROUP_SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
