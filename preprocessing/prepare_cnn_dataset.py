"""Create paired CNN samples and aligned train/validation/test CSVs."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
MODELS = ROOT / "models"
MANIFEST_PATH = DATASETS / "cnn_image_yield_manifest.csv"
MODELING_DATASET_PATH = DATASETS / "modeling_dataset.csv"
TABULAR_PREDICTIONS_PATH = MODELS / "tabular_baseline_test_predictions.csv"
SAMPLES_PATH = DATASETS / "cnn_samples.csv"
SPLIT_PATHS = {
    "train": DATASETS / "cnn_train.csv",
    "validation": DATASETS / "cnn_validation.csv",
    "test": DATASETS / "cnn_test.csv",
}
MAX_REASONABLE_YIELD = 15.0


def normalize(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def add_sample_key(frame: pd.DataFrame, year_column: str) -> pd.Series:
    return frame.apply(
        lambda row: "|".join(
            (
                normalize(row["State"] if "State" in row else row["state"]),
                normalize(row["Crop"] if "Crop" in row else row["crop"]),
                str(int(row[year_column])),
                normalize(row["Season"] if "Season" in row else row["season"]),
            )
        ),
        axis=1,
    )


def assign_split(year: int) -> str:
    if year <= 2016:
        return "train"
    if year <= 2018:
        return "validation"
    return "test"


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    eligible = manifest.loc[manifest["cnn_eligible"]].copy()
    if eligible.empty:
        raise ValueError(f"No eligible image records found in {MANIFEST_PATH}")

    duplicates = eligible.duplicated(["feature_key", "modality"], keep=False)
    if duplicates.any():
        examples = eligible.loc[duplicates, ["feature_key", "modality"]].head()
        raise ValueError(
            "Eligible manifest contains duplicate feature_key/modality rows:\n"
            f"{examples.to_string(index=False)}"
        )

    paired_paths = eligible.pivot(
        index="feature_key", columns="modality", values="image_path"
    )
    if not {"rgb", "ndvi"}.issubset(paired_paths.columns):
        raise ValueError("Manifest must contain both RGB and NDVI modalities.")
    paired_paths = paired_paths.dropna(subset=["rgb", "ndvi"]).reset_index()

    metadata_columns = [
        "feature_key",
        "state",
        "crop",
        "year",
        "season",
        "yield",
        "yield_unit",
    ]
    metadata = eligible[metadata_columns].drop_duplicates()
    if metadata["feature_key"].duplicated().any():
        raise ValueError("A feature_key maps to conflicting sample metadata.")

    samples = metadata.merge(paired_paths, on="feature_key", validate="one_to_one")
    samples["year"] = samples["year"].astype(int)
    samples["split"] = samples["year"].map(assign_split)

    modeling = pd.read_csv(MODELING_DATASET_PATH)
    modeling["feature_key"] = add_sample_key(modeling, "Year")
    if modeling["feature_key"].duplicated().any():
        raise ValueError("Modeling dataset has duplicate state/crop/year/season keys.")
    modeling_columns = ["feature_key", "Yield"]
    samples = samples.merge(
        modeling[modeling_columns].rename(columns={"Yield": "tabular_yield"}),
        on="feature_key",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = samples["_merge"] != "both"
    if unmatched.any():
        examples = samples.loc[unmatched, "feature_key"].head(5).tolist()
        print(
            f"[WARN] {int(unmatched.sum())} image samples have no match in the "
            f"tabular modeling dataset and will be skipped; examples: {examples}"
        )
        samples = samples.loc[~unmatched].copy()
    if not (samples["yield"] - samples["tabular_yield"]).abs().le(1e-8).all():
        raise ValueError("Manifest and modeling-dataset target values disagree.")
    samples = samples.drop(columns=["_merge", "tabular_yield"])

    outliers = (samples["yield"] < 0) | (samples["yield"] > MAX_REASONABLE_YIELD)
    excluded_outliers = int(outliers.sum())
    samples = samples.loc[~outliers].copy()

    predictions = pd.read_csv(TABULAR_PREDICTIONS_PATH)
    predictions["feature_key"] = add_sample_key(predictions, "Year")
    if predictions["feature_key"].duplicated().any():
        raise ValueError("Tabular test predictions have duplicate sample keys.")
    prediction_columns = ["feature_key", "Predicted_Yield"]
    test_mask = samples["split"].eq("test")
    test_keys = set(samples.loc[test_mask, "feature_key"])
    test_predictions = predictions.loc[
        predictions["feature_key"].isin(test_keys), prediction_columns
    ]
    missing_predictions = test_keys - set(test_predictions["feature_key"])
    if missing_predictions:
        print(
            f"[WARN] Tabular benchmark is missing predictions for {len(missing_predictions)} "
            f"CNN test samples; these will have NaN for tabular_predicted_yield."
        )
    samples = samples.merge(
        test_predictions.rename(
            columns={"Predicted_Yield": "tabular_predicted_yield"}
        ),
        on="feature_key",
        how="left",
        validate="one_to_one",
    )

    output_columns = [
        "feature_key",
        "state",
        "crop",
        "year",
        "season",
        "yield",
        "split",
        "rgb",
        "ndvi",
        "tabular_predicted_yield",
    ]
    samples = samples[output_columns].sort_values(
        ["year", "state", "crop", "season"]
    )
    samples.to_csv(SAMPLES_PATH, index=False)
    for split, path in SPLIT_PATHS.items():
        samples.loc[samples["split"].eq(split)].to_csv(path, index=False)

    print(f"Complete RGB/NDVI pairs: {len(paired_paths)}")
    print(f"Excluded implausible yields outside [0, {MAX_REASONABLE_YIELD}]: {excluded_outliers}")
    print(f"CNN samples matching tabular dataset: {len(samples)}")
    print("Samples per split:", samples.groupby("split").size().to_dict())
    print("Samples per split and year:")
    print(samples.groupby(["split", "year"]).size().to_string())
    print(f"Wrote {SAMPLES_PATH.relative_to(ROOT)} and split CSVs.")


if __name__ == "__main__":
    main()
