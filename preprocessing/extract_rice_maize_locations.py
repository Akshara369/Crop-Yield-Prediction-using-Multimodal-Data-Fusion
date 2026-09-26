from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "datasets" / "crop_yield.csv"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df.columns = df.columns.str.strip()

    for column in ["Crop", "State"]:
        df[column] = df[column].astype(str).str.strip()

    filtered = df[df["Crop"].str.casefold().isin(["rice", "maize"])].copy()
    states = sorted(filtered["State"].unique())

    print(f"Rows for Rice/Maize: {len(filtered)}")
    print(f"Unique states for satellite/soil-image lookup: {len(states)}")
    print("\nAll states:")
    for state in states:
        print(state)

    print("\nRice states:")
    print(", ".join(sorted(filtered.loc[filtered["Crop"].eq("Rice"), "State"].unique())))

    print("\nMaize states:")
    print(", ".join(sorted(filtered.loc[filtered["Crop"].eq("Maize"), "State"].unique())))


if __name__ == "__main__":
    main()
