from pathlib import Path
import pandas as pd
import requests
import time


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "datasets" / "crop_yield.csv"
OUTPUT_PATH = ROOT / "datasets" / "state_coordinates.csv"


# ============================================================
# GET STATE COORDINATES USING NOMINATIM
# ============================================================

def get_state_coordinates(state):

    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": f"{state}, India",
        "format": "json",
        "limit": 1
    }

    headers = {
        "User-Agent": "CropYieldPrediction/1.0"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if len(data) == 0:
        return None, None

    return float(data[0]["lat"]), float(data[0]["lon"])


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load the same original dataset used by
    #    extract_rice_maize_locations.py
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_PATH)

    df.columns = df.columns.str.strip()

    # Clean Crop and State
    for column in ["Crop", "State"]:
        df[column] = df[column].astype(str).str.strip()


    # --------------------------------------------------------
    # 2. Filter only Rice and Maize
    # --------------------------------------------------------

    filtered = df[
        df["Crop"].str.casefold().isin(["rice", "maize"])
    ].copy()


    # --------------------------------------------------------
    # 3. Get unique states
    # --------------------------------------------------------

    states = sorted(
        filtered["State"].dropna().unique()
    )


    print(f"Rows for Rice/Maize: {len(filtered)}")
    print(f"Unique states for soil lookup: {len(states)}")

    print("\nStates to geocode:")

    for state in states:
        print(state)


    # --------------------------------------------------------
    # 4. Create state coordinate table
    # --------------------------------------------------------

    state_coordinates = []


    # --------------------------------------------------------
    # 5. Query Nominatim for every unique state
    # --------------------------------------------------------

    for state in states:

        print(f"\nGetting coordinates for: {state}")

        try:

            latitude, longitude = get_state_coordinates(state)

            print(f"Latitude : {latitude}")
            print(f"Longitude: {longitude}")

            state_coordinates.append({
                "State": state,
                "Latitude": latitude,
                "Longitude": longitude
            })

        except Exception as e:

            print(f"Error for {state}: {e}")

            state_coordinates.append({
                "State": state,
                "Latitude": None,
                "Longitude": None
            })

        # Wait before next Nominatim request
        time.sleep(1)


    # --------------------------------------------------------
    # 6. Convert to DataFrame
    # --------------------------------------------------------

    states_df = pd.DataFrame(state_coordinates)


    # --------------------------------------------------------
    # 7. Display results
    # --------------------------------------------------------

    print("\n========================================")
    print("STATE COORDINATES")
    print("========================================")

    print(states_df.to_string(index=False))


    # --------------------------------------------------------
    # 8. Check for failed states
    # --------------------------------------------------------

    failed = states_df[
        states_df["Latitude"].isna() |
        states_df["Longitude"].isna()
    ]


    if len(failed) > 0:

        print("\n========================================")
        print("FAILED STATES")
        print("========================================")

        print(failed.to_string(index=False))

    else:

        print("\nAll states successfully geocoded.")


    # --------------------------------------------------------
    # 9. Save state coordinate lookup
    # --------------------------------------------------------

    states_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n========================================")
    print("FILE SAVED")
    print("========================================")

    print(OUTPUT_PATH)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()