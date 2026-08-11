from pathlib import Path
import pandas as pd
import requests
import time


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "datasets" / "state_coordinates.csv"
OUTPUT_PATH = ROOT / "datasets" / "state_soil_lookup.csv"


# ============================================================
# SOILGRIDS API
# ============================================================

SOILGRIDS_URL = (
    "https://rest.isric.org/"
    "soilgrids/v2.0/properties/query"
)


# ============================================================
# SOIL PROPERTIES
# ============================================================

PROPERTIES = [
    "phh2o",
    "nitrogen",
    "soc",
    "clay",
    "sand",
    "silt"
]


# ============================================================
# GET SOIL DATA FOR ONE LOCATION
# ============================================================

def get_soil_data(latitude, longitude):

    params = [
        ("lon", longitude),
        ("lat", latitude),

        ("property", "phh2o"),
        ("property", "nitrogen"),
        ("property", "soc"),
        ("property", "clay"),
        ("property", "sand"),
        ("property", "silt"),

        ("depth", "0-5cm"),
        ("value", "mean")
    ]

    headers = {
        "User-Agent": "CropYieldPrediction/1.0"
    }

    response = requests.get(
        SOILGRIDS_URL,
        params=params,
        headers=headers,
        timeout=60
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# EXTRACT 0-5 CM VALUES
# ============================================================

def extract_soil_values(data):

    result = {
        "phh2o": None,
        "nitrogen": None,
        "soc": None,
        "clay": None,
        "sand": None,
        "silt": None
    }

    layers = data.get("properties", {}).get("layers", [])

    for layer in layers:

        name = layer.get("name")

        if name not in result:
            continue

        depths = layer.get("depths", [])

        for depth in depths:

            if depth.get("label") == "0-5cm":

                values = depth.get("values", {})

                mean_value = values.get("mean")

                if mean_value is not None:
                    result[name] = mean_value

                break

    return result


# ============================================================
# CONVERT SOILGRIDS VALUES TO CONVENTIONAL UNITS
# ============================================================

def convert_units(soil):

    # SoilGrids stores integer-scaled values.
    #
    # phh2o   : divide by 10 → pH
    # nitrogen: divide by 100 → g/kg
    # soc     : divide by 10 → g/kg
    # clay    : divide by 10 → %
    # sand    : divide by 10 → %
    # silt    : divide by 10 → %

    if soil["phh2o"] is not None:
        soil["phh2o"] = soil["phh2o"] / 10

    if soil["nitrogen"] is not None:
        soil["nitrogen"] = soil["nitrogen"] / 100

    if soil["soc"] is not None:
        soil["soc"] = soil["soc"] / 10

    if soil["clay"] is not None:
        soil["clay"] = soil["clay"] / 10

    if soil["sand"] is not None:
        soil["sand"] = soil["sand"] / 10

    if soil["silt"] is not None:
        soil["silt"] = soil["silt"] / 10

    return soil


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load state coordinates created by nomination.py
    # --------------------------------------------------------

    states = pd.read_csv(INPUT_PATH)

    print("Loaded state coordinates:")
    print(states)

    print("\nTotal states:", len(states))


    # --------------------------------------------------------
    # 2. Prepare output
    # --------------------------------------------------------

    soil_results = []


    # --------------------------------------------------------
    # 3. Query SoilGrids
    # --------------------------------------------------------

    for index, row in states.iterrows():

        state = row["State"]
        latitude = row["Latitude"]
        longitude = row["Longitude"]

        print("\n========================================")
        print(f"State: {state}")
        print(f"Latitude: {latitude}")
        print(f"Longitude: {longitude}")
        print("========================================")

        # Skip states without coordinates

        if pd.isna(latitude) or pd.isna(longitude):

            print("Skipping: coordinates missing")

            soil_results.append({
                "State": state,
                "Latitude": latitude,
                "Longitude": longitude,
                "phh2o": None,
                "nitrogen": None,
                "soc": None,
                "clay": None,
                "sand": None,
                "silt": None
            })

            continue


        try:

            print("Requesting SoilGrids...")

            data = get_soil_data(
                latitude,
                longitude
            )

            soil = extract_soil_values(data)

            soil = convert_units(soil)


            # Add geographic information

            soil["State"] = state
            soil["Latitude"] = latitude
            soil["Longitude"] = longitude

            soil_results.append(soil)


            print("Soil data:")
            print(soil)


        except requests.exceptions.HTTPError as e:

            print(
                f"SoilGrids HTTP error for {state}: {e}"
            )

            soil_results.append({
                "State": state,
                "Latitude": latitude,
                "Longitude": longitude,
                "phh2o": None,
                "nitrogen": None,
                "soc": None,
                "clay": None,
                "sand": None,
                "silt": None
            })


        except Exception as e:

            print(
                f"Error for {state}: {e}"
            )

            soil_results.append({
                "State": state,
                "Latitude": latitude,
                "Longitude": longitude,
                "phh2o": None,
                "nitrogen": None,
                "soc": None,
                "clay": None,
                "sand": None,
                "silt": None
            })


        # ----------------------------------------------------
        # SoilGrids fair-use limit
        # ----------------------------------------------------

        # ISRIC recommends max 5 API calls per minute.
        # Wait 13 seconds between requests.

        time.sleep(13)


    # --------------------------------------------------------
    # 4. Create soil lookup DataFrame
    # --------------------------------------------------------

    soil_df = pd.DataFrame(soil_results)


    # --------------------------------------------------------
    # 5. Arrange columns
    # --------------------------------------------------------

    soil_df = soil_df[
        [
            "State",
            "Latitude",
            "Longitude",
            "phh2o",
            "nitrogen",
            "soc",
            "clay",
            "sand",
            "silt"
        ]
    ]


    # --------------------------------------------------------
    # 6. Display final result
    # --------------------------------------------------------

    print("\n========================================")
    print("FINAL STATE SOIL LOOKUP")
    print("========================================")

    print(
        soil_df.to_string(index=False)
    )


    # --------------------------------------------------------
    # 7. Check missing soil values
    # --------------------------------------------------------

    print("\n========================================")
    print("MISSING VALUES")
    print("========================================")

    print(
        soil_df.isna().sum()
    )


    # --------------------------------------------------------
    # 8. Save
    # --------------------------------------------------------

    soil_df.to_csv(
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