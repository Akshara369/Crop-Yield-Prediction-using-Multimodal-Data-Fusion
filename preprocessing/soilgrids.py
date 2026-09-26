from pathlib import Path

import pandas as pd
import requests
from io import BytesIO
import rasterio


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "datasets" / "state_coordinates.csv"
OUTPUT_PATH = ROOT / "datasets" / "state_soil_lookup.csv"


# ============================================================
# SOILGRIDS WCS
# ============================================================

WCS_BASE_URL = "https://maps.isric.org/mapserv"

PROPERTIES = [
    "phh2o",
    "nitrogen",
    "soc",
    "clay",
    "sand",
    "silt"
]

SEARCH_WINDOWS = [
    0.01,
    0.03,
    0.05,
    0.10,
    0.20
]


# ============================================================
# SOILGRIDS CONVERSION FACTORS
# ============================================================

CONVERSION_FACTORS = {
    "phh2o": 10,
    "nitrogen": 100,
    "soc": 10,
    "clay": 10,
    "sand": 10,
    "silt": 10
}


# ============================================================
# GET WCS COVERAGE
# ============================================================

def get_soil_property(
    property_name,
    latitude,
    longitude
):

    map_path = f"/map/{property_name}.map"

    coverage_id = (
        f"{property_name}_0-5cm_Q0.5"
    )

    for window in SEARCH_WINDOWS:

        params = {
            "SERVICE": "WCS",
            "VERSION": "2.0.1",
            "REQUEST": "GetCoverage",
            "COVERAGEID": coverage_id,

            # Return a GeoTIFF
            "FORMAT": "GEOTIFF_INT16",

            # Area around the state point
            "SUBSET": [
                f"X({longitude - window},{longitude + window})",
                f"Y({latitude - window},{latitude + window})"
            ],

            "SUBSETTINGCRS":
                "http://www.opengis.net/def/crs/EPSG/0/4326",

            "OUTPUTCRS":
                "http://www.opengis.net/def/crs/EPSG/0/4326"
        }


        url = WCS_BASE_URL

        params["map"] = map_path


        response = requests.get(
            url,
            params=params,
            timeout=120
        )

        response.raise_for_status()


        # ----------------------------------------------------
        # Read returned GeoTIFF directly from memory
        # ----------------------------------------------------

        with rasterio.open(
            BytesIO(response.content)
        ) as dataset:

            array = dataset.read(1)

            nodata = dataset.nodata

            # Remove NoData pixels
            if nodata is not None:

                valid = array[array != nodata]

            else:

                valid = array.reshape(-1)


            # Some tiny urban windows return zero-filled
            # untagged NoData. Do not treat those as soil.
            valid = valid[valid > 0]


            if valid.size == 0:
                continue


            # Use the mean of the valid area
            value = float(valid.mean())


        # ----------------------------------------------------
        # Convert SoilGrids stored integer value
        # ----------------------------------------------------

        conversion_factor = (
            CONVERSION_FACTORS[property_name]
        )

        value = value / conversion_factor


        return value


    return None


# ============================================================
# GET ALL SOIL PROPERTIES FOR ONE STATE
# ============================================================

def get_state_soil(
    state,
    latitude,
    longitude
):

    result = {}

    print(
        f"\nGetting soil data for {state}"
    )

    print(
        f"Location: {latitude}, {longitude}"
    )


    for property_name in PROPERTIES:

        print(
            f"  → {property_name}"
        )

        try:

            value = get_soil_property(
                property_name,
                latitude,
                longitude
            )

            result[property_name] = value

            print(
                f"     {value}"
            )

        except Exception as e:

            print(
                f"     ERROR: {e}"
            )

            result[property_name] = None


    return result


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load existing state coordinates
    # --------------------------------------------------------

    states = pd.read_csv(
        INPUT_PATH
    )


    print(
        "Loaded state coordinates:"
    )

    print(states)

    print(
        f"\nTotal states: {len(states)}"
    )


    # --------------------------------------------------------
    # 2. Store results
    # --------------------------------------------------------

    soil_results = []


    # --------------------------------------------------------
    # 3. Process every state
    # --------------------------------------------------------

    for _, row in states.iterrows():

        state = row["State"]

        latitude = row["Latitude"]

        longitude = row["Longitude"]


        # Check coordinates

        if pd.isna(latitude) or pd.isna(longitude):

            print(
                f"\nSkipping {state}: "
                "coordinates missing"
            )

            continue


        # ----------------------------------------------------
        # Get soil properties
        # ----------------------------------------------------

        soil = get_state_soil(
            state,
            latitude,
            longitude
        )


        # ----------------------------------------------------
        # Add state information
        # ----------------------------------------------------

        result = {

            "State": state,

            "Latitude": latitude,

            "Longitude": longitude,

            "phh2o": soil["phh2o"],

            "nitrogen": soil["nitrogen"],

            "soc": soil["soc"],

            "clay": soil["clay"],

            "sand": soil["sand"],

            "silt": soil["silt"]

        }


        soil_results.append(result)


    # --------------------------------------------------------
    # 4. Create DataFrame
    # --------------------------------------------------------

    soil_df = pd.DataFrame(
        soil_results
    )


    # --------------------------------------------------------
    # 5. Display results
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "FINAL STATE SOIL LOOKUP"
    )

    print(
        "========================================"
    )

    print(
        soil_df.to_string(index=False)
    )


    # --------------------------------------------------------
    # 6. Check missing values
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "MISSING VALUES"
    )

    print(
        "========================================"
    )

    print(
        soil_df.isna().sum()
    )


    # --------------------------------------------------------
    # 7. Save
    # --------------------------------------------------------

    soil_df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print(
        "\n========================================"
    )

    print(
        "FILE SAVED"
    )

    print(
        "========================================"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
