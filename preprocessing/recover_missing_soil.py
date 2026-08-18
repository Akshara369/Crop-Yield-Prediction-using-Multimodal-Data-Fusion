from pathlib import Path
from io import BytesIO

import pandas as pd
import requests
import rasterio


ROOT = Path(__file__).resolve().parents[1]

FILE_PATH = ROOT / "datasets" / "state_soil_lookup.csv"

WCS_URL = "https://maps.isric.org/mapserv"


def get_tripura_sand(latitude, longitude):

    params = [
        ("map", "/map/sand.map"),
        ("SERVICE", "WCS"),
        ("VERSION", "2.0.1"),
        ("REQUEST", "GetCoverage"),
        ("COVERAGEID", "sand_0-5cm_Q0.5"),
        ("FORMAT", "GEOTIFF_INT16"),

        (
            "SUBSET",
            f"X({longitude - 0.01},{longitude + 0.01})"
        ),

        (
            "SUBSET",
            f"Y({latitude - 0.01},{latitude + 0.01})"
        ),

        (
            "SUBSETTINGCRS",
            "http://www.opengis.net/def/crs/EPSG/0/4326"
        ),

        (
            "OUTPUTCRS",
            "http://www.opengis.net/def/crs/EPSG/0/4326"
        )
    ]

    response = requests.get(
        WCS_URL,
        params=params,
        timeout=120
    )

    print("HTTP status:", response.status_code)
    print("Content-Type:", response.headers.get("Content-Type"))

    response.raise_for_status()

    with rasterio.open(
        BytesIO(response.content)
    ) as dataset:

        array = dataset.read(1)

        nodata = dataset.nodata

        if nodata is not None:
            valid = array[array != nodata]
        else:
            valid = array

        print("Valid pixels:", valid.size)

        if valid.size == 0:
            return None

        raw_value = float(valid.mean())

    # SoilGrids sand scale factor
    sand = raw_value / 10

    return sand


def main():

    df = pd.read_csv(FILE_PATH)

    tripura = df[
        df["State"].str.strip().str.casefold() == "tripura"
    ]

    if tripura.empty:
        print("Tripura not found.")
        return

    index = tripura.index[0]

    latitude = df.at[index, "Latitude"]
    longitude = df.at[index, "Longitude"]

    print("State: Tripura")
    print("Latitude:", latitude)
    print("Longitude:", longitude)

    sand = get_tripura_sand(
        latitude,
        longitude
    )

    print("Retrieved sand:", sand)

    if sand is not None:

        df.at[index, "sand"] = sand

        df.to_csv(
            FILE_PATH,
            index=False
        )

        print("\nTripura sand value successfully updated.")

    else:

        print("\nCould not retrieve Tripura sand value.")


if __name__ == "__main__":
    main()