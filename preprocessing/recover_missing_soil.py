from io import BytesIO
from pathlib import Path

import pandas as pd
import rasterio
import requests


ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = ROOT / "datasets" / "state_soil_lookup.csv"
WCS_URL = "https://maps.isric.org/mapserv"

PROPERTIES = [
    "phh2o",
    "nitrogen",
    "soc",
    "clay",
    "sand",
    "silt",
]

CONVERSION_FACTORS = {
    "phh2o": 10,
    "nitrogen": 100,
    "soc": 10,
    "clay": 10,
    "sand": 10,
    "silt": 10,
}

SEARCH_WINDOWS = [
    0.01,
    0.03,
    0.05,
    0.10,
    0.20,
]


def get_soil_property(property_name, latitude, longitude):
    for window in SEARCH_WINDOWS:
        params = [
            ("map", f"/map/{property_name}.map"),
            ("SERVICE", "WCS"),
            ("VERSION", "2.0.1"),
            ("REQUEST", "GetCoverage"),
            ("COVERAGEID", f"{property_name}_0-5cm_Q0.5"),
            ("FORMAT", "GEOTIFF_INT16"),
            ("SUBSET", f"X({longitude - window},{longitude + window})"),
            ("SUBSET", f"Y({latitude - window},{latitude + window})"),
            ("SUBSETTINGCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
            ("OUTPUTCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
        ]

        response = requests.get(WCS_URL, params=params, timeout=120)
        response.raise_for_status()

        with rasterio.open(BytesIO(response.content)) as dataset:
            array = dataset.read(1)
            nodata = dataset.nodata

            if nodata is not None:
                valid = array[array != nodata]
            else:
                valid = array.reshape(-1)

            # Delhi's tiny city-center coverage can come back as all zeroes.
            # SoilGrids NoData is not always tagged, so ignore zero-only windows.
            valid = valid[valid > 0]

            if valid.size == 0:
                print(f"  {property_name}: no valid pixels at +/- {window} degrees")
                continue

            raw_value = float(valid.mean())
            value = raw_value / CONVERSION_FACTORS[property_name]
            print(f"  {property_name}: {value} from +/- {window} degrees")
            return value

    return None


def main():
    df = pd.read_csv(FILE_PATH)

    delhi = df[df["State"].str.strip().str.casefold() == "delhi"]
    if delhi.empty:
        print("Delhi not found.")
        return

    index = delhi.index[0]
    latitude = df.at[index, "Latitude"]
    longitude = df.at[index, "Longitude"]

    print("State: Delhi")
    print("Latitude:", latitude)
    print("Longitude:", longitude)

    recovered = {}
    for property_name in PROPERTIES:
        recovered[property_name] = get_soil_property(
            property_name,
            latitude,
            longitude,
        )

    missing = [name for name, value in recovered.items() if value is None]
    if missing:
        print("\nCould not retrieve:", ", ".join(missing))
        return

    for property_name, value in recovered.items():
        df.at[index, property_name] = value

    df.to_csv(FILE_PATH, index=False)
    print("\nDelhi soil values successfully updated.")


if __name__ == "__main__":
    main()
