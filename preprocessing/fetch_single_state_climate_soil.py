import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
COORDS_PATH = ROOT / "datasets" / "state_coordinates.csv"
OUTPUT_DIR = ROOT / "datasets" / "test_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_single_state_climate_soil(
    state_name: str = "Punjab",
    crop: str = "Rice",
    year: int = 2020,
    season_months: tuple[int, int] = (6, 10),  # June to October (Kharif season)
) -> None:
    print(f"\n============================================================")
    print(f"FETCHING CLIMATE & SOIL HEATMAP: {state_name} | {crop} | {year}")
    print(f"============================================================")

    # 1. Get State Coordinates
    coords_df = pd.read_csv(COORDS_PATH)
    state_row = coords_df[coords_df["State"].str.casefold() == state_name.casefold()]

    if state_row.empty:
        print(f"[!] State '{state_name}' not found in {COORDS_PATH}")
        return

    lat = float(state_row["Latitude"].values[0])
    lon = float(state_row["Longitude"].values[0])
    print(f"[+] State Centroid: Latitude {lat}, Longitude {lon}")

    # 2. Fetch Weather Data from NASA POWER API (Daily Rainfall & Temperature)
    start_date = f"{year}{season_months[0]:02d}01"
    end_date = f"{year}{season_months[1]:02d}31"

    print(f"[+] Querying NASA POWER API for Kharif Season ({start_date} to {end_date})...")

    nasa_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "PRECTOTCORR,T2M_MAX,T2M_MIN",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }

    resp = requests.get(nasa_url, params=params, timeout=30)
    if resp.status_code != 200:
        print(f"[!] Failed to fetch NASA POWER data. Status: {resp.status_code}")
        return

    weather_data = resp.json()["properties"]["parameter"]
    rainfall = list(weather_data["PRECTOTCORR"].values())
    tmax = list(weather_data["T2M_MAX"].values())
    tmin = list(weather_data["T2M_MIN"].values())

    # Create a 2D spatial-temporal synthetic grid (64x64) representing environmental heatmaps
    grid_size = 64
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)

    # Spatial variations based on average season metrics
    total_rain = np.mean(rainfall)
    avg_tmax = np.mean(tmax)
    avg_tmin = np.mean(tmin)

    rain_channel = total_rain * (1 - 0.3 * r + 0.1 * np.sin(4 * xx))
    tmax_channel = avg_tmax * (1 + 0.1 * yy + 0.05 * np.cos(3 * xx))
    tmin_channel = avg_tmin * (1 + 0.08 * xx + 0.04 * np.sin(2 * yy))

    # 3. Fetch Soil Property from SoilGrids REST API (Organic Carbon stock at 0-30cm)
    print("[+] Querying SoilGrids API for Soil Organic Carbon...")
    soil_url = f"https://rest.isric.org/soilgrids/v2.0/properties/query"
    soil_params = {
        "lon": lon,
        "lat": lat,
        "property": "soc",
        "depth": "0-30cm",
        "value": "mean",
    }

    try:
        soil_resp = requests.get(soil_url, params=soil_params, timeout=15)
        if soil_resp.status_code == 200:
            soc_val = soil_resp.json()["properties"]["layers"][0]["depths"][0]["values"]["mean"]
            if soc_val is None:
                soc_val = 150
        else:
            soc_val = 150
    except Exception:
        soc_val = 150

    soc_channel = (soc_val / 10.0) * (1 - 0.2 * r)

    # Stack channels into a 4-channel tensor: (64, 64, 4)
    heatmap_tensor = np.stack(
        [rain_channel, tmax_channel, tmin_channel, soc_channel], axis=-1
    )

    # 4. Save Multi-Channel Heatmap Tensor (.npy)
    tensor_filename = (
        f"{state_name.lower().replace(' ', '_')}_{year}_{crop.lower()}_env_tensor.npy"
    )
    np.save(OUTPUT_DIR / tensor_filename, heatmap_tensor)
    print(f"[+] 4-Channel Environmental Tensor saved to: {OUTPUT_DIR / tensor_filename}")

    # 5. Render Composite Image Visual Plot (.png)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = [
        f"Rainfall ({total_rain:.1f} mm/day)",
        f"Max Temp ({avg_tmax:.1f} °C)",
        f"Min Temp ({avg_tmin:.1f} °C)",
        f"Soil Organic Carbon ({soc_val/10:.1f} g/kg)",
    ]
    cmaps = ["Blues", "YlOrRd", "Oranges", "YlGn"]

    for i in range(4):
        im = axes[i].imshow(heatmap_tensor[:, :, i], cmap=cmaps[i])
        axes[i].set_title(titles[i], fontsize=11, fontweight="bold")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], shrink=0.7)

    plt.tight_layout()
    img_filename = f"{state_name.lower().replace(' ', '_')}_{year}_{crop.lower()}_heatmap.png"
    plt.savefig(OUTPUT_DIR / img_filename, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] Environmental Heatmap Image saved to:")
    print(f"          {OUTPUT_DIR / img_filename}")


if __name__ == "__main__":
    fetch_single_state_climate_soil(state_name="Punjab", crop="Rice", year=2020)
