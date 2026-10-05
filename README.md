# Crop Yield Prediction Using Multimodal Data Fusion

A crop-yield prediction project focused on rice and maize cultivation in India, combining tabular agricultural data with geospatial and satellite-derived features. The repository includes a Streamlit dashboard for exploratory analysis, state-wise visualization, and a prediction interface backed by the trained tabular baseline model, with a rule-based fallback if the model artifact is unavailable.

## Overview

This project aims to estimate crop yield by fusing multiple data sources:

- Historical crop production and yield records
- Climate and agronomic variables such as rainfall, fertilizer, and pesticide usage
- Soil properties from soil-profile lookup data
- Sentinel-2 derived vegetation and spectral features
- State-level geographic metadata for spatial exploration

The app is designed to showcase the project pipeline and provide an interactive way to inspect the datasets and estimate yield under different field conditions.

## Key Features

- Crop yield dashboard with filtering by crop, state, and year
- Trend analysis for yield and production over time
- Spatial map of Indian states and geolocation metadata
- Satellite imagery explorer for Sentinel-2 RGB and NDVI visualizations
- Soil profile summaries for selected states
- Yield prediction form for Rice and Maize inputs
- Project architecture and pipeline documentation in the app UI

## Project Structure

```text
.
├── app.py                           # Main Streamlit application
├── maize_data.ipynb                # Exploratory notebook for maize analysis
├── datasets/
│   ├── crop_yield.csv              # Main crop yield dataset
│   ├── state_coordinates.csv       # State latitude/longitude metadata
│   ├── state_soil_lookup.csv       # Soil property references
│   ├── sentinel2_features.csv      # Sentinel-2 feature summaries
│   ├── rice_data.csv               # Rice raw data
│   ├── maize_data.csv              # Maize raw data
│   ├── rice_data_preprocessed.csv  # Preprocessed rice dataset
│   ├── maize_data_preprocessed.csv  # Preprocessed maize dataset
│   ├── rice_data_cleaned.csv       # Cleaned rice dataset
│   ├── maize_data_cleaned.csv       # Cleaned maize dataset
│   └── test_images/                # Cached RGB and NDVI satellite samples
├── preprocessing/
│   ├── fetch_single_state_satellite.py
│   ├── fetch_single_state_climate_soil.py
│   ├── extract_rice_maize_locations.py
│   ├── preprocess_rice_data.py
│   ├── recover_missing_soil.py
│   ├── soilgrids.py
│   ├── nomination.py
│   ├── view_npy.py
│   ├── Crop_Separation.ipynb
│   └── test_earth_engine.py
└── README.md
```

## Data Sources

The repository uses a multimodal feature set based on agricultural and remote-sensing data:

- Crop yield and production data: `datasets/crop_yield.csv`
- State coordinates: `datasets/state_coordinates.csv`
- Soil properties: `datasets/state_soil_lookup.csv`
- Sentinel-2 spectral features: `datasets/sentinel2_features.csv`
- Satellite sample visualizations: `datasets/test_images/`

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- Jupyter Notebook

## Setup

1. Clone the repository.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

3. Install the required Python libraries:

```bash
pip install streamlit pandas numpy matplotlib plotly scikit-learn joblib
```

## Run the App

From the project root:

```bash
streamlit run app.py
```

This will launch the Streamlit dashboard in the browser and open the navigation pages for project overview, spatial data exploration, and prediction.

## Project Workflow

1. Collect agricultural records and spatial metadata.
2. Merge crop, state, soil, and climate variables.
3. Download and process Sentinel-2 imagery and derived vegetation indices.
4. Clean and preprocess data for rice and maize separately.
5. Explore feature relationships and correlations in the dashboard.
6. Train predictive models and compare baselines.
7. Extend the dashboard to use a trained multimodal regression model in place of the current fallback logic.

## Tabular Baseline Model

The tabular modeling pipeline builds a rice/maize training dataset by merging crop-yield records with harmonized satellite features, then compares several regression baselines.

Build the modeling dataset:

```bash
python preprocessing/build_modeling_dataset.py
```

Train and evaluate tabular models:

```bash
python preprocessing/train_tabular_baseline.py
```

Current best model: `GradientBoostingRegressor`

| Model | Test RMSE | Test MAE | Test R2 |
| --- | ---: | ---: | ---: |
| Dummy mean baseline | 1.6198 | 1.0284 | -0.2520 |
| Random Forest | 1.0719 | 0.6418 | 0.4517 |
| Gradient Boosting | 0.8035 | 0.5104 | 0.6919 |
| Hist Gradient Boosting | 0.8342 | 0.5258 | 0.6679 |

Crop-wise performance for the best model:

| Crop | Test Rows | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: | ---: |
| Maize | 37 | 1.0216 | 0.6714 | 0.6750 |
| Rice | 31 | 0.4131 | 0.3183 | 0.7082 |

Generated artifacts:

- `datasets/modeling_dataset.csv`
- `models/tabular_best_model.joblib`
- `models/tabular_random_forest.joblib`
- `models/tabular_baseline_metrics.json`
- `models/tabular_model_comparison.csv`
- `models/tabular_crop_metrics.csv`
- `models/tabular_feature_importance.csv`
- `models/tabular_baseline_test_predictions.csv`
- `models/reports/actual_vs_predicted.png`
- `models/reports/error_distribution.png`
- `models/reports/feature_importance.png`

`Production` is excluded from training to avoid target leakage, and a small number of implausible yield outliers above 15 tonnes/ha are filtered before model fitting.

## CNN Image-Only Baseline

The image baseline uses only complete RGB/NDVI pairs whose paths exist, whose
yield target matches exactly, and whose image paths are not reused across
target keys. It uses the same chronological year boundaries as the tabular
baseline: train through 2016, validation in 2017-2018, and test from 2019.
The current image collection has no eligible 2020 pairs, so its test set only
covers 2019 and is too small for a definitive model comparison.

Install the CNN dependencies in the Python environment used to run training:

```bash
python -m pip install -r requirements-cnn.txt
```

Build the manifest and aligned paired-sample CSVs, then train the image-only
baseline:

```bash
python preprocessing/build_cnn_manifest.py
python preprocessing/prepare_cnn_dataset.py
python preprocessing/train_cnn_image_baseline.py
```

The preparation step writes `datasets/cnn_samples.csv` and the split files
`datasets/cnn_train.csv`, `datasets/cnn_validation.csv`, and
`datasets/cnn_test.csv`. The training step writes the Keras model,
test predictions, and metrics under `models/`. It reports tabular Gradient
Boosting metrics on the same eligible test samples for a like-for-like
comparison. Treat these results as exploratory because the CNN is trained
from scratch on a small image cohort and the test set is small.

Current aligned test results (13 complete image pairs from 2019):

| Model | Test RMSE | Test MAE | Test R2 |
| --- | ---: | ---: | ---: |
| Image-only CNN (RGB + NDVI) | 1.3560 | 0.9282 | 0.0610 |
| Gradient Boosting (same 13 samples) | 0.7936 | 0.5489 | 0.6784 |

This small test set is not sufficient to conclude that image-only modeling
cannot help; the current result only shows that this CNN baseline is not
ready to replace the tabular model. See `models/cnn_image_baseline_metrics.json`
and `models/cnn_image_baseline_test_predictions.csv` for the full evaluation.

## Current Status

The repository currently includes:

- A working interactive dashboard for exploring crop data and spatial inputs
- Data preprocessing scripts for rice/maize and geospatial enrichment
- A trained tabular baseline model integrated into the prediction interface
- An auditable CNN manifest, aligned paired-image splits, and a trained image-only CNN baseline
- A rule-based fallback when the trained model artifact is unavailable

## Future Improvements

- Add CNN image-only and CNN-plus-tabular fusion baselines
- Add proper multimodal fusion between tabular and image features
- Add XGBoost or LightGBM if those dependencies are available
- Generate production-ready deployment and packaging for the app

## License

This project is intended for academic and research use. Please check the repository owner or institutional guidelines for licensing details before reuse or redistribution.

