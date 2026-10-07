# Academic Project Report (Draft)

**Title:** Crop Yield Prediction Using Multimodal Data Fusion  
**Domain:** Machine Learning, Computer Vision, Geospatial Remote Sensing & Precision Agriculture  
**Focus Crops:** Rice (*Oryza sativa*) and Maize (*Zea mays*) across India  
**Status:** Mid-Project Interim Draft / Pre-Completion Technical Report  

---

## Abstract

Accurate crop yield prediction is crucial for food security planning, agricultural market stability, and localized resource management. Traditional agricultural yield estimation relies primarily on regional agronomic surveys and historical production records, which often lag in time and fail to capture real-time intra-seasonal vegetative dynamics. Conversely, satellite remote sensing provides continuous vegetation indices (e.g., NDVI, EVI) but lacks critical ground-truth context such as fertilizer dosage, pesticide application, and localized soil chemical composition. This project develops a multimodal data fusion pipeline that integrates historical agricultural statistics, meteorological patterns, soil profile lookup tables, and Sentinel-2 multispectral surface reflectance data to forecast rice and maize yields across Indian states. 

At this interim project stage, we have established:
1. An auditable, leakage-free data harmonisation pipeline across 1,774 agricultural records (spanning 1997–2020) and matched Sentinel-2 satellite tiles.
2. A temporal evaluation scheme (Training: $\le 2016$, Validation: $2017\text{--}2018$, Testing: $2019\text{--}2020$).
3. Strong tabular regression baselines, where Gradient Boosting achieves a test $R^2$ of **0.6919** and an RMSE of **0.8035 tonnes/ha** (Rice $R^2 = 0.7082$, Maize $R^2 = 0.6750$).
4. Image-only exploratory baselines (custom 6-channel CNN and spatial summary regressors) evaluating Sentinel-2 RGB and NDVI imagery.
5. An interactive Streamlit analytical and inference dashboard.

This report documents the architectural design, data curation, baseline benchmark results, and lays out the methodology for the subsequent deep multimodal fusion architecture.

---

## 1. Introduction

### 1.1 Problem Statement & Background
Agriculture remains the primary livelihood for over 50% of India's population. Rice and maize serve as foundational staple and commercial crops. However, crop yield fluctuates significantly across seasons and agro-climatic zones due to variations in precipitation, nutrient inputs, and local soil fertility.

While statistical yield regression using historical data provides an overarching baseline, it cannot observe actual crop vigor during the phenological growth cycle. Concurrently, satellite remote sensing (such as Sentinel-2 from the European Space Agency) captures vegetation greenness and canopy moisture, but satellite images alone cannot infer human inputs like pesticide or fertilizer application. Consequently, fusing ground agronomic records with satellite observations offers a superior, holistic predictive framework.

### 1.2 Key Objectives
- **Multimodal Data Integration:** Construct a unified dataset harmonizing tabular agricultural records (area, rainfall, fertilizer, pesticide), soil chemistry profiles, geographic coordinates, and Sentinel-2 spectral features.
- **Leakage Prevention & Robust Splitting:** Implement strict temporal train-validation-test splitting based on agricultural years to reflect genuine forecasting conditions, while eliminating circular target variables (such as raw total production).
- **Benchmarking Baseline Models:** Train and evaluate classical machine learning baselines (Random Forest, Gradient Boosting, HistGradientBoosting) alongside image-only architectures.
- **Interactive Decision-Support UI:** Deploy a functional web interface enabling farmers and policy analysts to inspect state-level geospatial distributions and forecast yields under varying input scenarios.
- **Deep Multimodal Fusion (Next Stage):** Design and evaluate joint feature-fusion architectures (CNN + Tabular Dense Embeddings) to test whether multimodal fusion surpasses single-modality predictors.

---

## 2. Literature Review & Theoretical Background

### 2.1 Tabular & Meteorological Yield Modeling
Early crop yield models utilized linear regression and autoregressive time-series models linking seasonal rainfall to total harvest. The advent of tree-based ensemble methods—specifically Random Forests (Breiman, 2001) and Gradient Boosted Decision Trees (Friedman, 2001)—demonstrated superior handling of non-linear environmental interactions (e.g., non-linear crop response to excessive fertilizer or heat stress). However, tabular models are fundamentally constrained by spatial aggregation and reporting latency.

### 2.2 Remote Sensing and Multispectral Indices
The European Space Agency (ESA) Sentinel-2 constellation offers high-resolution (10m–20m) multispectral imagery revisited every 5 days. Key vegetation indices derived from these bands include:
- **Normalized Difference Vegetation Index (NDVI):**
  $$\text{NDVI} = \frac{\text{NIR} - \text{RED}}{\text{NIR} + \text{RED}} = \frac{B8 - B4}{B8 + B4}$$
  Quantifies chlorophyll absorption and canopy density.
- **Enhanced Vegetation Index (EVI):**
  $$\text{EVI} = 2.5 \times \frac{\text{NIR} - \text{RED}}{\text{NIR} + 6 \times \text{RED} - 7.5 \times \text{BLUE} + 1}$$
  Corrects for atmospheric aerosols and soil background saturation in dense vegetative canopies.
- **Normalized Difference Water Index (NDWI):**
  $$\text{NDWI} = \frac{\text{NIR} - \text{SWIR}}{\text{NIR} + \text{SWIR}} \quad \text{or} \quad \frac{\text{GREEN} - \text{NIR}}{\text{GREEN} + \text{NIR}}$$
  Monitors plant canopy water content and liquid surface water.

### 2.3 Multimodal Data Fusion Paradigms
Data fusion is typically categorized into three paradigms:
1. **Early Fusion (Data-level):** Merging statistical features and spatial summaries into a single vector prior to modeling.
2. **Intermediate / Joint Fusion (Feature-level):** Passing spatial raster data through convolutional neural networks (CNNs) and tabular features through fully connected layers (MLPs), concatenating the latent embedding vectors, and jointly optimizing the regression head via backpropagation.
3. **Late Fusion (Decision-level):** Averaging or stacking independent predictions from separate tabular and image models.

---

## 3. Data Engineering & Preprocessing Pipeline

### 3.1 Data Ingestion & Sources
The project harmonizes five primary datasets:
1. **Crop Yield Base Data (`crop_yield.csv`):** 1,781 historical records detailing state, crop type (Rice, Maize), season (Kharif, Rabi, Summer, Whole Year), area (hectares), production (tonnes), annual rainfall (mm), fertilizer (kg), and pesticide (kg).
2. **Geographic Coordinates (`state_coordinates.csv`):** State centroid latitudes and longitudes for spatial mapping.
3. **Soil Chemical Properties (`state_soil_lookup.csv`):** State-level reference profiles covering nitrogen, phosphorus, potassium, organic carbon, and pH.
4. **Satellite Spectral Statistics (`sentinel2_features.csv`):** Aggregated Sentinel-2 multi-band statistics (Blue, Green, Red, NIR, EVI, NDVI, NDWI: mean, min, max, stdDev).
5. **Satellite Image Archive (`datasets/test_images/`):** Paired $96 \times 96$ PNG tiles of RGB and rendered NDVI samples organized chronologically from 1997 to 2020.

### 3.2 Target Leakage Prevention & Outlier Curation
- **Target Leakage:** Because $\text{Yield} = \frac{\text{Production}}{\text{Area}}$, including `Production` as an input feature creates near-perfect mathematical leakage ($R^2 \to 1.0$), invalidating real-world predictive utility. `Production` is rigorously stripped from the training feature set.
- **Agronomic Rate Normalization:** Two engineered interaction features were created:
  $$\text{Fertilizer per Area} = \frac{\text{Fertilizer}}{\text{Area}}, \quad \text{Pesticide per Area} = \frac{\text{Pesticide}}{\text{Area}}$$
- **Outlier Filtering:** Agronomic examination identified 7 implausible record anomalies with reported yields exceeding $15\text{ tonnes/ha}$ (well above standard biological limits for non-experimental field conditions). These were removed, leaving **1,774 clean records**.

### 3.3 Temporal Train / Validation / Test Partitioning
Random $k$-fold cross-validation introduces temporal data leakage when working with time-dependent climatic trends. Therefore, an explicit chronological split was enforced:

| Split Set | Year Window | Number of Samples | Purpose |
| :--- | :--- | :---: | :--- |
| **Training Set** | $1997 \le \text{Year} \le 2016$ | 1,556 | Model parameter optimization |
| **Validation Set** | $2017 \le \text{Year} \le 2018$ | 150 | Hyperparameter tuning & early stopping |
| **Test Set** | $2019 \le \text{Year} \le 2020$ | 68 | Final unbiased out-of-sample evaluation |

---

## 4. Current Experimental Setup & Baseline Results

### 4.1 Evaluation Metrics
Performance is measured across three standard regression metrics:
- **Root Mean Squared Error (RMSE):** $\sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$
- **Mean Absolute Error (MAE):** $\frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$
- **Coefficient of Determination ($R^2$):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

### 4.2 Tabular Baseline Comparison
Four tabular models were trained on 43 engineered features (40 numeric, 3 categorical with one-hot encoding):

| Model Architecture | Validation RMSE | Validation MAE | Validation $R^2$ | Test RMSE | Test MAE | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Mean Baseline** | 1.4777 | 0.9144 | -0.1749 | 1.6198 | 1.0284 | -0.2520 |
| **Random Forest Regressor** | 1.0031 | 0.5878 | 0.4586 | 1.0719 | 0.6418 | 0.4517 |
| **HistGradientBoosting** | 0.7787 | 0.4624 | 0.6738 | 0.8342 | 0.5258 | 0.6679 |
| **Gradient Boosting Regressor (Best)** | **0.7316** | **0.4658** | **0.7120** | **0.8035** | **0.5104** | **0.6919** |

#### Crop-Wise Performance Breakdown (Gradient Boosting)
On the out-of-sample test split, performance broken down by crop species illustrates consistent predictive capacity:
- **Rice ($N = 31$):** $\text{RMSE} = 0.4131\text{ t/ha}$, $\text{MAE} = 0.3183\text{ t/ha}$, **$R^2 = 0.7082$**
- **Maize ($N = 37$):** $\text{RMSE} = 1.0216\text{ t/ha}$, $\text{MAE} = 0.6714\text{ t/ha}$, **$R^2 = 0.6750$**

#### Feature Importance Analysis
Feature importance extraction from the best Gradient Boosting Regressor demonstrates that spatial geography and spectral vegetation metrics heavily influence predictions:
- **Top Drivers:** Latitude (10.39%), Longitude (4.24%), State indicator (Andhra Pradesh, 3.82%), Cultivated Area (3.32%), Fertilizer (3.24%), Tamil Nadu indicator (3.22%), Pesticide (3.12%), and Year (2.91%).
- **Satellite Drivers:** EVI Mean (2.61%), NDVI Mean (2.36%), RED Mean (2.28%), and NDWI Mean (2.25%).

### 4.3 Image-Only Baseline Experiments
To evaluate the standalone predictive strength of remote sensing rasters without tabular inputs, experiments were executed on aligned paired-image splits:

1. **6-Channel CNN Architecture:** A custom convolutional network taking a concatenated $96 \times 96 \times 6$ tensor (RGB + NDVI), trained from scratch over 37 epochs.
2. **Image Summary Feature Descriptors:** Extracting per-channel means, standard deviations, quantiles, and spatial histograms from RGB and NDVI PNGs.

**Results on Aligned 2019 Test Subset ($N = 13$):**

| Input Modality / Architecture | Test RMSE | Test MAE | Test $R^2$ |
| :--- | :---: | :---: | :---: |
| **Small 6-Channel CNN (from scratch)** | 1.3560 | 0.9282 | 0.0610 |
| **RGB Summary Features (Extra Trees)** | 1.1762 | 0.7446 | 0.2935 |
| **NDVI Summary Features (Extra Trees)** | 1.0987 | 0.7723 | 0.3836 |
| **RGB + NDVI Summary (Ridge Regressor)** | 1.1226 | 0.7639 | 0.3565 |
| **Tabular Gradient Boosting (Reference on same 13)** | **0.7936** | **0.5489** | **0.6784** |

#### Critical Finding for Interim Stage:
The image-only baseline underperforms the tabular reference ($R^2 = 0.0610$ vs $0.6784$). This empirically confirms that **satellite imagery alone cannot predict crop yield effectively without knowledge of agricultural inputs (fertilizer, area, season, rainfall)**. Conversely, spectral indicators provide valuable canopy health adjustments. This directly justifies the necessity of an integrated **multimodal fusion** architecture.

---

## 5. Software & Web Application Architecture

The system is operationalized via a multi-page interactive Streamlit dashboard (`app.py`).

### 5.1 System Modules
- **Overview & Documentation Tab:** Explains the project workflow, multimodal feature definitions, and model evaluation metrics.
- **Geographic Exploration Tab:** Visualizes state coordinates and historical production distributions across India using interactive Plotly maps.
- **Satellite Explorer Tab:** Allows visual inspection of Sentinel-2 RGB imagery alongside rendered NDVI heatmaps across different cultivation years.
- **Yield Prediction Interface:** An end-to-end interactive inference form where users select crop type, state, season, area, rainfall, and input levels. The interface dynamically queries `models/tabular_best_model.joblib`. If the model artifact is absent, an intelligent rule-based domain fallback provides estimated yields.

---

## 6. Project Roadmap: Completed vs. Pending Milestones

| Milestone / Component | Status | Deliverable Artifacts |
| :--- | :--- | :--- |
| **1. Problem formulation, scoping, and literature survey** | COMPLETED | Project documentation & methodology specification |
| **2. Tabular, soil, and satellite metadata ingestion** | COMPLETED | `datasets/crop_yield.csv`, `state_soil_lookup.csv` |
| **3. Preprocessing, outlier filtering, and leakage controls** | COMPLETED | `datasets/modeling_dataset.csv`, `cnn_samples.csv` |
| **4. Tabular baseline benchmarking (GBR, RF, HistGBR)** | COMPLETED | `models/tabular_best_model.joblib` ($R^2 = 0.6919$) |
| **5. Image-only exploratory baseline benchmarking** | COMPLETED | `models/cnn_image_baseline.keras` ($R^2 = 0.0610$) |
| **6. Streamlit interactive visualization and prediction app** | COMPLETED | Functional dashboard (`app.py`) |
| **7. Deep Multimodal Fusion Network (CNN + Tabular MLP)** | IN PROGRESS | Model training scripts (`fusion_model.py`) |
| **8. Comprehensive test ablation study on fused model** | PENDING | Final metric comparison table |
| **9. In-depth error analysis and residual diagnostic study** | PENDING | Residual plots & state-wise error charts |
| **10. Final report compilation & production packaging** | PENDING | Final project submission report & defense slides |

---

## 7. Next Steps for Final Project Completion

To transition this draft into the final project submission, the remaining engineering and analytical tasks are:

1. **Multimodal Dual-Branch Network:**
   Implement a dual-branch neural network:
   - A vision backbone (CNN feature extractor) processing Sentinel-2 imagery.
   - A tabular feedforward network processing soil, weather, and agricultural inputs.
   - A fusion layer (concatenation / cross-attention) passing into the final dense regression head.
2. **Ablation Benchmark Table:**
   Generate the final comparison table across identical test instances:
   $$\text{Multimodal Fusion} \quad \text{vs} \quad \text{Tabular Only} \quad \text{vs} \quad \text{Vision Only}$$
3. **Residual & Error Diagnostics:**
   Conduct spatial and crop-specific error analysis to identify which states or climatic anomalies present the highest residual error.
4. **Final Deployment:**
   Bundle the trained multimodal weights into the Streamlit app to deliver real-time multimodal inference directly from the user interface.

---

## References

1. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
2. Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine*. Annals of Statistics, 1189-1232.
3. European Space Agency (ESA). *Sentinel-2 User Handbook*. Standard Earth Observation Reference Documents.
4. Lobell, D. B., et al. (2015). *The influence of climate on global crop productivity*. Science, 333(6042), 616-620.
5. You, J., et al. (2017). *Deep Gaussian Processes for Crop Yield Prediction Based on Remote Sensing Data*. AAAI Conference on Human Computation and Crowdsourcing.
