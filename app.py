from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="Crop Yield Prediction | Multimodal Data Fusion",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for UI styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        color: #123d8f;
        text-align: center;
        margin-bottom: 0.4rem;
        letter-spacing: -0.04em;
    }
    .sub-header {
        font-size: 1.15rem;
        font-weight: 500;
        color: #475569;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .hero-badge {
        display: inline-block;
        margin: 0 auto 0.8rem auto;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        background: linear-gradient(135deg, #dbeafe, #dcfce7);
        color: #0f172a;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border-radius: 18px;
        padding: 1.3rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
        text-align: center;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-left: 5px solid #10B981;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #475569;
        font-weight: 700;
    }
    .section-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%);
        border: 1px solid #A7F3D0;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 12px 28px rgba(16, 185, 129, 0.12);
    }
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.7rem 1.2rem !important;
        background: linear-gradient(135deg, #16a34a, #22c55e) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 10px 22px rgba(34, 197, 94, 0.25) !important;
    }
    .stButton > button:hover {
        filter: brightness(1.04);
        box-shadow: 0 12px 22px rgba(34, 197, 94, 0.35) !important;
    }
    .sidebar-content .block-container {
        padding-top: 1rem;
    }
    .dark-mode .prediction-box {
        background: linear-gradient(135deg, rgba(16,185,129,0.16), rgba(59,130,246,0.14));
        border: 1px solid rgba(16,185,129,0.35);
    }
    </style>
""",
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
TEST_IMAGES_DIR = DATASETS_DIR / "test_images"
MODEL_READY = False


def apply_theme(dark_mode: bool):
    if dark_mode:
        st.markdown(
            """
            <style>
            .stApp {
                background: radial-gradient(circle at top, rgba(59,130,246,0.2), transparent 20%), linear-gradient(180deg, #020817 0%, #0f172a 100%);
                color: #e5e7eb;
            }
            .main-header {
                color: #dbeafe !important;
            }
            .sub-header {
                color: #cbd5e1 !important;
            }
            .hero-badge {
                background: linear-gradient(135deg, rgba(191,219,254,0.22), rgba(134,239,172,0.22)) !important;
                color: #f8fafc !important;
            }
            .metric-card {
                background: linear-gradient(135deg, #111827 0%, #1f2937 100%) !important;
                border: 1px solid rgba(148, 163, 184, 0.18) !important;
                border-left: 5px solid #34d399 !important;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.6) !important;
            }
            .metric-value { color: #f9fafb !important; }
            .metric-label { color: #cbd5e1 !important; }
            .section-card {
                background: #111827 !important;
                border: 1px solid rgba(148, 163, 184, 0.25) !important;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.35) !important;
            }
            .block-container, .stDataFrame, .stMarkdown, .stTabs, .stSelectbox, .stSlider, .stNumberInput {
                color: #e5e7eb !important;
            }
            .prediction-box {
                background: linear-gradient(135deg, rgba(16,185,129,0.16), rgba(59,130,246,0.14)) !important;
                border: 1px solid rgba(52, 211, 153, 0.35) !important;
            }
            .stButton > button {
                background: linear-gradient(135deg, #10b981, #3b82f6) !important;
                box-shadow: 0 12px 24px rgba(59,130,246,0.32) !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
                color: #0f172a;
            }
            .main-header { color: #1E3A8A !important; }
            .sub-header { color: #4B5563 !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def predict_crop_yield(crop, state, year, area, rainfall, fertilizer, pesticide, soil_ph, soil_carbon):
    """Simple rule-based fallback used until the actual trained model is available."""
    # Scale inputs to meaningful ranges observed in agricultural datasets.
    area_norm = max((area - 20) / 60, 0)
    rainfall_norm = max((rainfall - 600) / 1000, 0)
    fertilizer_norm = max((fertilizer - 50) / 200, 0)
    pesticide_norm = max((pesticide - 10) / 80, 0)
    soil_ph_norm = 1 - abs(soil_ph - 6.5) / 4
    soil_carbon_norm = max((soil_carbon - 5) / 20, 0)

    if crop == "Rice":
        base = 2.8
        yield_est = (
            base
            + (0.70 * area_norm)
            + (1.45 * rainfall_norm)
            + (1.15 * fertilizer_norm)
            + (0.35 * pesticide_norm)
            + (1.10 * soil_ph_norm)
            + (0.80 * soil_carbon_norm)
        )
    else:
        base = 2.2
        yield_est = (
            base
            + (0.80 * area_norm)
            + (1.20 * rainfall_norm)
            + (1.00 * fertilizer_norm)
            + (0.45 * pesticide_norm)
            + (1.05 * soil_ph_norm)
            + (0.90 * soil_carbon_norm)
        )

    # Keep outputs realistic and plant-specific.
    if crop == "Rice":
        return max(1.2, min(9.5, yield_est))
    return max(1.0, min(8.5, yield_est))


# Theme defaults for a light-first experience.
if "theme" not in st.session_state:
    st.session_state.theme = "light"

apply_theme(st.session_state.theme == "dark")


# ============================================================
# DATA LOADING (CACHED)
# ============================================================
@st.cache_data
def load_datasets():
    crop_yield_path = DATASETS_DIR / "crop_yield.csv"
    coords_path = DATASETS_DIR / "state_coordinates.csv"
    soil_path = DATASETS_DIR / "state_soil_lookup.csv"
    sentinel_features_path = DATASETS_DIR / "sentinel2_features.csv"

    rice_prep_path = DATASETS_DIR / "rice_data_preprocessed.csv"
    maize_prep_path = DATASETS_DIR / "maize_data_preprocessed.csv"

    raw_df = pd.read_csv(crop_yield_path) if crop_yield_path.exists() else pd.DataFrame()
    coords_df = pd.read_csv(coords_path) if coords_path.exists() else pd.DataFrame()
    soil_df = pd.read_csv(soil_path) if soil_path.exists() else pd.DataFrame()
    sentinel_df = pd.read_csv(sentinel_features_path) if sentinel_features_path.exists() else pd.DataFrame()

    rice_df = pd.read_csv(rice_prep_path) if rice_prep_path.exists() else pd.DataFrame()
    maize_df = pd.read_csv(maize_prep_path) if maize_prep_path.exists() else pd.DataFrame()

    if not raw_df.empty:
        raw_df.columns = raw_df.columns.str.strip()

    return raw_df, coords_df, soil_df, sentinel_df, rice_df, maize_df


raw_df, coords_df, soil_df, sentinel_df, rice_df, maize_df = load_datasets()

# ============================================================
# HEADER & SIDEBAR NAVIGATION
# ============================================================
st.markdown(
    '<div style="text-align: center;"><div class="hero-badge">Agricultural Intelligence</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-header">🌾 Crop Yield Prediction Using Multimodal Data Fusion</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Fusing Tabular Agricultural Drivers + Sentinel-2 Satellite Optical Imagery & Soil Grids</div>',
    unsafe_allow_html=True,
)

st.sidebar.image(
    "https://img.icons8.com/color/96/000000/wheat.png", width=70
)
st.sidebar.title("Navigation")
st.sidebar.caption("Dashboard controls")
menu_option = st.sidebar.radio(
    "Go to",
    [
        "📊 Project Overview & Data Insights",
        "🛰️ Satellite & Spatial Data Explorer",
        "🌾 Crop Yield Prediction",
        "🧠 Project Pipeline & Architecture",
    ],
)

dark_mode = st.sidebar.toggle("🌙 Dark mode", value=st.session_state.theme == "dark")
st.session_state.theme = "dark" if dark_mode else "light"
apply_theme(dark_mode)

# ============================================================
# PAGE 1: OVERVIEW & DATA INSIGHTS
# ============================================================
if menu_option == "📊 Project Overview & Data Insights":
    st.markdown("### 📈 Executive Summary & Key Dataset Metrics")

    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-value">{len(raw_df):,}</div>
            <div class="metric-label">Total Record Rows</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        num_states = len(coords_df) if not coords_df.empty else 30
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-value">{num_states}</div>
            <div class="metric-label">Unique Indian States</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value">Rice & Maize</div>
            <div class="metric-label">Target Crops</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """<div class="metric-card">
            <div class="metric-value">1997 - 2020</div>
            <div class="metric-label">Temporal Coverage</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Filters
    st.markdown("#### 🔍 Filter Dataset")
    f_col1, f_col2, f_col3 = st.columns(3)

    if not raw_df.empty:
        crops_available = sorted(raw_df["Crop"].unique())
        selected_crop = f_col1.selectbox(
            "Select Crop", crops_available, index=crops_available.index("Rice") if "Rice" in crops_available else 0
        )

        states_available = ["All States"] + sorted(raw_df["State"].unique())
        selected_state = f_col2.selectbox("Select State", states_available)

        min_yr, max_yr = int(raw_df["Crop_Year"].min()), int(raw_df["Crop_Year"].max())
        selected_years = f_col3.slider("Crop Year Range", min_yr, max_yr, (min_yr, max_yr))

        # Filter dataframe
        filtered_df = raw_df[
            (raw_df["Crop"] == selected_crop)
            & (raw_df["Crop_Year"].between(selected_years[0], selected_years[1]))
        ]
        if selected_state != "All States":
            filtered_df = filtered_df[filtered_df["State"] == selected_state]

        # Charts Section
        c_col1, c_col2 = st.columns(2)

        with c_col1:
            st.subheader("Yield Trend over Years")
            avg_yield_yr = (
                filtered_df.groupby("Crop_Year")["Yield"].mean().reset_index()
            )
            fig_yield = px.line(
                avg_yield_yr,
                x="Crop_Year",
                y="Yield",
                title=f"Average Yield (tonnes/ha) for {selected_crop}",
                markers=True,
                color_discrete_sequence=["#10B981"],
            )
            st.plotly_chart(fig_yield, use_container_width=True)

        with c_col2:
            st.subheader("Area vs Production Relationship")
            fig_scatter = px.scatter(
                filtered_df,
                x="Area",
                y="Production",
                color="State" if selected_state == "All States" else None,
                hover_data=["Annual_Rainfall", "Yield"],
                title="Cultivated Area vs Total Production",
                log_x=True,
                log_y=True,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Top States by Yield
        st.subheader("🏆 Top 10 States by Average Yield")
        if selected_state == "All States":
            top_states = (
                filtered_df.groupby("State")["Yield"]
                .mean()
                .nlargest(10)
                .reset_index()
            )
            fig_top = px.bar(
                top_states,
                x="State",
                y="Yield",
                color="Yield",
                color_continuous_scale="Greens",
                title=f"Top 10 States – Average {selected_crop} Yield (tonnes/ha)",
            )
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info(f"Showing data for {selected_state} only. Select **All States** to compare.")

        # Correlation Heatmap
        st.subheader("🔥 Feature Correlation Heatmap")
        numeric_cols = ["Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
        avail_cols = [c for c in numeric_cols if c in filtered_df.columns]
        if len(avail_cols) > 2:
            corr = filtered_df[avail_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                title="Feature Correlation Matrix",
                aspect="auto",
            )
            fig_corr.update_layout(height=450)
            st.plotly_chart(fig_corr, use_container_width=True)

        # Data Quality & Download
        with st.expander("📋 Data Quality Report"):
            dq1, dq2, dq3 = st.columns(3)
            dq1.metric("Total Rows", f"{len(filtered_df):,}")
            dq2.metric("Missing Values", f"{filtered_df.isnull().sum().sum()}")
            dq3.metric("Duplicate Rows", f"{filtered_df.duplicated().sum()}")
            st.dataframe(filtered_df.describe(), use_container_width=True)

        st.download_button(
            "📥 Download Filtered Data (CSV)",
            filtered_df.to_csv(index=False),
            file_name=f"{selected_crop}_{selected_state}_filtered.csv",
            mime="text/csv",
        )

    # State Coordinates Map
    st.subheader("📍 Geospatial Distribution of Target States")
    if not coords_df.empty:
        if hasattr(px, "scatter_map"):
            fig_map = px.scatter_map(
                coords_df,
                lat="Latitude",
                lon="Longitude",
                hover_name="State",
                zoom=3.8,
                center={"lat": 22.5937, "lon": 78.9629},
                height=450,
                color_discrete_sequence=["#2563EB"],
            )
        elif hasattr(px, "scatter_mapbox"):
            fig_map = px.scatter_mapbox(
                coords_df,
                lat="Latitude",
                lon="Longitude",
                hover_name="State",
                zoom=3.8,
                center={"lat": 22.5937, "lon": 78.9629},
                mapbox_style="carto-positron",
                height=450,
                color_discrete_sequence=["#2563EB"],
            )
        else:
            fig_map = px.scatter_geo(
                coords_df,
                lat="Latitude",
                lon="Longitude",
                hover_name="State",
                scope="asia",
                height=450,
            )
        st.plotly_chart(fig_map, use_container_width=True)

# ============================================================
# PAGE 2: SATELLITE & SPATIAL DATA EXPLORER
# ============================================================
elif menu_option == "🛰️ Satellite & Spatial Data Explorer":
    st.markdown("### 🛰️ Multimodal Spatial Data Explorer (Sentinel-2 Optical & Soil Grids)")

    col1, col2, col3 = st.columns(3)
    exp_state = col1.selectbox("Select State for Visual Analysis", sorted(coords_df["State"].unique()) if not coords_df.empty else ["Punjab"])
    exp_crop = col2.selectbox("Select Crop Modal", ["Rice", "Maize"])
    exp_year = col3.selectbox("Select Year", list(range(2020, 1999, -1)))

    st.markdown("---")

    # Displays Satellite Photo and Climate-Soil Heatmaps side-by-side
    img_col1, img_col2 = st.columns(2)
    state_slug = exp_state.lower().replace(" ", "_")
    crop_slug = exp_crop.lower().replace(" ", "_")

    with img_col1:
        st.subheader("📷 Sentinel-2 True-Color Optical Photo (10m Resolution)")
        gee_s2_image_path = TEST_IMAGES_DIR / "rgb" / f"{state_slug}_{exp_year}_{crop_slug}_sentinel2_gee_rgb.png"
        old_s2_image_path = TEST_IMAGES_DIR / f"{state_slug}_{exp_year}_sentinel2_rgb.jpg"
        s2_image_path = gee_s2_image_path if gee_s2_image_path.exists() else old_s2_image_path

        if s2_image_path.exists():
            st.image(
                str(s2_image_path),
                caption=f"Sentinel-2 True Color Optical Photo ({exp_state}, {exp_year})",
                use_container_width=True,
            )
            st.success("✅ Real 10m Sentinel-2 Optical Satellite Scene Loaded!")
        else:
            # Check for fallback punjab image
            default_s2 = TEST_IMAGES_DIR / "punjab_2020_rice_sentinel2_gee_rgb.png"
            if not default_s2.exists():
                default_s2 = TEST_IMAGES_DIR / "punjab_2020_sentinel2_rgb.jpg"
            if default_s2.exists():
                st.image(
                    str(default_s2),
                    caption=f"Sample Sentinel-2 True Color Optical Photo (Punjab, 2020)",
                    use_container_width=True,
                )
                st.info(f"Showing sample Sentinel-2 scene. Run single-state downloader to generate for {exp_state}.")
            else:
                st.warning(f"No cached Sentinel-2 image found for {exp_state}. Run fetch_single_state_satellite.py to download.")

    with img_col2:
        st.subheader("🌡️ 4-Channel Environmental & Soil Spatial Heatmap")
        ndvi_path = TEST_IMAGES_DIR / "ndvi" / f"{state_slug}_{exp_year}_{crop_slug}_sentinel2_ndvi.png"
        heatmap_path = TEST_IMAGES_DIR / f"{state_slug}_{exp_year}_{crop_slug}_heatmap.png"
        default_heatmap = TEST_IMAGES_DIR / "punjab_2020_rice_heatmap.png"

        if ndvi_path.exists():
            st.subheader("Sentinel-2 NDVI Crop Vigor Heatmap")
            st.image(
                str(ndvi_path),
                caption=f"Cloud-masked seasonal NDVI composite ({exp_state}, {exp_crop}, {exp_year})",
                use_container_width=True,
            )
        elif heatmap_path.exists():
            st.image(
                str(heatmap_path),
                caption=f"Spatial Climate & Soil Multi-Channel Heatmap ({exp_state})",
                use_container_width=True,
            )
        elif default_heatmap.exists():
            st.image(
                str(default_heatmap),
                caption=f"Sample 4-Channel Spatial Heatmap [Rainfall, Temp, Soil] (Punjab)",
                use_container_width=True,
            )
        else:
            st.warning("No NDVI or environmental heatmap cached yet.")

    if not sentinel_df.empty:
        feature_row = sentinel_df[
            (sentinel_df["State"].str.casefold() == exp_state.casefold())
            & (sentinel_df["Crop"].str.casefold() == exp_crop.casefold())
            & (sentinel_df["Year"] == exp_year)
        ]
        if not feature_row.empty:
            st.markdown(f"#### Sentinel-2 Vegetation Features for {exp_state}")
            feature_cols = [
                col
                for col in [
                    "Image_Count",
                    "NDVI_mean",
                    "NDVI_stdDev",
                    "NDVI_min",
                    "NDVI_max",
                    "EVI_mean",
                    "NDWI_mean",
                    "NDRE_mean",
                    "NIR_mean",
                    "RED_EDGE_mean",
                ]
                if col in feature_row.columns
            ]
            st.dataframe(feature_row[["State", "Crop", "Year", "Season"] + feature_cols], use_container_width=True)
        else:
            st.info("No Sentinel-2 vegetation feature row cached for this state/crop/year yet.")

    with st.expander("How to generate clean Sentinel-2 images and feature rows"):
        st.code(
            "python preprocessing/fetch_single_state_satellite.py --state Punjab --crop Rice --year 2020\n"
            "python preprocessing/fetch_single_state_satellite.py --all-states --crop Rice --year 2020\n"
            "python preprocessing/fetch_single_state_satellite.py --from-crop-yield --crops Rice,Maize --years 2020",
            language="bash",
        )

    # Soil Grid Table for selected State
    st.markdown(f"#### 🧪 SoilGrids Geochemical Profile for {exp_state}")
    if not soil_df.empty:
        s_row = soil_df[soil_df["State"].str.casefold() == exp_state.casefold()]
        if not s_row.empty:
            soil_display = s_row[["State", "phh2o", "nitrogen", "soc", "clay", "sand", "silt"]].copy()
            soil_display.columns = ["State", "Soil pH", "Nitrogen (g/kg)", "Organic Carbon (g/kg)", "Clay %", "Sand %", "Silt %"]
            st.dataframe(soil_display, use_container_width=True)

# ============================================================
# PAGE 3: CROP YIELD PREDICTION
# ============================================================
elif menu_option == "🌾 Crop Yield Prediction":
    st.markdown("### 🌾 Crop Yield Predictor")
    st.info("This prediction panel is ready for the trained model. Until the model is trained, a rule-based fallback is used to validate the interface and user flow.")

    prediction_col1, prediction_col2 = st.columns(2)
    with prediction_col1:
        selected_crop = st.selectbox("Crop", ["Rice", "Maize"], index=0)
        selected_state = st.selectbox(
            "State",
            sorted(coords_df["State"].unique()) if not coords_df.empty else ["Punjab", "West Bengal", "Tamil Nadu"],
            index=0,
        )
        selected_year = st.slider("Year", 1997, 2025, 2023)
    with prediction_col2:
        area_ha = st.number_input("Cultivated Area (hectares)", min_value=10.0, max_value=5000.0, value=250.0, step=5.0)
        rainfall_mm = st.number_input("Annual Rainfall (mm)", min_value=200.0, max_value=3000.0, value=1200.0, step=25.0)

    soil_col1, soil_col2, soil_col3 = st.columns(3)
    with soil_col1:
        fertilizer_kg = st.number_input("Fertilizer (kg/ha)", min_value=0.0, max_value=500.0, value=150.0, step=5.0)
    with soil_col2:
        pesticide_kg = st.number_input("Pesticide (kg/ha)", min_value=0.0, max_value=200.0, value=30.0, step=2.0)
    with soil_col3:
        soil_ph = st.number_input("Soil pH", min_value=3.0, max_value=9.5, value=6.5, step=0.1)

    soc_value = st.number_input("Soil Organic Carbon (g/kg)", min_value=1.0, max_value=60.0, value=18.0, step=1.0)

    if st.button("Predict Crop Yield", type="primary"):
        predicted_yield = predict_crop_yield(
            selected_crop,
            selected_state,
            selected_year,
            area_ha,
            rainfall_mm,
            fertilizer_kg,
            pesticide_kg,
            soil_ph,
            soc_value,
        )

        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Predicted Yield", f"{predicted_yield:.2f} tonnes/ha")
        with colB:
            status_text = "Ready for training" if not MODEL_READY else "Model active"
            st.metric("Model Status", status_text)
        with colC:
            st.metric("Crop", selected_crop)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Prediction Interpretation")
        st.write(
            "The prediction is influenced mainly by rainfall, fertilizer intensity, soil condition, and crop type. "
            "Replace the rule-based fallback with the trained model weights once the model pipeline has been finalized."
        )

        scenario_df = pd.DataFrame(
            {
                "Feature": ["Area", "Rainfall", "Fertilizer", "Pesticide", "Soil pH", "Organic Carbon"],
                "Value": [area_ha, rainfall_mm, fertilizer_kg, pesticide_kg, soil_ph, soc_value],
            }
        )
        st.dataframe(scenario_df, use_container_width=True)
    else:
        st.markdown("### 🔍 Live preview")
        st.caption("Enter the crop and field conditions, then click Predict Crop Yield to see the estimate.")

# ============================================================
# PAGE 4: PROJECT PIPELINE & ARCHITECTURE
# ============================================================
elif menu_option == "🧠 Project Pipeline & Architecture":
    st.markdown("### 🧠 Project Pipeline & Proposed Architecture")

    # Project Status
    st.markdown("#### 🚦 Development Status")
    phases = [
        ("📦 Raw Data Collection (Crop Yield, Rainfall, Fertilizer)", "✅ Complete"),
        ("🛰️ Sentinel-2 Satellite Imagery via GEE Pipeline", "✅ Complete"),
        ("🧪 SoilGrids Geochemical Data Extraction", "✅ Complete"),
        ("⚙️ Data Preprocessing & Feature Engineering", "✅ Complete"),
        ("📊 Exploratory Data Analysis Dashboard", "✅ Complete"),
        ("🤖 Model Training & Evaluation", "🔧 In Progress"),
        ("🔗 Multimodal Fusion Implementation", "📋 Planned"),
    ]
    for phase, status in phases:
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{status} &ensp; {phase}")

    st.markdown("---")

    # Architecture
    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown(
            """
            #### 🏢 Proposed Dual-Stream Pipeline
            1. **Tabular Stream (MLP Branch)**:
               * Inputs: Area, Rainfall, Fertilizer/ha, Pesticide/ha, Soil pH/SOC.
               * Architecture: 3-Layer Dense MLP with BatchNorm & Dropout.
            2. **Vision Stream (CNN/ViT Branch)**:
               * Inputs: 4-Channel Sentinel-2 Optical RGB + NDVI Heatmaps `(64, 64, 4)`.
               * Architecture: ResNet-18 Backbone pretrained on Remote Sensing imagery.
            3. **Multimodal Fusion Head**:
               * Concatenation of Tabular Embeddings (64-dim) + Vision Feature Vector (128-dim).
               * Dense Regression Layers → Predict Yield (tonnes/ha).
        """
        )

    with m_col2:
        st.markdown("#### 📊 Data Readiness Summary")
        ready_col1, ready_col2 = st.columns(2)
        ready_col1.metric("Rice Samples", f"{len(rice_df):,}" if not rice_df.empty else "0")
        ready_col2.metric("Maize Samples", f"{len(maize_df):,}" if not maize_df.empty else "0")

        ready_col3, ready_col4 = st.columns(2)
        ready_col3.metric("Satellite Feature Rows", f"{len(sentinel_df):,}" if not sentinel_df.empty else "0")
        ready_col4.metric("Soil Profiles", f"{len(soil_df):,}" if not soil_df.empty else "0")

        st.markdown("#### 🗂️ Multimodal Feature Summary")
        feature_summary = pd.DataFrame(
            {
                "Data Source": ["Tabular (Crop Yield)", "Sentinel-2 Imagery", "SoilGrids"],
                "Key Features": [
                    "Area, Production, Rainfall, Fertilizer, Pesticide, Season, State",
                    "NDVI, EVI, NDWI, NDRE, NIR, RED_EDGE (mean/std/min/max)",
                    "pH, Nitrogen, SOC, Clay%, Sand%, Silt%",
                ],
                "Records": [
                    f"{len(raw_df):,}" if not raw_df.empty else "—",
                    f"{len(sentinel_df):,}" if not sentinel_df.empty else "—",
                    f"{len(soil_df):,}" if not soil_df.empty else "—",
                ],
            }
        )
        st.dataframe(feature_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🎯 Next Steps")
    st.markdown(
        """
    1. **Baseline Models** — Train Random Forest & XGBoost on tabular features; evaluate R², RMSE, MAE.
    2. **CNN Feature Extractor** — Fine-tune ResNet-18 on Sentinel-2 NDVI/RGB patches per state.
    3. **Multimodal Fusion** — Concatenate tabular embeddings + CNN features; train fusion regression head.
    4. **Evaluation** — Compare tabular-only vs vision-only vs fused model performance on held-out test set.
    """
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>Crop Yield Prediction using Multimodal Data Fusion | Built with Streamlit & Plotly</div>",
    unsafe_allow_html=True,
)
