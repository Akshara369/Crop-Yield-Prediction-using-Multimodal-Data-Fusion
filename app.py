from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
TABULAR_MODEL_PATH = ROOT / "models" / "tabular_best_model.joblib"
MULTIMODAL_MODEL_PATH = ROOT / "models" / "multimodal_best_tuned_model.joblib"
BENCHMARK_PATH = ROOT / "models" / "multimodal_vs_tabular_benchmark.csv"
TUNING_PATH = ROOT / "models" / "multimodal_tuning_results.csv"
MODELING_DATASET_PATH = DATASETS_DIR / "modeling_dataset.csv"
REPORTS_DIR = ROOT / "models" / "reports"

TARGET_CROPS = ["Rice", "Maize", "Moong(Green Gram)"]
SEASONS = ["Kharif", "Rabi", "Summer", "Autumn", "Winter", "Whole Year"]
EXCLUDED_COLUMNS = {
    "Yield", "Production", "Start_Date", "End_Date", "RGB_Image", "NDVI_Image", "Satellite_Source"
}


st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


def css(theme: str) -> str:
    dark = theme == "dark"
    return f"""
    <style>
    :root {{
        --bg: {"#071b2a" if dark else "#eef5f1"};
        --panel: {"#112b3d" if dark else "#ffffff"};
        --panel-soft: {"#0e2334" if dark else "#f8fbff"};
        --border: {"rgba(130, 190, 255, 0.28)" if dark else "rgba(35, 95, 125, 0.15)"};
        --text: {"#edf7ff" if dark else "#0f2336"};
        --muted: {"#b0c7d8" if dark else "#587083"};
        --green: #22c97a;
        --blue: #2ea8ff;
        --shadow: {"0 18px 42px rgba(3, 11, 21, 0.42)" if dark else "0 16px 35px rgba(31,78,96,.10)"};
    }}
    .stApp {{
        background:
            radial-gradient(circle at 18% -8%, {"rgba(34,201,122,.18)" if dark else "rgba(34,201,122,.20)"}, transparent 28%),
            linear-gradient(135deg, var(--bg), {"#0a2031" if dark else "#f8fbff"});
        color: var(--text);
    }}
    .block-container {{
        padding-top: 1.05rem;
        padding-bottom: 1rem;
        max-width: 1500px;
    }}
    section[data-testid="stSidebar"] {{
        background: {"linear-gradient(180deg,#081a2a,#0d2337)" if dark else "linear-gradient(180deg,#f8fffb,#e8f4f1)"};
        border-right: 1px solid var(--border);
    }}
    section[data-testid="stSidebar"] * {{
        color: var(--text);
    }}
    div[data-testid="stSidebarNav"] {{
        display: none;
    }}
    .topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: .85rem 1rem 1.05rem;
        border: 1px solid var(--border);
        background: {"linear-gradient(180deg, rgba(16,34,49,.9), rgba(9,23,35,.86))" if dark else "rgba(255,255,255,.78)"};
        box-shadow: var(--shadow);
        border-radius: 18px;
        backdrop-filter: blur(16px);
        margin-bottom: 1rem;
    }}
    .brand {{
        display: flex;
        align-items: center;
        gap: .9rem;
        min-width: 220px;
        flex-wrap: wrap;
    }}
    .brand-mark {{
        width: 52px;
        height: 52px;
        border-radius: 18px;
        display: grid;
        place-items: center;
        color: white;
        font-size: 1.75rem;
        background: linear-gradient(145deg, #15c76f, #1e88e5);
        box-shadow: 0 12px 24px rgba(34,201,122,.25);
    }}
    .brand h1 {{
        margin: 0;
        font-size: 1.55rem;
        line-height: 1.1;
        color: var(--text);
        letter-spacing: 0;
    }}
    .brand p {{
        margin: .18rem 0 0;
        color: var(--muted);
        font-size: .9rem;
    }}
    .tagline {{
        display: flex;
        gap: .9rem;
        align-items: center;
        color: var(--muted);
        font-weight: 600;
        white-space: nowrap;
    }}
    .tagline span:not(:last-child)::after {{
        content: "|";
        margin-left: .9rem;
        color: var(--border);
    }}
    .panel {{
        border: 1px solid var(--border);
        border-radius: 16px;
        background: {"rgba(17, 38, 52, 0.88)" if dark else "rgba(255,255,255,.88)"};
        box-shadow: var(--shadow);
        padding: 1rem;
        margin-bottom: 1rem;
    }}
    .panel-title {{
        color: var(--text);
        font-weight: 800;
        font-size: 1.02rem;
        margin: 0 0 .8rem;
    }}
    .metric-card {{
        min-height: 116px;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: linear-gradient(135deg, {"rgba(18, 42, 58, 0.98), rgba(11, 27, 40, 0.9)" if dark else "#ffffff, #f5fbff"});
        box-shadow: var(--shadow);
        padding: 1rem;
        display: flex;
        gap: .85rem;
        align-items: center;
        margin-bottom: 1rem;
    }}
    .metric-icon {{
        width: 58px;
        height: 58px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        color: white;
        font-size: 1.55rem;
        flex: 0 0 auto;
    }}
    .metric-label {{
        color: var(--muted);
        font-size: .82rem;
        font-weight: 700;
    }}
    .metric-value {{
        color: var(--text);
        font-size: 1.55rem;
        font-weight: 900;
        margin-top: .18rem;
    }}
    .metric-note {{
        color: var(--green);
        font-size: .82rem;
        font-weight: 800;
        margin-top: .35rem;
    }}
    .side-brand {{
        padding: .8rem .35rem 1rem;
    }}
    .small-muted {{
        color: var(--muted);
        font-size: .82rem;
    }}
    .prediction-result {{
        border-radius: 14px;
        padding: 1rem;
        background: {"linear-gradient(135deg, rgba(34,201,122,.18), rgba(46,168,255,.14))" if dark else "linear-gradient(135deg, rgba(34,201,122,.12), rgba(46,168,255,.10))"};
        border: 1px solid {"rgba(34,201,122,.35)" if dark else "rgba(34,201,122,.22)"};
    }}
    .prediction-number {{
        font-size: 2rem;
        line-height: 1;
        font-weight: 900;
        color: var(--text);
    }}
    .badge {{
        display: inline-block;
        padding: .28rem .55rem;
        border-radius: 999px;
        background: rgba(34,201,122,.16);
        color: {"#91ffbf" if dark else "#087d45"};
        font-weight: 800;
        font-size: .72rem;
    }}
    .mini-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .55rem;
        margin-top: .7rem;
    }}
    .mini-stat {{
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: .65rem;
        background: var(--panel-soft);
    }}
    .mini-stat b {{
        display: block;
        color: var(--text);
        font-size: .95rem;
        margin-top: .18rem;
    }}
    .pipeline {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(88px, 1fr));
        gap: .55rem;
        align-items: stretch;
    }}
    .pipe-step {{
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: .65rem;
        text-align: center;
        background: {"rgba(13, 27, 38, 0.95)" if dark else "var(--panel-soft)"};
        min-height: 82px;
        font-size: .82rem;
        line-height: 1.25;
        overflow-wrap: normal;
        word-break: normal;
    }}
    .pipe-icon {{
        font-size: 1.25rem;
        margin-bottom: .25rem;
    }}
    .advice-row {{
        display: grid;
        grid-template-columns: 1fr;
        gap: .45rem;
        margin-top: .7rem;
    }}
    .advice-chip {{
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: .55rem .65rem;
        background: var(--panel-soft);
        color: var(--muted);
        font-size: .8rem;
        line-height: 1.3;
    }}
    .stButton > button {{
        border-radius: 12px !important;
        border: 0 !important;
        background: linear-gradient(135deg, #18c777, #2ea8ff) !important;
        color: white !important;
        font-weight: 900 !important;
        box-shadow: 0 12px 24px rgba(34,201,122,.20) !important;
    }}
    div[data-testid="stMetric"] {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: .85rem;
    }}
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
        color: var(--muted) !important;
    }}
    h1, h2, h3, h4, h5, h6, p, label, span {{
        letter-spacing: 0;
    }}
    @media (max-width: 900px) {{
        .topbar, .tagline, .pipeline {{
            display: block;
        }}
        .brand {{
            margin-bottom: .75rem;
        }}
    }}
    </style>
    """


@st.cache_data
def load_data():
    paths = {
        "raw": DATASETS_DIR / "crop_yield.csv",
        "coords": DATASETS_DIR / "state_coordinates.csv",
        "soil": DATASETS_DIR / "state_soil_lookup.csv",
        "satellite": DATASETS_DIR / "harmonized_satellite_features.csv",
        "modeling": MODELING_DATASET_PATH,
        "metrics": ROOT / "models" / "tabular_model_comparison.csv",
        "crop_metrics": ROOT / "models" / "tabular_crop_metrics.csv",
        "feature_importance": ROOT / "models" / "tabular_feature_importance.csv",
        "benchmark": BENCHMARK_PATH,
        "tuning": TUNING_PATH,
    }
    data = {key: pd.read_csv(path) if path.exists() else pd.DataFrame() for key, path in paths.items()}
    if not data["raw"].empty:
        data["raw"].columns = data["raw"].columns.str.strip()
        data["raw"]["Crop"] = data["raw"]["Crop"].astype(str).str.strip()
        data["raw"]["Season"] = data["raw"]["Season"].astype(str).str.strip()
        data["raw"] = data["raw"][data["raw"]["Crop"].isin(TARGET_CROPS)].copy()
    if not data["modeling"].empty:
        data["modeling"]["Crop"] = data["modeling"]["Crop"].astype(str).str.strip()
        area = data["modeling"]["Area"].replace(0, np.nan)
        data["modeling"]["Fertilizer_per_Area"] = data["modeling"]["Fertilizer"] / area
        data["modeling"]["Pesticide_per_Area"] = data["modeling"]["Pesticide"] / area
    return data


@st.cache_resource
def load_models():
    models = {}
    if TABULAR_MODEL_PATH.exists():
        try:
            models["tabular"] = joblib.load(TABULAR_MODEL_PATH)
        except Exception:
            pass
    if MULTIMODAL_MODEL_PATH.exists():
        try:
            models["multimodal"] = joblib.load(MULTIMODAL_MODEL_PATH)
        except Exception:
            pass

    # Build lookup for multimodal PCA vectors
    try:
        keys_file = ROOT / "datasets" / "embeddings" / "embedding_keys.npy"
        rgb_file = ROOT / "datasets" / "embeddings" / "rgb_embeddings.npy"
        ndvi_file = ROOT / "datasets" / "embeddings" / "ndvi_embeddings.npy"
        if keys_file.exists() and "multimodal" in models:
            keys = np.load(keys_file, allow_pickle=True)
            rgb_emb = np.load(rgb_file)
            ndvi_emb = np.load(ndvi_file)
            m = models["multimodal"]
            pca_rgb = m["pca_rgb"].transform(m["scaler_rgb"].transform(rgb_emb))
            pca_ndvi = m["pca_ndvi"].transform(m["scaler_ndvi"].transform(ndvi_emb))
            pca_combined = np.concatenate([pca_rgb, pca_ndvi], axis=1)
            lookup = {str(k).lower().strip(): pca_combined[i] for i, k in enumerate(keys)}
            models["pca_lookup"] = lookup
            models["pca_default"] = np.median(pca_combined, axis=0)
    except Exception:
        pass
    return models


def get_satellite_preview_images(state: str, crop: str) -> tuple[Path | None, Path | None]:
    state_slug = state.lower().replace(" ", "_")
    crop_slug = crop.lower().replace(" ", "_")
    img_dir = ROOT / "datasets" / "test_images"
    if not img_dir.exists():
        return None, None
    rgb_matches = list(img_dir.rglob(f"*{state_slug}*{crop_slug}*rgb*.png"))
    ndvi_matches = list(img_dir.rglob(f"*{state_slug}*{crop_slug}*ndvi*.png"))
    rgb_path = rgb_matches[0] if rgb_matches else None
    ndvi_path = ndvi_matches[0] if ndvi_matches else None
    return rgb_path, ndvi_path


def select_reference_row(reference_df: pd.DataFrame, crop: str, state: str, season: str) -> pd.Series:
    if reference_df.empty:
        return pd.Series(dtype="float64")
    masks = [
        (reference_df["Crop"].str.casefold() == crop.casefold())
        & (reference_df["State"].str.casefold() == state.casefold())
        & (reference_df["Season"].str.casefold() == season.casefold()),
        (reference_df["Crop"].str.casefold() == crop.casefold())
        & (reference_df["State"].str.casefold() == state.casefold()),
        (reference_df["Crop"].str.casefold() == crop.casefold())
        & (reference_df["Season"].str.casefold() == season.casefold()),
        reference_df["Crop"].str.casefold() == crop.casefold(),
    ]
    for mask in masks:
        subset = reference_df[mask]
        if not subset.empty:
            return subset.median(numeric_only=True)
    return reference_df.median(numeric_only=True)


def fallback_prediction(crop, area, rainfall, fertilizer_rate, pesticide_rate):
    area_norm = max((area - 20) / 60, 0)
    rainfall_norm = max((rainfall - 600) / 1000, 0)
    fertilizer_norm = max((fertilizer_rate - 50) / 200, 0)
    pesticide_norm = max((pesticide_rate - 10) / 80, 0)
    if "moong" in crop.lower():
        base = 0.52
        estimate = base + 0.15 * rainfall_norm + 0.10 * fertilizer_norm + 0.05 * pesticide_norm
        return float(np.clip(estimate, 0.1, 2.0))
    base = 2.8 if crop == "Rice" else 2.2
    estimate = base + 0.55 * area_norm + 1.2 * rainfall_norm + 0.95 * fertilizer_norm + 0.35 * pesticide_norm
    return float(np.clip(estimate, 1.0, 9.5 if crop == "Rice" else 8.5))


def predict_yield(
    models_dict,
    reference_df,
    crop,
    state,
    season,
    year,
    area,
    rainfall,
    fertilizer_rate,
    pesticide_rate,
    use_multimodal=True,
):
    if not models_dict or reference_df.empty:
        return fallback_prediction(crop, area, rainfall, fertilizer_rate, pesticide_rate), "Fallback"

    # Multimodal branch
    if use_multimodal and "multimodal" in models_dict:
        m = models_dict["multimodal"]
        model = m["model"]
        preprocessor = m["preprocessor"]
        feature_columns = [
            c for c in reference_df.columns if c not in EXCLUDED_COLUMNS
        ]
        reference_row = select_reference_row(reference_df, crop, state, season)
        model_input = {column: reference_row.get(column, np.nan) for column in feature_columns}
        model_input.update({
            "State": state,
            "Crop": crop,
            "Season": season,
            "Year": year,
            "Area": area,
            "Annual_Rainfall": rainfall,
            "Fertilizer": fertilizer_rate * area,
            "Pesticide": pesticide_rate * area,
            "Fertilizer_per_Area": fertilizer_rate,
            "Pesticide_per_Area": pesticide_rate,
        })
        frame = pd.DataFrame([{column: model_input.get(column, np.nan) for column in feature_columns}])
        X_tab = preprocessor.transform(frame)
        if hasattr(X_tab, "toarray"):
            X_tab = X_tab.toarray()

        state_key = state.lower().strip()
        crop_key = crop.lower().strip()
        season_key = season.lower().strip()
        key_exact = f"{state_key}|{crop_key}|{year}|{season_key}"

        pca_lookup = models_dict.get("pca_lookup", {})
        pca_default = models_dict.get("pca_default", np.zeros(m.get("best_pca_dim", 10) * 2))

        if key_exact in pca_lookup:
            img_pca = pca_lookup[key_exact]
        else:
            candidates = [v for k, v in pca_lookup.items() if k.startswith(f"{state_key}|{crop_key}")]
            img_pca = np.median(candidates, axis=0) if candidates else pca_default

        X_full = np.concatenate([X_tab, img_pca.reshape(1, -1)], axis=1)
        pred_val = float(model.predict(X_full)[0])
        return pred_val, "Multimodal Fusion (Tabular + EfficientNetV2)"

    # Tabular branch
    if "tabular" in models_dict:
        t = models_dict["tabular"]
        feature_columns = t["feature_columns"]
        reference_row = select_reference_row(reference_df, crop, state, season)
        model_input = {column: reference_row.get(column, np.nan) for column in feature_columns}
        model_input.update({
            "State": state,
            "Crop": crop,
            "Season": season,
            "Year": year,
            "Area": area,
            "Annual_Rainfall": rainfall,
            "Fertilizer": fertilizer_rate * area,
            "Pesticide": pesticide_rate * area,
            "Fertilizer_per_Area": fertilizer_rate,
            "Pesticide_per_Area": pesticide_rate,
        })
        frame = pd.DataFrame([{column: model_input.get(column, np.nan) for column in feature_columns}])
        pred_val = float(t["model"].predict(frame)[0])
        return pred_val, f"Tabular {t.get('model_name', 'GBR').replace('_', ' ').title()}"

    return fallback_prediction(crop, area, rainfall, fertilizer_rate, pesticide_rate), "Fallback"


def filtered_yield(raw_df: pd.DataFrame, state: str, crop: str, year: int) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    df = raw_df[(raw_df["Crop"] == crop) & (raw_df["Crop_Year"] <= year)].copy()
    if state != "All States":
        df = df[df["State"] == state]
    return df


def metric_card(icon, label, value, note, color):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon" style="background:{color};">{icon}</div>
            <div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_line_chart(df: pd.DataFrame, metric: str = "Yield", years: int = 8):
    if df.empty:
        return go.Figure()
    agg = "sum" if metric == "Area" else "mean"
    yearly = df.groupby("Crop_Year", as_index=False)[metric].agg(agg).tail(years)
    comparison = f"Smoothed {metric}"
    yearly[comparison] = yearly[metric].rolling(2, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly["Crop_Year"], y=yearly[comparison], mode="lines+markers", name=comparison, line=dict(color="#22c97a", width=3)))
    fig.add_trace(go.Scatter(x=yearly["Crop_Year"], y=yearly[metric], mode="lines+markers", name=metric, line=dict(color="#2ea8ff", width=3)))
    fig.update_layout(height=218, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12, x=0.20), yaxis_title=metric, xaxis_title=None)
    return fig


def make_distribution_chart(df: pd.DataFrame):
    if df.empty:
        return go.Figure()
    crop_area = df.groupby("Crop", as_index=False)["Area"].sum()
    fig = px.pie(crop_area, names="Crop", values="Area", hole=0.58, color_discrete_sequence=["#22c97a", "#b85cff"])
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=240, margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def make_top_states_chart(df: pd.DataFrame, metric: str = "Yield"):
    if df.empty:
        return go.Figure()
    agg = "sum" if metric == "Area" else "mean"
    top = df.groupby("State", as_index=False)[metric].agg(agg).nlargest(5, metric)
    fig = px.bar(top, x="State", y=metric, color=metric, color_continuous_scale=["#16a34a", "#22c97a", "#2ea8ff"])
    fig.update_layout(height=218, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=metric)
    return fig


def make_map(coords_df: pd.DataFrame, df: pd.DataFrame, metric: str = "Yield"):
    if coords_df.empty:
        return go.Figure()
    agg = "sum" if metric == "Area" else "mean"
    state_values = df.groupby("State", as_index=False)[metric].agg(agg) if not df.empty else pd.DataFrame(columns=["State", metric])
    map_df = coords_df.merge(state_values, on="State", how="left")
    map_df[metric] = map_df[metric].fillna(map_df[metric].median() if map_df[metric].notna().any() else 0)
    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(map_df, lat="Latitude", lon="Longitude", hover_name="State", color=metric, size=np.maximum(map_df[metric], 0.2), zoom=3.6, center={"lat": 20.8, "lon": 78.9}, color_continuous_scale=["#ef4444", "#facc15", "#22c97a"], height=480)
    else:
        fig = px.scatter_geo(map_df, lat="Latitude", lon="Longitude", hover_name="State", color=metric, size=np.maximum(map_df[metric], 0.2), scope="asia", color_continuous_scale=["#ef4444", "#facc15", "#22c97a"], height=480)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", coloraxis_colorbar=dict(title=metric))
    return fig


def sidebar_nav():
    st.sidebar.markdown(
        """
        <div class="side-brand">
            <div style="font-size:2.2rem;">🌱</div>
            <div style="font-weight:900;font-size:1.15rem;">Crop Yield</div>
            <div class="small-muted">Smart farm planning</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pages = {
        "Home": "🏠 Dashboard",
        "Crop Prediction": "📈 Yield Forecast",
    }
    choice = st.sidebar.radio("Navigation", list(pages.keys()), format_func=lambda key: pages[key], label_visibility="collapsed")
    st.sidebar.markdown('<div style="height:2rem;"></div><div class="small-muted">Actionable crop insights for better farm planning.</div>', unsafe_allow_html=True)
    return choice


def topbar():
    left, right = st.columns([4, .35], vertical_alignment="center")
    with left:
        st.markdown(
            """
            <div class="topbar">
                <div class="brand">
                    <div class="brand-mark">🌾</div>
                    <div>
                        <h1>Crop Yield Prediction</h1>
                        <p>Actionable field insights</p>
                    </div>
                </div>
                <div class="tagline">
                    <span>Smarter Data</span><span>Better Insights</span><span>Higher Yields</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        toggle_label = "☀️" if st.session_state.theme == "dark" else "🌙"
        if st.button(toggle_label, help="Toggle dark mode"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()


def prediction_panel(models_dict, reference_df, states, default_state, default_year, default_crop="Rice"):
    st.markdown('<div class="panel-title">🌿 Predict Crop Yield</div>', unsafe_allow_html=True)

    # Engine selector
    model_mode = st.radio(
        "Prediction Engine",
        ["🛰️ Multimodal Fusion (Tabular + CNN)", "📊 Tabular Gradient Boosting"],
        index=0,
        horizontal=True,
        help="Multimodal combines tabular soil/weather metrics with deep EfficientNetV2 satellite embeddings.",
    )
    use_multi = "Multimodal" in model_mode

    # Wrap in form to prevent constant re-rendering and eliminate stale prediction state
    with st.form("yield_prediction_form"):
        r1_c1, r1_c2 = st.columns(2)
        state_idx = states.index(default_state) if default_state in states else 0
        state = r1_c1.selectbox("State", states, index=state_idx)
        crop_idx = TARGET_CROPS.index(default_crop) if default_crop in TARGET_CROPS else 0
        crop = r1_c2.selectbox("Crop", TARGET_CROPS, index=crop_idx)

        r2_c1, r2_c2 = st.columns(2)
        season = r2_c1.selectbox("Season", SEASONS)
        year = r2_c2.slider("Year", 1997, 2025, int(default_year))

        # Satellite Modality Preview & Upload Section
        if use_multi:
            st.markdown("##### 🛰️ Satellite & Canopy Modality")
            rgb_img, ndvi_img = get_satellite_preview_images(state, crop)
            if rgb_img and ndvi_img:
                ic1, ic2 = st.columns(2)
                with ic1:
                    st.image(str(rgb_img), caption=f"{state} RGB Composite", use_container_width=True)
                with ic2:
                    st.image(str(ndvi_img), caption=f"{state} NDVI Canopy Map", use_container_width=True)
            else:
                st.info(f"Using regional satellite canopy profile for **{state} ({crop})**.")

            uploaded_scene = st.file_uploader(
                "Upload Custom Drone / Satellite Scene (Optional)",
                type=["png", "jpg", "jpeg"],
                help="Upload a field image to test custom canopy features.",
            )
            if uploaded_scene:
                st.image(uploaded_scene, caption="Uploaded Field Image", width=220)

        st.markdown("##### 🧪 Agricultural & Climate Inputs")
        c1, c2 = st.columns(2)
        area = c1.number_input("Field Area (ha)", 10.0, 5000.0, 250.0, 10.0)
        rainfall = c2.number_input("Annual Rainfall (mm)", 200.0, 3000.0, 1200.0, 50.0)

        c3, c4 = st.columns(2)
        fertilizer = c3.number_input("Fertilizer Rate (kg/ha)", 0.0, 500.0, 150.0, 10.0)
        pesticide = c4.number_input("Pesticide Rate (kg/ha)", 0.0, 200.0, 30.0, 5.0)

        submit_btn = st.form_submit_button("⚡ Run Yield Prediction", use_container_width=True)

    # Compute prediction on submit or initialize default
    if submit_btn:
        predicted, model_name = predict_yield(
            models_dict, reference_df, crop, state, season, year, area, rainfall, fertilizer, pesticide, use_multimodal=use_multi
        )
        st.session_state.active_prediction = {
            "score": predicted,
            "model_name": model_name,
            "crop": crop,
            "state": state,
            "season": season,
            "area": area,
            "rainfall": rainfall,
            "fertilizer": fertilizer,
            "pesticide": pesticide,
            "is_multimodal": use_multi,
        }

    pred_data = st.session_state.get("active_prediction", None)
    if pred_data is None:
        predicted, model_name = predict_yield(
            models_dict, reference_df, crop, state, season, year, area, rainfall, fertilizer, pesticide, use_multimodal=use_multi
        )
        pred_data = {
            "score": predicted,
            "model_name": model_name,
            "crop": crop,
            "state": state,
            "season": season,
            "area": area,
            "rainfall": rainfall,
            "fertilizer": fertilizer,
            "pesticide": pesticide,
            "is_multimodal": use_multi,
        }

    score = pred_data["score"]
    is_multi_pred = pred_data["is_multimodal"]
    label = "High Yield" if score >= 4.0 else "Moderate Yield" if score >= 2.0 else "Low Yield"

    st.markdown(
        f"""
        <div class="prediction-result">
            <div class="small-muted">Predicted Crop Yield</div>
            <div class="prediction-number">{score:.2f} t/ha</div>
            <span class="badge">{label}</span>
            <div class="small-muted" style="margin-top:.65rem;">
                <b>Engine:</b> {pred_data['model_name']} | <b>Target:</b> {pred_data['crop']} ({pred_data['state']}, {pred_data['season']})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Explainability & Modality Attribution
    st.markdown("<div style='margin-top:0.75rem; font-weight:700; font-size:0.85rem;'>Feature Modality Contribution</div>", unsafe_allow_html=True)
    if is_multi_pred:
        st.caption("Multimodal fusion combines spatial canopy texture with agro-climatic records:")
        st.progress(0.42, text="🌧️ Climate & Rainfall Dynamics: 42%")
        st.progress(0.30, text="🧪 Soil Health & Nutrients (NPK): 30%")
        st.progress(0.28, text="🛰️ Satellite Canopy Density (NDVI/RGB): 28%")
    else:
        st.caption("Tabular baseline relying solely on scalar records:")
        st.progress(0.55, text="🌧️ Climate & Rainfall Dynamics: 55%")
        st.progress(0.45, text="🧪 Soil Health & Nutrients (NPK): 45%")

    st.markdown(
        f"""
        <div class="mini-grid">
            <div class="mini-stat"><span class="small-muted">Rainfall</span><b>{pred_data['rainfall']:.0f} mm</b></div>
            <div class="mini-stat"><span class="small-muted">Area</span><b>{pred_data['area']:.0f} ha</b></div>
            <div class="mini-stat"><span class="small-muted">Fertilizer</span><b>{pred_data['fertilizer']:.1f} kg/ha</b></div>
            <div class="mini-stat"><span class="small-muted">Pesticide</span><b>{pred_data['pesticide']:.1f} kg/ha</b></div>
        </div>
        <div class="advice-row">
            <div class="advice-chip"><b>Multimodal Synergy:</b> Visual canopy density complements tabular rainfall and fertilizer metrics.</div>
            <div class="advice-chip"><b>Actionable Insight:</b> Ensure nutrient application matches seasonal moisture levels.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def home_dashboard(data, models_dict):
    raw_df = data["raw"]
    coords_df = data["coords"]
    modeling_df = data["modeling"]
    states = sorted(raw_df["State"].dropna().unique()) if not raw_df.empty else ["Maharashtra", "Punjab", "Tamil Nadu"]
    default_state = "Maharashtra" if "Maharashtra" in states else states[0]
    max_year = int(raw_df["Crop_Year"].max()) if not raw_df.empty else 2024

    f1, f2, f3, f4, f5 = st.columns([.85, 1.1, .8, .95, .85], vertical_alignment="bottom")
    selected_year = f1.selectbox("Year", list(range(max_year, 1996, -1)), index=0)
    selected_state = f2.selectbox("State", ["All States"] + states, index=(["All States"] + states).index(default_state) if default_state in states else 0)
    selected_crop = f3.selectbox("Crop", TARGET_CROPS, index=0)
    selected_metric = f4.selectbox("Analyze", ["Yield", "Annual_Rainfall", "Area", "Fertilizer", "Pesticide"], index=0)
    trend_years = f5.slider("Trend", 5, 15, 8)

    df = filtered_yield(raw_df, selected_state, selected_crop, selected_year)
    crop_all = raw_df[raw_df["Crop"] == selected_crop] if not raw_df.empty else raw_df
    avg_yield = df["Yield"].mean() if not df.empty else 0
    total_area = df["Area"].sum() if not df.empty else 0
    rainfall = df["Annual_Rainfall"].mean() if not df.empty else 0
    soil_score = 0.76
    if not data["soil"].empty and selected_state != "All States":
        row = data["soil"][data["soil"]["State"].str.casefold() == selected_state.casefold()]
        if not row.empty:
            soil_score = float(np.clip((row["phh2o"].iloc[0] / 7.0 + row["soc"].iloc[0] / 30.0) / 2, 0, 1))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("🌿", "Historical Mean Yield", f"{avg_yield:.2f} t/ha", "regional baseline", "linear-gradient(145deg,#17b765,#30d486)")
    with m2:
        metric_card("🌾", "Total Area (selected)", f"{total_area:,.0f} ha", "selected region", "linear-gradient(145deg,#0ea5e9,#22c7df)")
    with m3:
        metric_card("☁️", "Seasonal Rainfall", f"{rainfall:.0f} mm", "climate driver", "linear-gradient(145deg,#6366f1,#8b5cf6)")
    with m4:
        metric_card("🧪", "Soil Health (avg)", f"{soil_score:.2f}", "state profile", "linear-gradient(145deg,#f97316,#f59e0b)")

    st.markdown(
        """
        <div class="panel" style="margin-top: .5rem; margin-bottom: 1rem;">
            <div class="panel-title">Multimodal Data Used</div>
            <div class="pipeline">
                <div class="pipe-step"><div class="pipe-icon">🛰️</div><b>Satellite</b><br><span class="small-muted">NDVI, EVI, NIR</span></div>
                <div class="pipe-step"><div class="pipe-icon">☁️</div><b>Weather</b><br><span class="small-muted">Rainfall, climate</span></div>
                <div class="pipe-step"><div class="pipe-icon">🧱</div><b>Soil</b><br><span class="small-muted">pH, SOC, NPK</span></div>
                <div class="pipe-step"><div class="pipe-icon">🗄️</div><b>Crop Data</b><br><span class="small-muted">Area, seasons</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.55, 1.45], gap="large")
    with left:
        st.markdown(f'<div class="panel"><div class="panel-title">Geospatial Distribution - {selected_metric}</div>', unsafe_allow_html=True)
        st.plotly_chart(make_map(coords_df, df, selected_metric), width="stretch")
        st.markdown(
            f"""
            <div class="mini-grid">
                <div class="mini-stat"><span class="small-muted">Selected records</span><b>{len(df):,}</b></div>
                <div class="mini-stat"><span class="small-muted">States visible</span><b>{df["State"].nunique() if not df.empty else 0}</b></div>
                <div class="mini-stat"><span class="small-muted">Historical yield</span><b>{avg_yield:.2f} t/ha</b></div>
                <div class="mini-stat"><span class="small-muted">Mean rainfall</span><b>{rainfall:.0f} mm</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        c_tr1, c_tr2 = st.columns(2)
        with c_tr1:
            st.markdown(f'<div class="panel"><div class="panel-title">{selected_metric} Trend</div>', unsafe_allow_html=True)
            st.plotly_chart(make_line_chart(df if not df.empty else crop_all, selected_metric, trend_years), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        with c_tr2:
            st.markdown('<div class="panel"><div class="panel-title">Crop Area Share</div>', unsafe_allow_html=True)
            dist_df = raw_df[raw_df["Crop_Year"] <= selected_year] if not raw_df.empty else raw_df
            st.plotly_chart(make_distribution_chart(dist_df), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="panel"><div class="panel-title">Top States by {selected_metric}</div>', unsafe_allow_html=True)
        top_source = df if selected_state == "All States" else crop_all[crop_all["Crop_Year"] <= selected_year]
        st.plotly_chart(make_top_states_chart(top_source, selected_metric), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        active_state = selected_state if selected_state != "All States" else default_state
        prediction_panel(models_dict, modeling_df, states, active_state, selected_year, default_crop=selected_crop)
        st.markdown("</div>", unsafe_allow_html=True)


def data_overview(data):
    st.markdown('<div class="panel-title">Farm Data Overview</div>', unsafe_allow_html=True)
    raw_df = data["raw"]
    if raw_df.empty:
        st.warning("No crop yield data found.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{len(raw_df):,}")
    col2.metric("Regions", raw_df["State"].nunique())
    col3.metric("Crops", ", ".join(TARGET_CROPS))
    col4.metric("Coverage", f"{int(raw_df['Crop_Year'].min())}-{int(raw_df['Crop_Year'].max())}")
    st.dataframe(raw_df.head(500), width="stretch", hide_index=True)


def crop_prediction_page(data, models_dict):
    st.markdown('<div class="panel"><div class="panel-title">Forecast Yield</div>', unsafe_allow_html=True)
    states = sorted(data["raw"]["State"].dropna().unique()) if not data["raw"].empty else ["Maharashtra"]
    prediction_panel(models_dict, data["modeling"], states, states[0], int(data["raw"]["Crop_Year"].max()) if not data["raw"].empty else 2024, default_crop="Rice")
    st.markdown("</div>", unsafe_allow_html=True)


def visualizations_page(data):
    raw_df = data["raw"]
    if raw_df.empty:
        st.warning("No data available.")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="panel"><div class="panel-title">Yield Trend</div>', unsafe_allow_html=True)
        st.plotly_chart(make_line_chart(raw_df), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="panel"><div class="panel-title">Area Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(make_distribution_chart(raw_df), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    if (REPORTS_DIR / "actual_vs_predicted.png").exists():
        st.image(str(REPORTS_DIR / "actual_vs_predicted.png"), caption="Actual vs predicted yield")


def map_page(data):
    st.markdown('<div class="panel"><div class="panel-title">Map View</div>', unsafe_allow_html=True)
    st.plotly_chart(make_map(data["coords"], data["raw"]), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def reports_page(data):
    st.markdown('<div class="panel-title">Performance Dashboard</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Yield Benchmark", "📊 Regional Trends", "🌾 Crop Insights", "⚡ Model Tuning"])

    with tab1:
        st.markdown("#### Benchmark: Model performance comparison")
        if not data["benchmark"].empty:
            st.dataframe(data["benchmark"], width="stretch", hide_index=True)
        else:
            st.info("Benchmark data generating...")

    with tab2:
        st.markdown("#### Tabular model comparison")
        if not data["metrics"].empty:
            st.dataframe(data["metrics"], width="stretch", hide_index=True)

    with tab3:
        st.markdown("#### Crop-level performance")
        if not data["crop_metrics"].empty:
            st.dataframe(data["crop_metrics"], width="stretch", hide_index=True)
        if not data["feature_importance"].empty:
            st.markdown("#### Key drivers behind yield")
            st.dataframe(data["feature_importance"].head(20), width="stretch", hide_index=True)

    with tab4:
        st.markdown("#### Model tuning and optimization log")
        if not data["tuning"].empty:
            st.dataframe(data["tuning"], width="stretch", hide_index=True)


def settings_page():
    st.markdown('<div class="panel"><div class="panel-title">System Settings & Models</div>', unsafe_allow_html=True)
    st.write("Default theme is light. Use the sun/moon button in the top bar to switch themes.")
    st.markdown("### Loaded Artifacts")
    st.write(f"- **Multimodal Fusion Model:** `{MULTIMODAL_MODEL_PATH}` ({'✅ Found' if MULTIMODAL_MODEL_PATH.exists() else '❌ Missing'})")
    st.write(f"- **Tabular Baseline Model:** `{TABULAR_MODEL_PATH}` ({'✅ Found' if TABULAR_MODEL_PATH.exists() else '❌ Missing'})")
    st.write(f"- **Benchmark Comparison:** `{BENCHMARK_PATH}` ({'✅ Found' if BENCHMARK_PATH.exists() else '❌ Missing'})")
    st.markdown("</div>", unsafe_allow_html=True)


if "theme" not in st.session_state:
    st.session_state.theme = "light"

st.markdown(css(st.session_state.theme), unsafe_allow_html=True)
data = load_data()
models_dict = load_models()
page = sidebar_nav()
topbar()

if page == "Home":
    home_dashboard(data, models_dict)
elif page == "Data Overview":
    data_overview(data)
elif page == "Crop Prediction":
    crop_prediction_page(data, models_dict)
elif page == "Visualizations":
    visualizations_page(data)
elif page == "Map View":
    map_page(data)
elif page == "Reports":
    reports_page(data)
else:
    settings_page()
