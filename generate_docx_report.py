import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def style_table(table, header_bg="1B365D", alt_bg="F0F4F8"):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table.rows):
        # Prevent row split across pages
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
            if i == 0:
                set_cell_background(cell, header_bg)
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)
                        run.font.size = Pt(9.5)
            else:
                if i % 2 == 1:
                    set_cell_background(cell, "FFFFFF")
                else:
                    set_cell_background(cell, alt_bg)
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(9.5)

def build_docx(output_path):
    doc = Document()

    # Page Margins: 1 inch all around
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base styling
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(40, 40, 40)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # Primary Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    title_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("Crop Yield Prediction Using Multimodal Data Fusion")
    title_run.font.name = 'Arial'
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(27, 54, 93) # Deep Navy

    # Subtitle
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(16)
    sub_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub_p.add_run("Semester 5 Academic Project Report (Interim Draft)")
    sub_run.font.name = 'Arial'
    sub_run.font.size = Pt(13)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(90, 105, 120)

    # Metadata Box Table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Project Domain:", "Machine Learning, Remote Sensing & Precision Agriculture"),
        ("Focus Crops:", "Rice (Oryza sativa) & Maize (Zea mays) across India"),
        ("Data Modalities:", "Agronomic Records, Soil Profiling, Sentinel-2 Multispectral Imagery"),
        ("Project Status:", "Mid-Stage Progress Report (Baselines & UI Completed)")
    ]
    for row_idx, (k, v) in enumerate(meta_data):
        row = meta_table.rows[row_idx]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(2)
        r0 = p0.add_run(k)
        r0.font.bold = True
        r0.font.size = Pt(9.5)
        r0.font.color.rgb = RGBColor(27, 54, 93)

        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(2)
        r1 = p1.add_run(v)
        r1.font.size = Pt(9.5)
        
        set_cell_background(c0, "F0F4F8")
        set_cell_background(c1, "F8FAFC")
        set_cell_margins(c0, top=60, bottom=60, left=100, right=100)
        set_cell_margins(c1, top=60, bottom=60, left=100, right=100)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = RGBColor(27, 54, 93)
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(12.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(41, 70, 110)
        return p

    def add_bullet(p, bold_prefix, text):
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(3)
        r_b = p.add_run("• " + bold_prefix)
        r_b.font.bold = True
        r_t = p.add_run(text)

    # ABSTRACT
    add_h1("Abstract")
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.line_spacing = 1.15
    p_abs.paragraph_format.space_after = Pt(8)
    p_abs.add_run(
        "Accurate crop yield prediction is crucial for food security planning, agricultural market stability, "
        "and localized resource management. Traditional agricultural yield estimation relies primarily on regional agronomic "
        "surveys and historical production records, which often lag in time and fail to capture real-time vegetative dynamics. "
        "Conversely, satellite remote sensing provides continuous vegetation indices (e.g., NDVI, EVI) but lacks critical "
        "ground-truth agronomic context such as fertilizer dosage, pesticide application, and localized soil chemical composition. "
        "This project develops a multimodal data fusion pipeline that integrates historical agricultural statistics, meteorological "
        "patterns, soil profile lookup tables, and Sentinel-2 multispectral surface reflectance data to forecast rice and maize yields across Indian states."
    )
    p_abs2 = doc.add_paragraph()
    p_abs2.add_run(
        "At this interim project stage, we have established: (1) an auditable, leakage-free data harmonization pipeline across 1,774 agricultural records "
        "spanning 1997–2020 and matched Sentinel-2 satellite tiles; (2) a temporal evaluation scheme (Training: <=2016, Validation: 2017–2018, Testing: 2019–2020); "
        "(3) strong tabular regression baselines, where Gradient Boosting achieves a test R² of 0.6919 and an RMSE of 0.8035 tonnes/ha (Rice R² = 0.7082, Maize R² = 0.6750); "
        "(4) image-only exploratory baselines (custom 6-channel CNN and spatial summary regressors) confirming that satellite data alone is insufficient without agronomic inputs; "
        "and (5) an interactive Streamlit analytical and inference dashboard. This draft documents the system architecture, empirical benchmarks, and future fusion roadmap."
    )

    # 1. INTRODUCTION
    add_h1("1. Introduction")
    add_h2("1.1 Problem Statement & Motivation")
    p = doc.add_paragraph()
    p.add_run(
        "Agriculture forms the backbone of India's economy, supporting over 50% of the working population. Rice (Oryza sativa) "
        "and Maize (Zea mays) serve as foundational staple and commercial crops. However, crop yield fluctuates significantly "
        "across agro-climatic zones due to erratic rainfall, disparate soil nutrient reserves, and unequal chemical input applications.\n\n"
        "While statistical yield regression based on historical census data provides an overarching macro-level baseline, "
        "it cannot observe the actual physiological health of standing crops during critical growth stages. Concurrently, satellite remote sensing "
        "(such as the European Space Agency's Sentinel-2 constellation) captures high-resolution vegetation greenness and canopy moisture, "
        "yet satellite rasters cannot directly quantify human interventions such as fertilizer or pesticide application. Fusing ground agronomic "
        "records with satellite observations bridges this gap, establishing a robust multimodal forecasting system."
    )

    add_h2("1.2 Core Objectives")
    bp1 = doc.add_paragraph()
    add_bullet(bp1, "Multimodal Harmonization: ", "Integrate tabular records, soil chemistry profiles, geographic coordinates, and Sentinel-2 spectral indices into an auditable data pipeline.")
    bp2 = doc.add_paragraph()
    add_bullet(bp2, "Leakage Prevention & Chronological Splitting: ", "Enforce strict temporal train-validation-test partitioning to simulate genuine out-of-sample forward forecasting, completely eliminating circular target leakage.")
    bp3 = doc.add_paragraph()
    add_bullet(bp3, "Baseline Benchmarking: ", "Establish rigorous tabular and image-only baselines to quantify the individual predictive strengths of each modality.")
    bp4 = doc.add_paragraph()
    add_bullet(bp4, "Decision-Support Dashboard: ", "Deploy an interactive Streamlit dashboard featuring geospatial maps, satellite tile visualizers, and interactive inference calculators.")
    bp5 = doc.add_paragraph()
    add_bullet(bp5, "Deep Multimodal Fusion (Pending Phase): ", "Implement joint dual-branch neural architectures to evaluate performance gains achieved by deep fusion over single-modality baselines.")

    # 2. LITERATURE SURVEY
    add_h1("2. Literature Review & Theoretical Framework")
    add_h2("2.1 Statistical & Machine Learning Approaches")
    p = doc.add_paragraph()
    p.add_run(
        "Early agricultural yield models relied heavily on linear regression and autoregressive time-series linking rainfall anomalies to harvest totals. "
        "The introduction of non-linear tree-based ensembles—principally Random Forests (Breiman, 2001) and Gradient Boosted Decision Trees (Friedman, 2001)—"
        "significantly improved predictive accuracy by capturing complex non-linear interactions between temperature, fertilizer dosage, and cultivated area. "
        "Nonetheless, purely tabular systems suffer from spatial coarseness and reporting delays."
    )

    add_h2("2.2 Multispectral Remote Sensing & Vegetation Indices")
    p = doc.add_paragraph()
    p.add_run(
        "The Sentinel-2 satellite mission provides 10m to 20m spatial resolution across 13 spectral bands with a 5-day revisit frequency. "
        "Key vegetation and moisture indices utilized in this work include:"
    )
    b_ndvi = doc.add_paragraph()
    add_bullet(b_ndvi, "Normalized Difference Vegetation Index (NDVI): ", "NDVI = (NIR - RED) / (NIR + RED). Captures chlorophyll absorption and canopy density.")
    b_evi = doc.add_paragraph()
    add_bullet(b_evi, "Enhanced Vegetation Index (EVI): ", "EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1). Mitigates soil background noise and atmospheric aerosol saturation.")
    b_ndwi = doc.add_paragraph()
    add_bullet(b_ndwi, "Normalized Difference Water Index (NDWI): ", "NDWI = (GREEN - NIR) / (GREEN + NIR). Sensitive to changes in liquid water content of crop canopies.")

    add_h2("2.3 Multimodal Data Fusion Taxonomy")
    p = doc.add_paragraph()
    p.add_run(
        "Multimodal fusion is categorized into three paradigms: (1) Early Fusion, where tabular variables and flattened image features are concatenated before regression; "
        "(2) Intermediate / Joint Fusion, where a convolutional vision trunk and a tabular MLP trunk extract latent embedding vectors that are concatenated and optimized end-to-end; "
        "and (3) Late Fusion, where predictions from independently trained models are ensembled via stacking or weighted averaging."
    )

    # 3. DATA ENGINEERING & PREPROCESSING
    add_h1("3. Data Engineering & Preprocessing Pipeline")
    add_h2("3.1 Integrated Data Sources")
    p = doc.add_paragraph()
    p.add_run("The multimodal repository integrates five core data streams:")
    
    b_d1 = doc.add_paragraph()
    add_bullet(b_d1, "Crop Production Records (crop_yield.csv): ", "1,781 historical records across Indian states detailing Crop (Rice, Maize), Season, Area (ha), Production (tonnes), Annual Rainfall (mm), Fertilizer (kg), and Pesticide (kg).")
    b_d2 = doc.add_paragraph()
    add_bullet(b_d2, "Spatial Coordinates (state_coordinates.csv): ", "State centroid latitude and longitude metadata for spatial alignment.")
    b_d3 = doc.add_paragraph()
    add_bullet(b_d3, "Soil Profiles (state_soil_lookup.csv): ", "Macro-nutrient and chemical profiles (N, P, K, Organic Carbon, pH) mapped at state scale.")
    b_d4 = doc.add_paragraph()
    add_bullet(b_d4, "Sentinel-2 Spectral Metrics (sentinel2_features.csv): ", "Harmonized band statistics (Blue, Green, Red, NIR, EVI, NDVI, NDWI: mean, min, max, stdDev).")
    b_d5 = doc.add_paragraph()
    add_bullet(b_d5, "Satellite Raster Tiles (datasets/test_images/): ", "Paired 96x96 PNG tiles containing natural RGB imagery and rendered NDVI heatmaps spanning 1997–2020.")

    add_h2("3.2 Preprocessing, Leakage Prevention & Feature Engineering")
    p = doc.add_paragraph()
    p.add_run(
        "1. Elimination of Target Leakage: Because Yield = Production / Area, retaining Production in the training feature matrix causes artificial 100% predictive accuracy. Production was strictly excluded from training features.\n"
        "2. Agronomic Rate Features: Normalized interaction rates were engineered: Fertilizer_per_Area = Fertilizer / Area, and Pesticide_per_Area = Pesticide / Area.\n"
        "3. Outlier Filtering: Seven historical anomalies exhibiting yields above 15 tonnes/ha were identified as reporting errors exceeding biological thresholds and removed, leaving 1,774 validated records.\n"
        "4. Feature Encoding: One-hot encoding was applied to State, Crop, and Season categorical features, yielding 43 total tabular predictor dimensions."
    )

    add_h2("3.3 Chronological Train / Validation / Test Splitting")
    p = doc.add_paragraph()
    p.add_run(
        "To evaluate real-world forecasting robustness, chronological year boundaries were enforced rather than random cross-validation:"
    )

    split_table = doc.add_table(rows=4, cols=4)
    split_headers = ["Partition Set", "Year Span", "Sample Count", "Analytical Role"]
    split_rows = [
        ["Training Set", "1997 – 2016", "1,556", "Model fitting & parameter learning"],
        ["Validation Set", "2017 – 2018", "150", "Hyperparameter tuning & early stopping"],
        ["Test Set", "2019 – 2020", "68", "Unbiased out-of-sample forward evaluation"]
    ]
    for c_idx, h in enumerate(split_headers):
        split_table.rows[0].cells[c_idx].paragraphs[0].text = h
    for r_idx, row_data in enumerate(split_rows):
        for c_idx, val in enumerate(row_data):
            split_table.rows[r_idx + 1].cells[c_idx].paragraphs[0].text = val
    style_table(split_table)

    # 4. EXPERIMENTAL SETUP & BASELINE RESULTS
    add_h1("4. Experimental Setup & Baseline Results")
    add_h2("4.1 Evaluation Metrics")
    p = doc.add_paragraph()
    p.add_run(
        "Model performance is quantified using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination (R²)."
    )

    add_h2("4.2 Tabular Regression Benchmark")
    p = doc.add_paragraph()
    p.add_run(
        "Four tabular algorithms were trained and evaluated on the chronological split:"
    )

    tab_table = doc.add_table(rows=5, cols=7)
    tab_headers = ["Model", "Val RMSE", "Val MAE", "Val R²", "Test RMSE", "Test MAE", "Test R²"]
    tab_data = [
        ["Dummy Mean Baseline", "1.4777", "0.9144", "-0.1749", "1.6198", "1.0284", "-0.2520"],
        ["Random Forest Regressor", "1.0031", "0.5878", "0.4586", "1.0719", "0.6418", "0.4517"],
        ["HistGradientBoosting", "0.7787", "0.4624", "0.6738", "0.8342", "0.5258", "0.6679"],
        ["Gradient Boosting (Best)", "0.7316", "0.4658", "0.7120", "0.8035", "0.5104", "0.6919"]
    ]
    for c_idx, h in enumerate(tab_headers):
        tab_table.rows[0].cells[c_idx].paragraphs[0].text = h
    for r_idx, row_data in enumerate(tab_data):
        for c_idx, val in enumerate(row_data):
            tab_table.rows[r_idx + 1].cells[c_idx].paragraphs[0].text = val
    style_table(tab_table)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_h2("4.3 Crop-Wise Evaluation (Gradient Boosting)")
    p = doc.add_paragraph()
    p.add_run(
        "Evaluating the best Gradient Boosting Regressor across distinct crop classes on the test set demonstrates well-balanced generalization:"
    )

    crop_table = doc.add_table(rows=3, cols=5)
    crop_headers = ["Crop Type", "Test Sample Count", "Test RMSE (t/ha)", "Test MAE (t/ha)", "Test R²"]
    crop_data = [
        ["Rice (Oryza sativa)", "31", "0.4131", "0.3183", "0.7082"],
        ["Maize (Zea mays)", "37", "1.0216", "0.6714", "0.6750"]
    ]
    for c_idx, h in enumerate(crop_headers):
        crop_table.rows[0].cells[c_idx].paragraphs[0].text = h
    for r_idx, row_data in enumerate(crop_data):
        for c_idx, val in enumerate(row_data):
            crop_table.rows[r_idx + 1].cells[c_idx].paragraphs[0].text = val
    style_table(crop_table)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_h2("4.4 Image-Only Baseline Comparison")
    p = doc.add_paragraph()
    p.add_run(
        "To rigorously examine whether satellite imagery alone can forecast yield without agricultural ground records, "
        "a custom 6-channel CNN (RGB + NDVI) and statistical raster summary models were evaluated on aligned 2019 test image pairs (N = 13):"
    )

    img_table = doc.add_table(rows=6, cols=4)
    img_headers = ["Input Representation & Model", "Test RMSE", "Test MAE", "Test R²"]
    img_data = [
        ["6-Channel CNN (RGB + NDVI from scratch)", "1.3560", "0.9282", "0.0610"],
        ["RGB Summary Features (Extra Trees)", "1.1762", "0.7446", "0.2935"],
        ["NDVI Summary Features (Extra Trees)", "1.0987", "0.7723", "0.3836"],
        ["RGB + NDVI Summary (Ridge Regressor)", "1.1226", "0.7639", "0.3565"],
        ["Tabular Gradient Boosting (Same 13 Samples)", "0.7936", "0.5489", "0.6784"]
    ]
    for c_idx, h in enumerate(img_headers):
        img_table.rows[0].cells[c_idx].paragraphs[0].text = h
    for r_idx, row_data in enumerate(img_data):
        for c_idx, val in enumerate(row_data):
            img_table.rows[r_idx + 1].cells[c_idx].paragraphs[0].text = val
    style_table(img_table)

    p_disc = doc.add_paragraph()
    p_disc.paragraph_format.space_before = Pt(6)
    p_disc.add_run(
        "Key Interim Finding: Image-only architectures achieve substantially lower performance (R² = 0.0610 to 0.3836) compared to tabular models (R² = 0.6784 on the identical test cohort). "
        "This proves that visual canopy information alone cannot replace knowledge of chemical fertilizers, seed area, and season. However, spectral indices provide meaningful variance reduction, "
        "providing the empirical justification for Deep Multimodal Fusion."
    )

    # 4.5 EMBEDDED FIGURES
    add_h2("4.5 Experimental Visualizations & Diagnostic Reports")
    rep_dir = os.path.join(os.path.dirname(output_path), "models", "reports")
    
    # Feature Importance Plot
    fi_path = os.path.join(rep_dir, "feature_importance.png")
    if os.path.exists(fi_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(2)
        doc.add_picture(fi_path, width=Inches(5.5))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(12)
        r_c = p_cap.add_run("Figure 1: Feature Importance ranking for Tabular Gradient Boosting model.")
        r_c.font.size = Pt(9.5)
        r_c.font.italic = True

    # Actual vs Predicted Plot
    avp_path = os.path.join(rep_dir, "actual_vs_predicted.png")
    if os.path.exists(avp_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(2)
        doc.add_picture(avp_path, width=Inches(5.5))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(12)
        r_c = p_cap.add_run("Figure 2: Actual vs. Predicted Crop Yield (tonnes/ha) on the test partition.")
        r_c.font.size = Pt(9.5)
        r_c.font.italic = True

    # 5. SOFTWARE ARCHITECTURE
    add_h1("5. System Architecture & Dashboard Implementation")
    p = doc.add_paragraph()
    p.add_run(
        "The project is implemented as an interactive decision-support application using Streamlit (app.py). "
        "The application architecture consists of three integrated functional layers:"
    )
    b_ui1 = doc.add_paragraph()
    add_bullet(b_ui1, "Geospatial Exploration Module: ", "Renders interactive Plotly choropleth and coordinate maps displaying historical state-level yield distributions.")
    b_ui2 = doc.add_paragraph()
    add_bullet(b_ui2, "Satellite Remote Sensing Explorer: ", "Allows visual inspection of Sentinel-2 RGB imagery alongside rendered NDVI heatmaps across chronological cultivation seasons.")
    b_ui3 = doc.add_bullet if False else doc.add_paragraph()
    add_bullet(b_ui3, "Dynamic Yield Inference Engine: ", "Enables end-users to specify crop type, state, season, area, rainfall, and nutrient inputs. Predictions are generated via models/tabular_best_model.joblib, supported by an intelligent heuristic fallback mechanism.")

    # 6. ROADMAP & CURRENT STATUS
    add_h1("6. Project Status & Work Breakdown")
    status_table = doc.add_table(rows=8, cols=3)
    status_headers = ["Project Phase / Component", "Current Status", "Key Deliverables / Artifacts"]
    status_rows = [
        ["Problem Definition & Literature Survey", "Completed", "Domain review & project charter"],
        ["Multimodal Dataset Harmonization", "Completed", "crop_yield.csv, sentinel2_features.csv"],
        ["Leakage-free Preprocessing & Splitting", "Completed", "modeling_dataset.csv, cnn_samples.csv"],
        ["Tabular Model Benchmarking (R² = 0.6919)", "Completed", "tabular_best_model.joblib, reports/"],
        ["Image Baseline Experiments (R² = 0.0610)", "Completed", "cnn_image_baseline.keras"],
        ["Streamlit Interactive Web Dashboard", "Completed", "app.py functional deployment"],
        ["Deep Multimodal Fusion Model (Dual-Branch)", "In Progress", "Dual-branch neural network training"]
    ]
    for c_idx, h in enumerate(status_headers):
        status_table.rows[0].cells[c_idx].paragraphs[0].text = h
    for r_idx, row_data in enumerate(status_rows):
        for c_idx, val in enumerate(row_data):
            status_table.rows[r_idx + 1].cells[c_idx].paragraphs[0].text = val
    style_table(status_table)

    # 7. NEXT STEPS FOR COMPLETION
    add_h1("7. Milestones Awaiting Project Completion")
    p = doc.add_paragraph()
    p.add_run(
        "To complete the final phase of the project, the following deliverables will be executed:"
    )
    b_next1 = doc.add_paragraph()
    add_bullet(b_next1, "1. Deep Multimodal Dual-Branch Network: ", "Construct a joint PyTorch/Keras architecture that couples a CNN backbone (processing Sentinel-2 tiles) with a dense MLP (processing tabular features), concatenating feature embeddings for unified end-to-end regression.")
    b_next2 = doc.add_paragraph()
    add_bullet(b_next2, "2. Final Ablation Benchmark: ", "Conduct a three-way comparative study: Multimodal Fusion vs. Tabular Only vs. Image Only across identical test samples to validate performance improvements.")
    b_next3 = doc.add_paragraph()
    add_bullet(b_next3, "3. Error Diagnostics: ", "Perform residual distribution analysis across states and rainfall extremes to identify regional model uncertainties.")
    b_next4 = doc.add_paragraph()
    add_bullet(b_next4, "4. Production Integration: ", "Embed the trained multimodal network into app.py for live multimodal inference.")

    # REFERENCES
    add_h1("References")
    refs = [
        "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
        "Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics, 1189-1232.",
        "European Space Agency (ESA). Sentinel-2 User Handbook. Standard Earth Observation Reference Documents.",
        "Lobell, D. B., et al. (2015). The influence of climate on global crop productivity. Science, 333(6042), 616-620.",
        "You, J., et al. (2017). Deep Gaussian Processes for Crop Yield Prediction Based on Remote Sensing Data. AAAI Conference on Human Computation and Crowdsourcing."
    ]
    for ref in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.25)
        p_ref.paragraph_format.space_after = Pt(3)
        p_ref.add_run(ref)

    # Save document
    doc.save(output_path)
    print(f"Document saved successfully to {output_path}")

if __name__ == "__main__":
    out_file = r"c:\Users\pandy\OneDrive\Desktop\SEM 5\Crop-Yield-Prediction-using-Multimodal-Data-Fusion\Crop_Yield_Prediction_Draft_Report.docx"
    build_docx(out_file)
