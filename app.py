"""
app.py
------
Synthetic dataset generated for educational AI/ML demonstration.

Streamlit web dashboard for the AI/ML Crop Advisory System.

Run with:
    streamlit run app.py
"""

import os
import sys
import pickle
import csv
import numpy as np
import streamlit as st
import pandas as pd

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI/ML Crop Advisory System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

NOTE = "Synthetic dataset generated for educational AI/ML demonstration."

MODEL_DIR = "models"
DATA_CSV  = os.path.join("data", "synthetic_crop_data.csv")

FEATURE_COLS = [
    "nitrogen_N", "phosphorus_P", "potassium_K",
    "temperature_C", "humidity_pct", "soil_pH", "rainfall_mm",
]

CROP_EMOJI = {
    "Rice": "🌾", "Wheat": "🌾", "Maize": "🌽", "Chickpea": "🫘",
    "Lentil": "🫘", "Cotton": "🌿", "Sugarcane": "🎋", "Soybean": "🫘",
    "Groundnut": "🥜", "Mungbean": "🫘", "Blackgram": "🫘",
    "Pigeonpea": "🫘", "Jute": "🌿", "Coffee": "☕", "Apple": "🍎",
    "Mango": "🥭", "Grapes": "🍇", "Watermelon": "🍉",
    "Papaya": "🍈", "Coconut": "🥥",
}

# ── load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    def _load(fname):
        path = os.path.join(MODEL_DIR, fname)
        with open(path, "rb") as f:
            return pickle.load(f)
    clf    = _load("crop_recommender.pkl")
    reg    = _load("yield_estimator.pkl")
    le     = _load("label_encoder.pkl")
    scaler = _load("feature_scaler.pkl")
    return clf, reg, le, scaler

@st.cache_data
def load_dataset():
    with open(DATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    df = pd.DataFrame(rows)
    for col in FEATURE_COLS + ["estimated_yield"]:
        df[col] = pd.to_numeric(df[col])
    return df

# ── prediction helper ─────────────────────────────────────────────────────────
def predict(features, clf, reg, le, scaler):
    X = np.array([features])
    X_s = scaler.transform(X)
    proba = clf.predict_proba(X_s)[0]
    top3_idx = proba.argsort()[::-1][:3]
    top3 = [(le.classes_[i], round(float(proba[i]) * 100, 1)) for i in top3_idx]
    yield_est = round(float(reg.predict(X_s)[0]), 2)
    return top3, yield_est

# ── check models exist ────────────────────────────────────────────────────────
models_ready = all(
    os.path.exists(os.path.join(MODEL_DIR, f))
    for f in ["crop_recommender.pkl", "yield_estimator.pkl",
              "label_encoder.pkl", "feature_scaler.pkl"]
)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾")
    st.title("Crop Advisory")
    st.caption(NOTE)
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Predict Crop", "📊 Dataset Explorer", "📈 Model Performance"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Project Info**")
    st.markdown("- 3,000 synthetic records")
    st.markdown("- 20 crop types")
    st.markdown("- Random Forest Classifier")
    st.markdown("- Gradient Boosting Regressor")
    st.caption("For educational use only.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🌾 AI/ML Crop Advisory System")
    st.info(f"**{NOTE}**", icon="ℹ️")

    st.markdown("""
    This system uses machine learning to recommend the best crop to grow
    based on soil nutrients and climate conditions, and estimates the expected yield.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", "3,000", "synthetic")
    col2.metric("Crop Types", "20", "classes")
    col3.metric("Classifier Accuracy", "64.3%", "RandomForest")
    col4.metric("Yield R² Score", "0.80", "GradientBoosting")

    st.divider()

    st.subheader("How to use this app")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🔍 Predict Crop")
        st.markdown("Enter your soil nutrients and climate values to get a crop recommendation with estimated yield.")
    with c2:
        st.markdown("### 📊 Dataset Explorer")
        st.markdown("Browse and filter the 3,000-record synthetic dataset. View distributions and statistics.")
    with c3:
        st.markdown("### 📈 Model Performance")
        st.markdown("See classifier accuracy, feature importances, and yield estimator metrics.")

    st.divider()
    st.subheader("Input Features Used")
    feat_df = pd.DataFrame({
        "Feature": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)",
                    "Temperature", "Humidity", "Soil pH", "Rainfall"],
        "Unit": ["kg/ha", "kg/ha", "kg/ha", "°C", "%", "—", "mm/month"],
        "Range": ["0–200", "0–120", "0–150", "0–50", "0–100", "3–10", "0–400"],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict Crop":
    st.title("🔍 Crop Prediction")
    st.info(f"**{NOTE}**", icon="ℹ️")

    if not models_ready:
        st.error("Models not found. Run `python train_model.py` first.")
        st.stop()

    clf, reg, le, scaler = load_models()

    st.markdown("### Enter Soil & Climate Parameters")
    st.markdown("Adjust the sliders below, then click **Predict**.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🧪 Soil Nutrients**")
        n_val  = st.slider("Nitrogen (N) — kg/ha",   0,   200, 90)
        p_val  = st.slider("Phosphorus (P) — kg/ha", 0,   120, 45)
        k_val  = st.slider("Potassium (K) — kg/ha",  0,   150, 40)
        ph_val = st.slider("Soil pH",                3.0, 10.0, 6.5, step=0.1)

    with col2:
        st.markdown("**🌤️ Climate Conditions**")
        t_val  = st.slider("Temperature — °C",       0,   50,  25)
        h_val  = st.slider("Humidity — %",           0,   100, 80)
        r_val  = st.slider("Rainfall — mm/month",    0,   400, 180)

    st.divider()

    if st.button("🌱 Predict Best Crop", type="primary", use_container_width=True):
        features = [n_val, p_val, k_val, float(t_val), float(h_val), float(ph_val), float(r_val)]
        top3, yield_est = predict(features, clf, reg, le, scaler)

        st.markdown("### 🏆 Results")
        st.caption("Synthetic demonstration only — not real agricultural advice.")

        c1, c2, c3 = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        cols = [c1, c2, c3]
        for i, (crop, prob) in enumerate(top3):
            emoji = CROP_EMOJI.get(crop, "🌿")
            with cols[i]:
                st.metric(
                    label=f"{medals[i]} Rank {i+1}",
                    value=f"{emoji} {crop}",
                    delta=f"{prob}% confidence",
                )

        st.divider()
        best_crop = top3[0][0]
        best_emoji = CROP_EMOJI.get(best_crop, "🌿")
        st.markdown(f"### 📦 Estimated Yield for {best_emoji} {best_crop}")

        ycol1, ycol2 = st.columns([1, 2])
        with ycol1:
            st.metric("Estimated Yield", f"{yield_est} t/ha", "tonnes per hectare")
        with ycol2:
            st.progress(min(top3[0][1] / 100, 1.0), text=f"Model confidence: {top3[0][1]}%")

        st.success(f"Top recommendation: **{best_crop}** with **{yield_est} tonnes/hectare** estimated yield.")

        st.markdown("#### All Top 3 Crops")
        result_df = pd.DataFrame({
            "Rank": ["1st", "2nd", "3rd"],
            "Crop": [f"{CROP_EMOJI.get(c,'🌿')} {c}" for c, _ in top3],
            "Confidence (%)": [p for _, p in top3],
        })
        st.dataframe(result_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.info(f"**{NOTE}**", icon="ℹ️")

    if not os.path.exists(DATA_CSV):
        st.error("Dataset not found. Run `python generate_dataset.py` first.")
        st.stop()

    df = load_dataset()

    st.markdown(f"**{len(df):,} synthetic records · 20 crop types · No personal information**")

    # filters
    col1, col2 = st.columns(2)
    with col1:
        selected_crops = st.multiselect(
            "Filter by crop type",
            options=sorted(df["crop_type"].unique()),
            default=[],
        )
    with col2:
        ph_range = st.slider("Filter by Soil pH", 3.0, 10.0, (3.0, 10.0), step=0.1)

    filtered = df.copy()
    if selected_crops:
        filtered = filtered[filtered["crop_type"].isin(selected_crops)]
    filtered = filtered[(filtered["soil_pH"] >= ph_range[0]) & (filtered["soil_pH"] <= ph_range[1])]

    st.markdown(f"**Showing {len(filtered):,} records**")
    st.dataframe(
        filtered[["record_id", "crop_type"] + FEATURE_COLS + ["estimated_yield"]].head(200),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Summary Statistics")
    st.dataframe(filtered[FEATURE_COLS + ["estimated_yield"]].describe().round(2), use_container_width=True)

    st.divider()
    st.subheader("Records per Crop Type")
    crop_counts = df["crop_type"].value_counts().reset_index()
    crop_counts.columns = ["Crop", "Count"]
    st.bar_chart(crop_counts.set_index("Crop"))

    st.divider()
    st.subheader("Average Yield by Crop")
    avg_yield = df.groupby("crop_type")["estimated_yield"].mean().sort_values(ascending=False).reset_index()
    avg_yield.columns = ["Crop", "Avg Yield (t/ha)"]
    avg_yield["Avg Yield (t/ha)"] = avg_yield["Avg Yield (t/ha)"].round(2)
    st.bar_chart(avg_yield.set_index("Crop"))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.info(f"**{NOTE}**", icon="ℹ️")

    if not models_ready:
        st.error("Models not found. Run `python train_model.py` first.")
        st.stop()

    clf, reg, le, scaler = load_models()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌳 Crop Recommender")
        st.markdown("**Model:** RandomForestClassifier (200 trees)")
        m1, m2 = st.columns(2)
        m1.metric("Test Accuracy", "64.3%")
        m2.metric("5-Fold CV Accuracy", "68.1%")

    with col2:
        st.markdown("### 📉 Yield Estimator")
        st.markdown("**Model:** GradientBoostingRegressor (200 estimators)")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", "6.26")
        m2.metric("RMSE", "11.36")
        m3.metric("R² Score", "0.80")

    st.divider()
    st.subheader("Feature Importances (Crop Recommender)")
    importances = clf.feature_importances_
    feat_labels = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "Soil pH", "Rainfall"]
    imp_df = pd.DataFrame({
        "Feature": feat_labels,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    imp_df["Importance"] = imp_df["Importance"].round(4)

    st.bar_chart(imp_df.set_index("Feature"))
    st.dataframe(imp_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Training Report")
    report_path = os.path.join("reports", "training_report.txt")
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            st.code(f.read(), language=None)
    else:
        st.warning("Training report not found. Run `python train_model.py` to generate it.")

    st.divider()
    st.subheader("Dataset & Privacy")
    st.markdown("""
    | Item | Detail |
    |---|---|
    | Dataset type | **Synthetic — artificially generated** |
    | Records | 3,000 |
    | Random seed | 42 (fully reproducible) |
    | Personal data | **None** — no names, phones, emails, addresses, GPS, or ID numbers |
    | Source | `generate_dataset.py` (this repository) |
    | Purpose | Educational AI/ML demonstration only |
    | Real farm data | **No** |
    | External download | **No** |
    """)
