"""
=============================================================
  Smart Crop Recommendation System — Streamlit Web App
=============================================================
Run:  streamlit run app.py
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Crop Recommender",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  /* ── Main area ── */
  .block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1200px; }

  /* ── Header ── */
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #1a472a;
    line-height: 1.15;
    margin-bottom: 0.3rem;
  }
  .hero-sub {
    font-size: 1.05rem;
    color: #4a7c59;
    margin-bottom: 2rem;
  }

  /* ── Cards ── */
  .metric-card {
    background: linear-gradient(135deg, #f0faf3 0%, #e8f5ec 100%);
    border: 1px solid #c8e6d0;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
  }
  .metric-card h3 { margin: 0; font-size: 2rem; color: #1a472a; }
  .metric-card p  { margin: 0; font-size: 0.85rem; color: #4a7c59; }

  /* ── Result banner ── */
  .result-banner {
    background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    color: white;
    margin: 1.5rem 0;
  }
  .result-banner h1 { font-family: 'Playfair Display', serif; font-size: 2.6rem; margin: 0.3rem 0; }
  .result-banner p  { font-size: 1rem; opacity: 0.85; margin: 0; }

  /* ── Confidence bar ── */
  .conf-label { font-size: 0.85rem; color: #555; margin-bottom: 4px; }
  .conf-outer {
    background: #e0e0e0; border-radius: 8px; height: 14px; overflow: hidden;
  }
  .conf-inner {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg, #40916c, #52b788);
    transition: width 0.6s ease;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] { background: #f8fbf9; }
  .sidebar-header {
    background: linear-gradient(135deg, #1a472a, #2d6a4f);
    border-radius: 12px;
    padding: 1.2rem;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
  }
  .sidebar-header h3 { margin: 0; font-family: 'Playfair Display', serif; font-size: 1.25rem; }

  /* ── Divider ── */
  hr { border: none; border-top: 2px solid #d4edda; margin: 1.5rem 0; }

  /* ── Button ── */
  .stButton > button {
    background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 100%);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] { gap: 12px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.2rem;
    background: #f0faf3;
    color: #1a472a;
  }
</style>
""", unsafe_allow_html=True)


# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model   = joblib.load("model/best_model.pkl")
    scaler  = joblib.load("model/scaler.pkl")
    le      = joblib.load("model/label_encoder.pkl")
    return model, scaler, le

model, scaler, le = load_artifacts()

# ── Crop emoji map ────────────────────────────────────────────────────────────
CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🟤", "mothbeans": "🟡", "mungbean": "💚", "blackgram": "⚫",
    "lentil": "🟠", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🧡", "coconut": "🥥", "cotton": "🌿",
    "jute": "🌿", "coffee": "☕",
}

# ── Ideal ranges reference ────────────────────────────────────────────────────
IDEAL = {
    "rice":        dict(N="60–120", P="30–60",  K="30–60",  temp="20–27°C", humidity="75–90%", ph="5.5–7.0", rainfall="150–250 mm"),
    "maize":       dict(N="60–100", P="30–60",  K="10–30",  temp="18–27°C", humidity="55–75%", ph="5.5–7.0", rainfall="50–80 mm"),
    "chickpea":    dict(N="30–50",  P="50–80",  K="60–100", temp="15–22°C", humidity="10–25%", ph="6.5–8.0", rainfall="65–95 mm"),
    "coffee":      dict(N="80–120", P="20–35",  K="20–40",  temp="20–30°C", humidity="50–65%", ph="6.0–7.0", rainfall="130–185 mm"),
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
      <h3>🌾 Input Parameters</h3>
      <p style="margin:0;opacity:0.8;font-size:0.82rem;">Enter soil & climate data</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🧪 Soil Nutrients (kg/ha)")
    N = st.slider("Nitrogen (N)",    0,   200, 70,  help="Nitrogen content in soil (kg/ha)")
    P = st.slider("Phosphorus (P)",  0,   145, 45,  help="Phosphorus content in soil (kg/ha)")
    K = st.slider("Potassium (K)",   0,   210, 40,  help="Potassium content in soil (kg/ha)")

    st.markdown("#### 🌡️ Climate Conditions")
    temperature = st.slider("Temperature (°C)",  8.0,  44.0, 25.0, 0.1)
    humidity    = st.slider("Humidity (%)",      14.0, 100.0, 70.0, 0.1)
    rainfall    = st.slider("Rainfall (mm)",     20.0, 300.0, 100.0, 1.0)

    st.markdown("#### 🧫 Soil Chemistry")
    ph = st.slider("pH Value", 3.5, 9.5, 6.5, 0.1,
                   help="Soil pH: 7 is neutral, <7 acidic, >7 alkaline")

    st.markdown("---")
    predict_btn = st.button("🌿 Recommend Crop", use_container_width=True)

    st.markdown("""
    <div style="margin-top:1.5rem;padding:1rem;background:#e8f5ec;border-radius:10px;font-size:0.8rem;color:#2d6a4f;">
    <b>💡 How to use:</b><br>
    1. Adjust the sliders to match your field's soil and climate conditions.<br>
    2. Click <b>Recommend Crop</b> to get an AI-powered suggestion.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">Smart Crop Recommendation System 🌾</div>
<div class="hero-sub">
  AI-powered crop selection using soil nutrients and climate data &nbsp;|&nbsp;
  Random Forest · 98.6% Accuracy
</div>
""", unsafe_allow_html=True)

tab_predict, tab_eda, tab_about = st.tabs(["🌿 Prediction", "📊 EDA & Charts", "ℹ️ About"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    # Current input summary cards
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    metrics = [
        ("N", N, "kg/ha", col1), ("P", P, "kg/ha", col2), ("K", K, "kg/ha", col3),
        ("Temp", temperature, "°C", col4), ("Humidity", humidity, "%", col5),
        ("pH", ph, "", col6), ("Rainfall", rainfall, "mm", col7),
    ]
    for label, val, unit, col in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <h3>{val:.1f}</h3>
              <p>{label}{' ' + unit if unit else ''}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    if predict_btn:
        # ── Inference ──
        input_arr = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_df  = pd.DataFrame(input_arr,
                                  columns=["N","P","K","temperature","humidity","ph","rainfall"])

        # Random Forest doesn't need scaling; model was trained on raw features
        probabilities = model.predict_proba(input_df)[0]
        pred_idx      = np.argmax(probabilities)
        pred_crop     = le.inverse_transform([pred_idx])[0]
        confidence    = probabilities[pred_idx] * 100
        emoji         = CROP_EMOJI.get(pred_crop, "🌱")

        # ── Result banner ──
        st.markdown(f"""
        <div class="result-banner">
          <p>✅ Recommended Crop</p>
          <h1>{emoji} {pred_crop.capitalize()}</h1>
          <p>Based on your soil and climate inputs</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bar ──
        bar_color = "#40916c" if confidence >= 70 else "#f4a261"
        st.markdown(f"""
        <div class="conf-label">Model Confidence: <b>{confidence:.1f}%</b></div>
        <div class="conf-outer">
          <div class="conf-inner" style="width:{confidence:.1f}%;background:{'linear-gradient(90deg,#40916c,#52b788)' if confidence>=70 else 'linear-gradient(90deg,#f4a261,#e9c46a)'};"></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Top-5 alternatives ──
        st.markdown("<br>", unsafe_allow_html=True)
        top5_idx   = np.argsort(probabilities)[::-1][:5]
        top5_crops = le.inverse_transform(top5_idx)
        top5_probs = probabilities[top5_idx] * 100

        st.markdown("#### 🔢 Top 5 Crop Probabilities")
        alt_cols = st.columns(5)
        for i, (crop, prob) in enumerate(zip(top5_crops, top5_probs)):
            em = CROP_EMOJI.get(crop, "🌱")
            rank_color = "#1a472a" if i == 0 else "#4a7c59"
            with alt_cols[i]:
                st.markdown(f"""
                <div style="text-align:center;padding:1rem;background:#f8fbf9;
                            border-radius:12px;border:1px solid #c8e6d0;">
                  <div style="font-size:1.6rem">{em}</div>
                  <div style="font-weight:600;color:{rank_color};font-size:0.9rem">
                    {crop.capitalize()}
                  </div>
                  <div style="color:#666;font-size:0.8rem">{prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Input summary table ──
        st.markdown("<br>#### 📋 Your Input Summary", unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Parameter":  ["N (Nitrogen)", "P (Phosphorus)", "K (Potassium)",
                           "Temperature", "Humidity", "pH", "Rainfall"],
            "Value":      [N, P, K, temperature, humidity, ph, rainfall],
            "Unit":       ["kg/ha", "kg/ha", "kg/ha", "°C", "%", "–", "mm"],
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

    else:
        st.info("👈  Adjust the sliders in the sidebar and click **Recommend Crop** to begin.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA & CHARTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_eda:
    chart_files = {
        "📈 Model Comparison":      "notebook/model_comparison.png",
        "🔥 Correlation Heatmap":   "notebook/correlation_heatmap.png",
        "📦 Boxplots by Crop":      "notebook/boxplots_by_crop.png",
        "🎯 Feature Distributions": "notebook/feature_distributions.png",
        "🌲 Feature Importance":    "notebook/feature_importance.png",
        "🔢 Confusion Matrix":      "notebook/confusion_matrix.png",
    }

    row1 = list(chart_files.items())[:2]
    row2 = list(chart_files.items())[2:4]
    row3 = list(chart_files.items())[4:]

    for row in [row1, row2, row3]:
        cols = st.columns(len(row))
        for col, (title, path) in zip(cols, row):
            with col:
                st.markdown(f"**{title}**")
                if os.path.exists(path):
                    st.image(path, use_column_width=True)
                else:
                    st.warning(f"Run `python src/train.py` first to generate this chart.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        ### 🌾 About This Project

        The **Smart Crop Recommendation System** uses Machine Learning to help
        farmers and agronomists identify the most suitable crop for their field
        based on:

        - **Soil nutrients** — N, P, K levels
        - **Climate conditions** — temperature, humidity, rainfall
        - **Soil chemistry** — pH value

        ---

        ### 🤖 ML Models Trained

        | Model | Test Accuracy |
        |---|---|
        | Random Forest ✅ | **98.64%** |
        | Logistic Regression | 97.73% |
        | Decision Tree | 94.32% |

        The best model (Random Forest, 200 estimators) is saved as a `.pkl`
        file and loaded at app startup.

        ---

        ### 📊 Dataset

        - **2 200 samples** across **22 crop classes** (100 per class)
        - Balanced, stratified 80/20 train–test split
        - No missing values; Standard Scaler applied for LR
        """)

    

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center;color:#4a7c59;font-size:0.82rem;padding:0.5rem 0 1rem;">
  🌾 Smart Crop Recommendation System &nbsp;|&nbsp;
  Built with Python · scikit-learn · Streamlit &nbsp;|&nbsp;
  Random Forest · 98.6% Accuracy
</div>
""", unsafe_allow_html=True)
