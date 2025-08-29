import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="HealthTune — AI Health Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Utilities & Caching ----------------------
@st.cache_data(show_spinner=False)
def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    try:
        models["log_reg"] = joblib.load(os.path.join(MODELS_DIR, "log_reg.pkl"))
    except Exception:
        models["log_reg"] = None
    try:
        models["random_forest"] = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    except Exception:
        models["random_forest"] = None
    try:
        models["scaler"] = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    except Exception:
        models["scaler"] = None
    try:
        models["kmeans"] = joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl"))
    except Exception:
        models["kmeans"] = None
    try:
        models["cluster_scaler"] = joblib.load(os.path.join(MODELS_DIR, "cluster_scaler.pkl"))
    except Exception:
        models["cluster_scaler"] = None
    return models


def generate_recommendations(user_row):
    recs = []
    if user_row["steps"] < 6000:
        recs.append("Increase daily steps by ~2,000 — start with two 20-min walks per day.")
    elif user_row["steps"] < 10000:
        recs.append("Add a 20-min brisk walk or short active breaks to reach ~10,000 steps.")
    else:
        recs.append("Excellent activity levels — maintain with 1–2 rest days weekly.")

    if user_row["sleep_hours"] < 7.0 or user_row["sleep_efficiency"] < 0.85:
        recs.append("Aim to go to bed 30 minutes earlier and maintain a consistent sleep schedule.")
    else:
        recs.append("Sleep routine looks stable — keep pre-sleep wind-down rituals.")

    if user_row["diet_balance"] < 0.75:
        recs.append("Improve dietary balance: target ~50% carbs / 20% protein / 30% fats; add lean protein to meals.")
    else:
        recs.append("Dietary balance is good — continue meal-planning and hydration (~2 L/day).")

    if user_row["resting_hr"] > 72 or user_row["hrv"] < 35:
        recs.append("Introduce 2× weekly zone-2 cardio sessions and 10 min daily breathing for HR/HRV benefits.")

    if user_row["bmi"] >= 27:
        recs.append("Consider a small caloric deficit (300–500 kcal/day) + more protein and resistance training for weight management.")

    c = user_row.get("cluster_name", None)
    if c is not None:
        recs.append(f"Cluster insight: '{c}' — choose one small habit to focus on this week.")
    return recs


# ---------------------- Layout ----------------------
st.title("HealthTune — AI-Powered Health Insights")
st.markdown("A production-style dashboard that analyzes wearable & lifestyle data to generate personalized insights, risk predictions, and recommendations.")

# Sidebar
st.sidebar.header("Data & Models")
uploaded_daily = st.sidebar.file_uploader("Upload daily logs CSV (optional)", type=["csv"])
uploaded_features = st.sidebar.file_uploader("Upload features CSV (optional)", type=["csv"])
use_sample = st.sidebar.checkbox("Use bundled sample data if available", value=True)

models = load_models()

# Load data
if uploaded_daily is not None:
    try:
        daily = pd.read_csv(uploaded_daily)
    except Exception:
        st.sidebar.error("Failed to read uploaded daily CSV.")
        daily = None
elif use_sample:
    daily = load_csv_safe(r"C:\Users\samee\Desktop\New folder (3)\lifestyle_daily_clean.csv")
else:
    daily = None

if uploaded_features is not None:
    try:
        features = pd.read_csv(uploaded_features)
    except Exception:
        st.sidebar.error("Failed to read uploaded features CSV.")
        features = None
elif use_sample:
    features = load_csv_safe(r"C:\Users\samee\Desktop\New folder (3)\user_features.csv")
else:
    features = None

if daily is None or features is None:
    st.warning("Daily logs or feature dataset not found. Upload CSVs in the sidebar or enable 'Use bundled sample data'.")
    st.stop()

# Cleaning
if "date" not in daily.columns:
    st.error("Daily dataset is missing 'date' column.")
    st.stop()

daily['date'] = pd.to_datetime(daily['date'])
features = features.copy()

# ✅ Agar cluster_name missing hai to add kar do
if "cluster_name" not in features.columns:
    features["cluster_name"] = "Unknown"

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Users", int(features['user_id'].nunique()))
col2.metric("Days of Logs", f"{int(daily.shape[0])}")
col3.metric("Avg daily steps (median)", f"{int(daily['steps'].median())}")
col4.metric("Avg sleep (hrs)", f"{daily['sleep_hours'].mean():.2f}")

st.markdown("---")

# Tabs
tabs = st.tabs(["Overview", "Exploratory Analysis", "Modeling & Predictions", "User Explorer", "Download & Export"])

# ---------------------- Overview ----------------------
with tabs[0]:
    st.header("Project Overview")
    st.write("""
    This dashboard demonstrates a full pipeline for wearable & lifestyle data:
    - Data ingestion & cleaning
    - Feature engineering (Activity Score, Sleep Index, Dietary Balance)
    - EDA (distributions, correlations)
    - Predictive models for obesity risk and sleep irregularity detection
    - Lifestyle clustering and personalized recommendations
    """)
    st.subheader("Data snapshot — daily logs (top 5 rows)")
    st.dataframe(daily.head())

# ---------------------- EDA ----------------------
with tabs[1]:
    st.header("Exploratory Data Analysis (EDA)")
    st.subheader("Steps distribution")
    fig1, ax1 = plt.subplots(figsize=(8,3))
    daily['steps'].hist(bins=50, ax=ax1)
    ax1.set_xlabel("Steps"); ax1.set_ylabel("Count"); ax1.set_title("Daily Steps Distribution")
    st.pyplot(fig1)

    st.subheader("Lag-1 Sleep vs Resting HR (correlation)")
    temp = daily.sort_values(['user_id','date']).copy()
    temp['sleep_prev'] = temp.groupby('user_id')['sleep_hours'].shift(1)
    corr_df = temp[['sleep_prev','resting_hr']].dropna()
    if not corr_df.empty:
        corr_val = corr_df.corr().iloc[0,1]
    else:
        corr_val = np.nan
    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.scatter(corr_df['sleep_prev'], corr_df['resting_hr'], s=6, alpha=0.25)
    ax2.set_xlabel("Sleep hours (previous day)"); ax2.set_ylabel("Resting HR (bpm)")
    ax2.set_title(f"Lag-1 Sleep vs Resting HR (corr = {corr_val:.2f})")
    st.pyplot(fig2)

    st.subheader("Top correlations (selected features)")
    corr_mat = features[["steps","sleep_hours","resting_hr","hrv","diet_balance","mood"]].corr()
    st.dataframe(corr_mat.round(2))

# ---------------------- Modeling & Predictions ----------------------
with tabs[2]:
    st.header("Modeling & Predictions")
    st.write("Models loaded from local path. If a model is missing, the corresponding UI will be disabled.")

    rf = models.get("random_forest", None)
    log_reg = models.get("log_reg", None)
    scaler = models.get("scaler", None)

    st.subheader("Obesity risk model outputs (per user)")
    if rf is None and log_reg is None:
        st.info("No predictive models available. Please train or upload models.")
    else:
        model_choice = st.selectbox(
            "Select model for scoring",
            ["Random Forest (preferred)" if rf is not None else None,
             "Logistic Regression (interpretable)" if log_reg is not None else None]
        )
        model_choice = model_choice or ("Random Forest (preferred)" if rf is not None else "Logistic Regression (interpretable)")
        st.write(f"Using: **{model_choice}**")

        feature_cols = [
            "age","height_cm","weight_kg","bmi","steps","active_minutes","sedentary_minutes",
            "sleep_hours","sleep_efficiency","resting_hr","hrv","calories_in","diet_balance",
            "activity_score","sleep_index","mood","steps_per_active_min","cal_per_kg"
        ]
        if not set(feature_cols).issubset(features.columns):
            st.error("Feature columns missing in features CSV.")
        else:
            X_batch = features[feature_cols].copy()
            if "Logistic" in model_choice and log_reg is not None and scaler is not None:
                X_scaled = scaler.transform(X_batch)
                probs = log_reg.predict_proba(X_scaled)[:,1]
            else:
                if rf is not None:
                    probs = rf.predict_proba(X_batch)[:,1]
                else:
                    probs = np.zeros(len(X_batch))

            features['pred_obesity_risk'] = probs

            # ✅ Safe cluster handling
            if "cluster_name" not in features.columns:
                features["cluster_name"] = "Unknown"

            top_risks = features.sort_values("pred_obesity_risk", ascending=False).head(10)
            st.subheader("Top 10 Users by Predicted Obesity Risk")
            st.table(
                top_risks[["user_id","bmi","pred_obesity_risk","cluster_name"]].assign(
                    pred_obesity_risk=lambda df: (df['pred_obesity_risk']*100).round(1).astype(str) + "%"
                )
            )

# ---------------------- User Explorer ----------------------
with tabs[3]:
    st.header("User Explorer — personalized insights & recommendations")
    user_id = st.selectbox("Choose user_id", features['user_id'].unique())
    user_row = features[features['user_id'] == user_id].iloc[0]

    st.subheader("User summary")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("BMI", f"{user_row['bmi']:.1f}")
    kpi2.metric("Avg steps", f"{int(user_row['steps'])}")
    kpi3.metric("Avg sleep (hrs)", f"{user_row['sleep_hours']:.2f}")
    kpi4.metric("Avg resting HR", f"{user_row['resting_hr']:.1f} bpm")

    st.markdown("#### Time-series snapshot (last 60 days)")
    uid = int(user_id)
    user_daily = daily[daily['user_id'] == uid].sort_values('date').tail(60)
    if not user_daily.empty:
        fig3, ax3 = plt.subplots(figsize=(10,3))
        ax3.plot(user_daily['date'], user_daily['steps'])
        ax3.set_title("Steps (last 60 days)"); ax3.set_xlabel("Date"); ax3.set_ylabel("Steps")
        plt.xticks(rotation=45)
        st.pyplot(fig3)

        fig4, ax4 = plt.subplots(figsize=(10,2))
        ax4.plot(user_daily['date'], user_daily['sleep_hours'])
        ax4.set_title("Sleep hours (last 60 days)"); ax4.set_xlabel("Date"); ax4.set_ylabel("Hours")
        plt.xticks(rotation=45)
        st.pyplot(fig4)
    else:
        st.info("No daily logs for this user found in the dataset.")

    st.markdown("#### Cluster & risk")
    st.write("Lifestyle cluster:", user_row.get("cluster_name", "Unknown"))

    rf_model = models.get("random_forest", None)
    scaler_model = models.get("scaler", None)
    log_model = models.get("log_reg", None)

    feature_cols = [
        "age","height_cm","weight_kg","bmi","steps","active_minutes","sedentary_minutes",
        "sleep_hours","sleep_efficiency","resting_hr","hrv","calories_in","diet_balance",
        "activity_score","sleep_index","mood","steps_per_active_min","cal_per_kg"
    ]
    if set(feature_cols).issubset(features.columns):
        x_user = user_row[feature_cols].values.reshape(1,-1)
        if rf_model is not None:
            rf_prob = rf_model.predict_proba(pd.DataFrame(x_user, columns=feature_cols))[0,1]
        else:
            rf_prob = None
        if log_model is not None and scaler_model is not None:
            try:
                x_scaled = scaler_model.transform(x_user)
                lr_prob = log_model.predict_proba(x_scaled)[0,1]
            except Exception:
                lr_prob = None
        else:
            lr_prob = None

        if rf_prob is not None:
            st.metric("RF Predicted obesity risk", f"{rf_prob*100:.1f}%")
        if lr_prob is not None:
            st.metric("LR Predicted obesity risk", f"{lr_prob*100:.1f}%")

    st.markdown("#### Personalized recommendations")
    recs = generate_recommendations(user_row)
    for r in recs:
        st.write("- " + r)

# ---------------------- Download & Export ----------------------
with tabs[4]:
    st.header("Download & Export")
    st.write("You can download the processed user features or top-risk table for reporting.")
    st.download_button("Download user features CSV", data=features.to_csv(index=False), file_name="user_features_with_clusters.csv", mime="text/csv")
    if 'pred_obesity_risk' in features.columns:
        top10 = features.sort_values("pred_obesity_risk", ascending=False).head(10)
        st.download_button("Download top 10 risk users", data=top10.to_csv(index=False), file_name="top10_predicted_risk.csv", mime="text/csv")

st.markdown("---")
st.caption(f"Generated: {datetime.utcnow().isoformat()}Z — Built for HealthTune demo.")
