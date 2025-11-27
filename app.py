# app.py — Airline Tweet Sentiment Dashboard (KODE FINAL MODERN SLIDER)

import os
import json
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk

# ===================== NLTK SETUP =====================
try:
    nltk.data.find('corpora/wordnet')
except Exception:
    nltk.download('wordnet')
    nltk.download('punkt')
# ======================================================

# Modul util & viz
from src.ingest import read_any_csv, coerce_columns_lower
from src.clean import clean_pipeline
from src.features import kpi_metrics, agg_sentiment, agg_hour_trend, agg_topic_vs_sentiment
from src.viz import fig_sentiment_bar, fig_sentiment_pie, fig_hour_trend, fig_topic_stack

# =================== PAGE SETUP ===================
st.set_page_config(page_title="Airline Tweet Sentiment", layout="wide")
st.title("✈️ Airline Tweet Sentiment Dashboard")
st.caption("Fokus: manajemen data (clean/transform) + visual interaktif + insight.")

# ===================== THEME SWITCHER =====================
THEMES = {
    "Terang": {
        "plotly_template": "plotly_white",
        "bg": "#ffffff", "text": "#0f172a",
        "card": "#f0f2f6", "muted": "#475569",
        "wc_bg": "white", "mpl_face": "white",
        "button": "#1e40af", "input_bg": "#ffffff",
        "border": "#d1d5db"
    },
    "Gelap": {
        "plotly_template": "plotly_dark",
        "bg": "#0b1220", "text": "#e5e7eb",
        "card": "#111827", "muted": "#9ca3af",
        "wc_bg": "black", "mpl_face": "#0b1220",
        "button": "#FF5722", "input_bg": "#1f2937",
        "border": "#374151"
    },
}

if "theme_name" not in st.session_state:
    st.session_state.theme_name = "Gelap"

with st.sidebar:
    st.header("Tampilan")
    cho = st.selectbox("Tema UI", THEMES.keys())
    st.session_state.theme_name = cho

THEME = THEMES[cho]

# Terapkan template Plotly
import plotly.io as pio
pio.templates.default = THEME["plotly_template"]

# ===================== CSS UI CUSTOM =====================
st.markdown(
    f"""
<style>

:root {{
    --bg: {THEME['bg']};
    --text: {THEME['text']};
    --card: {THEME['card']};
    --input-bg: {THEME['input_bg']};
    --border: {THEME['border']};
}}

html, body, [data-testid="stAppViewContainer"] {{
    background-color: var(--bg);
    color: var(--text);
}}

[data-testid="stSidebar"] {{
    background-color: var(--card);
    color: var(--text);
}}

[data-testid="stMetricValue"] {{
    color: var(--text) !important;
    white-space: normal !important;
    word-break: break-word !important;
    font-size: 1.3rem;
}}

[data-testid="stMetricLabel"] {{
    color: var(--muted) !important;
    white-space: normal !important;
    word-break: break-word !important;
}}

.stButton button {{
    background-color: {THEME['button']};
    color: white;
    border-radius: 6px;
}}

/* ---------------------------------------------
   ★ MODERN SLIDER STYLE (Hour of Day)
----------------------------------------------*/

[data-baseweb="slider"] {{
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}}

[data-baseweb="slider"] > div {{
    background-color: transparent !important;
    border-radius: 8px !important;
}}

[data-baseweb="slider"] div[data-baseweb="track"] {{
    background-color: #4b5563 !important;
}}

[data-baseweb="slider"] div[data-baseweb="track"] > div {{
    background-color: #ef4444 !important; /* warna track aktif */
}}

[data-baseweb="slider"] div[data-baseweb="thumb"] {{
    background-color: #ff6666 !important;
    width: 20px !important;
    height: 20px !important;
    border: 3px solid white !important;
    box-shadow: 0 0 4px rgba(255, 0, 0, .6);
}}

[data-baseweb="slider"] span {{
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text);
}}

</style>
""",
    unsafe_allow_html=True
)

# ===================== DATA LOADING =====================
@st.cache_data
def load_default(file="Tweets.csv"):
    return coerce_columns_lower(read_any_csv(file))

@st.cache_data
def load_upload(f):
    return coerce_columns_lower(read_any_csv(f))

@st.cache_data
def clean(df):
    return clean_pipeline(df)

# ---------------- Load Data ----------------
try:
    data_raw = load_default("Tweets.csv")
except:
    data_raw = None

with st.sidebar:
    st.header("Data Source")
    upl = st.file_uploader("Upload CSV", type=["csv"])
    if upl:
        data_raw = load_upload(upl)

if data_raw is None:
    st.error("Gagal memuat data. Pastikan Tweets.csv ada atau upload file.")
    st.stop()

data = clean(data_raw)

# ===================== DATE + HOUR UTILS =====================
def to_jkt(series):
    ts = pd.to_datetime(series, errors="coerce")
    try:
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
    except:
        try: ts = ts.dt.tz_localize(None)
        except: pass
    return ts

def fix_range(val, mn, mx):
    if isinstance(val, (list, tuple)):
        if len(val) == 2:
            lo, hi = val
        else:
            lo, hi = mn, mx
    else:
        lo, hi = mn, mx
    lo = max(mn, min(lo, mx))
    hi = max(mn, min(hi, mx))
    return (lo, hi)

# ===================== SIDEBAR FILTER =====================
with st.sidebar:
    st.header("Filters")

    # --- DATE RANGE ---
    if "tweet_created" in data:
        ts = to_jkt(data["tweet_created"])
        dmin = ts.min().date()
        dmax = ts.max().date()
    else:
        dmin = date.today() - timedelta(days=7)
        dmax = date.today()

    if "date_range" not in st.session_state:
        st.session_state.date_range = (dmin, dmax)

    dr = st.date_input("Date range", value=st.session_state.date_range,
                       min_value=dmin, max_value=dmax)

    st.session_state.date_range = fix_range(dr, dmin, dmax)
    d0, d1 = st.session_state.date_range

    # --- AIRLINE, SENTIMENT ---
    airlines = sorted(data["airline"].dropna().unique()) if "airline" in data else []
    sentiments = sorted(data["airline_sentiment"].dropna().unique()) if "airline_sentiment" in data else []

    if "flt_airline" not in st.session_state:
        st.session_state.flt_airline = airlines
    if "flt_sent" not in st.session_state:
        st.session_state.flt_sent = sentiments

    fa = st.multiselect("Pick airlines", airlines, st.session_state.flt_airline)
    fs = st.multiselect("Pick sentiments", sentiments, st.session_state.flt_sent)

    st.session_state.flt_airline = fa
    st.session_state.flt_sent = fs

    # --- MODERN HOUR SLIDER ---
    if "hour" in data:
        if "flt_hour" not in st.session_state:
            st.session_state.flt_hour = (0, 23)

        st.markdown("**Hour of day**")
        hr = st.slider(
            "", 0, 23,
            st.session_state.flt_hour,
            format="%02d:00",
            label_visibility="collapsed",
        )
        st.caption(f"Rentang jam aktif: {hr[0]:02d}:00 – {hr[1]:02d}:59")
        st.session_state.flt_hour = hr
    else:
        hr = (0, 23)

    # --- KEYWORD ---
    kw = st.text_input("Cari kata (opsional, di text_clean)",
                       st.session_state.get("kw", ""))
    st.session_state.kw = kw

    # --- RESET ---
    if st.button("Reset all filters"):
        st.session_state.clear()
        st.rerun()

# ===================== APPLY FILTER =====================
df = data.copy()

# Filter date
df["_dt"] = to_jkt(df["tweet_created"])
df = df[(df["_dt"] >= pd.Timestamp(d0)) & (df["_dt"] < pd.Timestamp(d1) + pd.Timedelta(days=1))]
df = df.drop(columns=["_dt"])

# Airline / Sentiment
if fa:
    df = df[df["airline"].isin(fa)]
if fs:
    df = df[df["airline_sentiment"].isin(fs)]

# Hour
if "hour" in df:
    df = df[df["hour"].notna()]
    df = df[(df["hour"] >= hr[0]) & (df["hour"] <= hr[1])]

# Keyword
if kw:
    col = "text_clean" if "text_clean" in df else "text"
    df = df[df[col].str.contains(kw, case=False, na=False)]

# ===================== ISSUES =====================
ISSUES = {
    "delay": ["delay", "late", "terlambat"],
    "refund": ["refund", "pengembalian"],
    "bagasi": ["bagasi", "koper", "luggage"],
    "service": ["service", "pelayanan", "crew", "pramugari"]
}

def find_issue(t):
    t = (t or "").lower()
    for cat, keys in ISSUES.items():
        if any(k in t for k in keys):
            return cat
    return "(other)"

if "issue" not in df:
    base = "text_clean" if "text_clean" in df else "text"
    df["issue"] = df[base].apply(find_issue)

# ===================== KPI =====================
kpi = kpi_metrics(df)
neg = df[df["airline_sentiment"] == "negative"]
delay_share = (len(neg[neg["issue"] == "delay"]) / len(neg) * 100) if len(neg) else 0

c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.2])
c1.metric("Total Tweets", f"{kpi['total_tweets']:,}")
c2.metric("% Negative", f"{kpi['neg_pct']:.1f}%")
c3.metric("Top Airline (Neg)", kpi["top_neg_airline"])
c4.metric("Delay in Negative", f"{delay_share:.1f}%")

# ===================== DOWNLOAD =====================
st.markdown("### ⬇️ Download")
d1c, d2c = st.columns(2)
d1c.download_button("Download CLEANED CSV", data.to_csv(index=False), "cleaned.csv")
d2c.download_button("Download FILTERED CSV", df.to_csv(index=False), "filtered.csv")

# ===================== TABS =====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["👋 Welcome", "Overview", "Time Trend", "Map", "Topics", "Data Quality"]
)

with tab2:
    st.subheader("Sentiment Overview")
    sc = agg_sentiment(df)
    if len(sc):
        st.plotly_chart(fig_sentiment_bar(sc), use_container_width=True)
        st.plotly_chart(fig_sentiment_pie(sc), use_container_width=True)

with tab3:
    st.subheader("Tweets per Hour")
    tr = agg_hour_trend(df)
    if len(tr):
        st.plotly_chart(fig_hour_trend(tr), use_container_width=True)
        st.dataframe(tr)

with tab4:
    st.subheader("Tweet Locations")
    if "latitude" in df and "longitude" in df:
        mdf = df[["latitude", "longitude"]].dropna()
        if len(mdf):
            st.map(mdf)

with tab5:
    st.subheader("Word Cloud")
    if "text_clean" in df:
        blob = " ".join(df["text_clean"])
        if blob:
            wc = WordCloud(width=900, height=400, background_color=THEME["wc_bg"]).generate(blob)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

    st.subheader("Topic Breakdown")
    tdf = agg_topic_vs_sentiment(df)
    st.plotly_chart(fig_topic_stack(tdf), use_container_width=True)

with tab6:
    st.subheader("Data Quality")
    st.write({
        "rows": len(data),
        "cols": data.shape[1],
        "missing_%": round(data.isna().mean().mean()*100,2)
    })
    st.dataframe(data.head(20))
