# app.py — Airline Tweet Sentiment Dashboard (FINAL KECILKAN TOMBOL DOWNLOAD)

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

# Modul util & viz milikmu
from src.ingest import read_any_csv, coerce_columns_lower
from src.clean import clean_pipeline
from src.features import (
    kpi_metrics, agg_sentiment, agg_hour_trend, agg_topic_vs_sentiment
)
from src.viz import (
    fig_sentiment_bar, fig_sentiment_pie, fig_hour_trend, fig_topic_stack
)

# =================== PAGE SETUP ===================
st.set_page_config(page_title="Airline Tweet Sentiment", layout="wide")
st.title("✈️ Airline Tweet Sentiment Dashboard")
st.caption("Fokus: manajemen data (clean/transform) + visual interaktif + insight.")

# ===================== THEME SWITCHER =====================
THEMES = {
    "Terang": {
        "plotly_template": "plotly_white",
        "bg": "#ffffff",
        "text": "#0f172a",
        "card": "#f0f2f6",
        "muted": "#475569",
        "wc_bg": "white",
        "mpl_face": "white",
        "button": "#1e40af",
        "input_bg": "#ffffff",
        "border": "#d1d5db",
    },
    "Gelap": {
        "plotly_template": "plotly_dark",
        "bg": "#0b1220",
        "text": "#e5e7eb",
        "card": "#111827",
        "muted": "#9ca3af",
        "wc_bg": "black",
        "mpl_face": "#0b1220",
        "button": "#FF5722",
        "input_bg": "#1f2937",
        "border": "#374151",
    },
}

if "theme_name" not in st.session_state:
    st.session_state.theme_name = "Gelap"

with st.sidebar:
    st.header("Tampilan")
    theme_name = st.selectbox(
        "Tema UI",
        options=list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme_name),
        help="Pilih: Terang / Gelap",
    )
    st.session_state.theme_name = theme_name

THEME = THEMES[st.session_state.theme_name]

# Terapkan template Plotly
import plotly.io as pio
try:
    pio.templates.default = THEME["plotly_template"]
except Exception as e:
    print(f"Error applying template: {e}")

# ===================== CSS =====================
st.markdown(
    f"""
    <style>
    :root {{
      --bg: {THEME['bg']};
      --text: {THEME['text']};
      --card: {THEME['card']};
      --button-bg: {THEME['button']};
      --button-hover: #333;
      --input-bg: {THEME['input_bg']};
      --muted: {THEME['muted']};
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
    [data-testid="stSidebar"] > div:first-child {{
      background-color: var(--card);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    [data-testid="stSidebar"] .stMarkdown > p,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label p,
    [data-testid="stSidebar"] [data-testid="stTextInput"] label p,
    [data-testid="stSidebar"] [data-testid="stDateInput"] label p,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label p,
    [data-testid="stSidebar"] [data-testid="stMultiselect"] label p,
    [data-testid="stSidebar"] [data-testid="stSlider"] label p {{
        color: var(--text);
        font-weight: 600;
    }}

    /* KPI value (angka / nama maskapai) */
    [data-testid="stMetricValue"] {{
        color: var(--text) !important;
        font-weight: 700;
        font-size: 1.3rem;
        white-space: normal !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
        text-overflow: clip !important;
    }}

    /* KPI label (judul kecil) */
    [data-testid="stMetricLabel"] {{
        color: var(--muted) !important;
        font-weight: 500;
        white-space: normal !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
        text-overflow: clip !important;
    }}

    .stMetric, .stMarkdown, .stCaption, .stDataFrame, .stPlotlyChart {{
      color: var(--text);
    }}

    .stSelectbox > div, .stMultiselect > div, .stSlider > div {{
        background-color: var(--input-bg);
        border: 1px solid var(--border);
        color: var(--text);
    }}
    .stTextInput > div > div > input {{
        background-color: var(--input-bg);
        color: var(--text);
        border: 1px solid var(--border);
    }}
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] small {{
        color: var(--text);
    }}

    [data-testid="stTabs"] button {{
        color: var(--muted) !important;
    }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
        color: var(--text) !important;
    }}

    .stButton button {{
      background-color: var(--button-bg);
      color: white;
      border-radius: 6px;
      padding: 10px;
      font-size: 14px;
    }}
    .stButton button:hover {{
      background-color: var(--button-hover);
    }}

    /* ====== PERKECIL & RAPIKAN TOMBOL DOWNLOAD ====== */
    [data-testid="stDownloadButton"] > button {{
        padding: 4px 10px !important;
        font-size: 0.72rem !important;
        line-height: 1.1 !important;
        border-radius: 999px !important;      /* pill style */
        border: 1px solid var(--border) !important;
        background-color: var(--card) !important;
        color: var(--text) !important;
        min-height: 26px !important;
        height: 26px !important;
        white-space: nowrap !important;       /* teks satu baris */
    }}
    [data-testid="stDownloadButton"] > button:hover {{
        background-color: #3b4252 !important;
        border-color: #6b7280 !important;
    }}

    /* ====== SLIDER HOUR TANPA BOX PUTIH ====== */
    [data-testid="stSlider"] > div {{
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    [data-baseweb="slider"] span {{
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text);
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

# ===================== DATA LOADING (CACHED) =====================
@st.cache_data(show_spinner=False)
def _load_default_csv(_default_path: str) -> pd.DataFrame:
    df = read_any_csv(_default_path)
    return coerce_columns_lower(df)

@st.cache_data(show_spinner=False)
def _load_uploaded_csv(_file) -> pd.DataFrame:
    df = read_any_csv(_file)
    return coerce_columns_lower(df)

@st.cache_data(show_spinner=False)
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return clean_pipeline(df)

# Default file (Tweets.csv di root project)
DEFAULT_FILE = os.path.join(
    os.path.dirname(__file__) if "__file__" in globals() else ".",
    "Tweets.csv",
)

data_raw = None
try:
    data_raw = _load_default_csv("Tweets.csv")
except Exception as e:
    st.warning(f"Catatan: {e}")

# ===================== SIDEBAR: DATA SOURCE =====================
with st.sidebar:
    st.header("Data Source")
    up = st.file_uploader(
        "Upload CSV (opsional)",
        type=["csv"],
        help="Jika diisi, menimpa Tweets.csv bawaan."
    )
    if up is not None:
        data_raw = _load_uploaded_csv(up)

if data_raw is None or len(data_raw) == 0:
    st.error("Tidak ada data yang bisa dimuat. Pastikan `Tweets.csv` ada di root, atau upload CSV.")
    st.stop()

# Pipeline clean/transform (cached)
data = _clean(data_raw)

# ===================== HELPER FUNCTIONS =====================
def to_jkt_naive(series: pd.Series) -> pd.Series:
    """Konversi ke datetime Asia/Jakarta lalu buang tz."""
    s = pd.to_datetime(series, errors="coerce")
    try:
        if s.dt.tz is not None:
            s = s.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
    except Exception:
        try:
            s = s.dt.tz_localize(None)
        except Exception:
            pass
    return s

def normalize_date_range(raw, dt_min: date, dt_max: date) -> tuple[date, date]:
    """Pastikan date_range valid & dalam [dt_min, dt_max]."""
    if isinstance(dt_min, datetime):
        dt_min = dt_min.date()
    if isinstance(dt_max, datetime):
        dt_max = dt_max.date()

    if raw is None:
        return (dt_min, dt_max)

    if isinstance(raw, (date, datetime)):
        start = raw
        end = raw
    elif isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            return (dt_min, dt_max)
        elif len(raw) == 1:
            start = raw[0]
            end = raw[0]
        else:
            start, end = raw[0], raw[1]
    else:
        return (dt_min, dt_max)

    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()

    if start is None:
        start = dt_min
    if end is None:
        end = dt_max

    if start < dt_min: start = dt_min
    if start > dt_max: start = dt_max
    if end < dt_min:   end = dt_min
    if end > dt_max:   end = dt_max

    if start > end:
        start, end = dt_min, dt_max

    return (start, end)

def normalize_multiselect_default(
    options: list,
    current,
    fallback_mode: str = "all",
    some_k: int = 3,
):
    """Pastikan default multiselect hanya opsi valid + fallback kalau kosong."""
    opts = list(options or [])
    if not opts:
        return []
    if current is None:
        cur = []
    elif isinstance(current, (str, int)):
        cur = [current]
    else:
        cur = list(current)
    cur = [c for c in cur if c in opts]
    if not cur:
        cur = opts if fallback_mode == "all" else opts[: min(some_k, len(opts))]
    return cur

# ===================== FILTERS =====================
with st.sidebar:
    st.header("Filters")

    # ---- Date range ----
    if "tweet_created" in data.columns:
        dt_jkt = to_jkt_naive(data["tweet_created"])
        dt_min_ts = dt_jkt.min()
        dt_max_ts = dt_jkt.max()
        dt_min = (dt_min_ts.date() if pd.notna(dt_min_ts) else date.today() - timedelta(days=30))
        dt_max = (dt_max_ts.date() if pd.notna(dt_max_ts) else date.today())
    else:
        dt_min = date.today() - timedelta(days=30)
        dt_max = date.today()

    if "date_range" not in st.session_state:
        st.session_state.date_range = (dt_min, dt_max)
    else:
        st.session_state.date_range = normalize_date_range(
            st.session_state.date_range, dt_min, dt_max
        )

    date_range_raw = st.date_input(
        "Date range", value=st.session_state.date_range,
        min_value=dt_min, max_value=dt_max,
    )
    date_range = normalize_date_range(date_range_raw, dt_min, dt_max)
    st.session_state.date_range = date_range

    # ---- Airlines & Sentiments ----
    airlines = sorted(data["airline"].dropna().unique().tolist()) if "airline" in data.columns else []
    sentiments = sorted(data["airline_sentiment"].dropna().unique().tolist()) if "airline_sentiment" in data.columns else []

    st.session_state.flt_airline = normalize_multiselect_default(
        airlines, st.session_state.get("flt_airline"), fallback_mode="all"
    )
    st.session_state.flt_sent = normalize_multiselect_default(
        sentiments, st.session_state.get("flt_sent"), fallback_mode="all"
    )

    flt_airline = st.multiselect("Pick airlines", airlines, default=st.session_state.flt_airline)
    flt_sent = st.multiselect("Pick sentiments", sentiments, default=st.session_state.flt_sent)

    st.session_state.flt_airline = flt_airline
    st.session_state.flt_sent = flt_sent

    # ---- Hour filter ----
    if "hour" in data.columns:
        if "flt_hour" not in st.session_state or st.session_state.flt_hour is None:
            st.session_state.flt_hour = (0, 23)

        st.markdown("**Hour of day**")
        flt_hour = st.slider(
            "",
            0, 23,
            st.session_state.flt_hour,
            format="%02d:00",
            label_visibility="collapsed",
            help="Filter tweet berdasarkan jam (waktu lokal Asia/Jakarta).",
        )
        st.caption(f"Rentang jam aktif: {flt_hour[0]:02d}:00 – {flt_hour[1]:02d}:59")
        st.session_state.flt_hour = flt_hour
    else:
        flt_hour = (0, 23)

    # ---- Keyword ----
    q = st.text_input(
        "Cari kata (opsional, di text_clean)",
        value=st.session_state.get("q", "")
    )
    st.session_state.q = q

    # ---- Reset all ----
    if st.button("Reset all filters"):
        for k in ["date_range", "flt_airline", "flt_sent", "flt_hour", "q"]:
            st.session_state.pop(k, None)
        st.rerun()

# ===================== APPLY FILTERS =====================
df = data.copy()

# Date filter
if "tweet_created" in df.columns and isinstance(date_range, tuple) and len(date_range) == 2:
    df["_dt_jkt"] = to_jkt_naive(df["tweet_created"])
    d0 = pd.Timestamp(date_range[0])
    d1 = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    df = df[(df["_dt_jkt"] >= d0) & (df["_dt_jkt"] < d1)].drop(columns=["_dt_jkt"])

# Airline & Sentiment
if flt_airline:
    df = df[df["airline"].isin(flt_airline)]
if flt_sent:
    df = df[df["airline_sentiment"].isin(flt_sent)]

# Hour filter
if "hour" in df.columns and isinstance(flt_hour, tuple):
    df = df[df["hour"].notna()]
    if not df.empty:
        h = df["hour"].astype(int)
        df = df[(h >= flt_hour[0]) & (h <= flt_hour[1])]
    else:
        df = pd.DataFrame()

# Keyword on text_clean / text
if q:
    col = "text_clean" if "text_clean" in df.columns else "text"
    df = df[df[col].str.contains(q, case=False, na=False)]

# ===================== TOP ISSUES (simple keyword buckets) =====================
ISSUES = {
    "delay":  ["delay", "terlambat", "late"],
    "refund":  ["refund", "pengembalian", "reimburse", "voucher"],
    "bagasi":  ["bagasi", "baggage", "koper", "lost baggage", "hilang"],
    "service": ["pelayanan", "service", "pramugari", "crew", "cs", "customer service", "komplain"],
}
def tag_issue(t: str) -> str:
    t = (t or "").lower()
    for k, kws in ISSUES.items():
        if any(kw in t for kw in kws):
            return k
    return "(other)"

if "issue" not in df.columns:
    base_col = "text_clean" if "text_clean" in df.columns else "text"
    df["issue"] = df[base_col].apply(tag_issue)

# ===================== KPIs =====================
kpi = kpi_metrics(df) if len(df) else {
    "total_tweets": 0,
    "neg_pct": 0.0,
    "top_neg_airline": "-",
    "delay_share_in_negative_pct": 0.0,
}
neg = df[df["airline_sentiment"] == "negative"] if "airline_sentiment" in df.columns else pd.DataFrame()
delay_share = 100 * (len(neg[neg["issue"] == "delay"]) / len(neg)) if len(neg) else 0.0

c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.2])
c1.metric("Total Tweets", f"{kpi['total_tweets']:,}")
c2.metric("% Negative", f"{kpi['neg_pct']:.1f}%")
c3.metric("Top Airline (Neg)", kpi["top_neg_airline"])
c4.metric("Delay in Negative", f"{delay_share:.1f}%")

# ===================== DOWNLOAD SECTION =====================
st.markdown("### ⬇️ Download")
col_dl1, col_dl2, col_dl3, col_dl4, col_dl5 = st.columns(5)

clean_csv = data.to_csv(index=False).encode("utf-8")
col_dl1.download_button(
    label="CLEANED CSV",
    data=clean_csv,
    file_name=f"tweets_clean_{datetime.now().date()}.csv",
    mime="text/csv",
)

filtered_csv = df.to_csv(index=False).encode("utf-8")
col_dl2.download_button(
    label="FILTERED CSV",
    data=filtered_csv,
    file_name=f"tweets_filtered_{datetime.now().date()}.csv",
    mime="text/csv",
)

sent_df = agg_sentiment(df) if len(df) else pd.DataFrame(columns=["sentiment", "tweets"])
trend_df = agg_hour_trend(df) if len(df) else pd.DataFrame(columns=["hour", "airline_sentiment", "tweets"])
topic_df_dl = agg_topic_vs_sentiment(df) if len(df) else pd.DataFrame(columns=["issue", "airline_sentiment", "tweets"])

col_dl3.download_button(
    label="agg_sentiment.csv",
    data=sent_df.to_csv(index=False).encode("utf-8"),
    file_name="agg_sentiment.csv",
    mime="text/csv",
)
col_dl4.download_button(
    label="agg_hourly_trend.csv",
    data=trend_df.to_csv(index=False).encode("utf-8"),
    file_name="agg_hourly_trend.csv",
    mime="text/csv",
)

cfg = {
    "theme": st.session_state.get("theme_name"),
    "filters": {
        "airlines": flt_airline,
        "sentiments": flt_sent,
        "hour": flt_hour,
        "date_range": [str(date_range[0]), str(date_range[1])] if isinstance(date_range, tuple) else None,
        "query": q,
    },
    "generated_at": datetime.utcnow().isoformat() + "Z",
}
col_dl5.download_button(
    "CONFIG (.json)",
    data=json.dumps(cfg, indent=2).encode("utf-8"),
    file_name="dashboard_config.json",
    mime="application/json",
)

st.divider()

# ===================== TABS =====================
tab_welcome, tab_overview, tab_time, tab_geo, tab_topics, tab_quality = st.tabs(
    ["👋 Selamat Datang", "Overview", "Time Trend", "Map", "Topics", "Data Quality"]
)

# =============== SELAMAT DATANG / PANDUAN PENGGUNA ===============
with tab_welcome:
    st.header("Selamat Datang di ✈️ Airline Tweet Sentiment Dashboard")
    st.markdown("""
        Dashboard ini dirancang untuk menganalisis sentimen publik terhadap maskapai penerbangan 
        berdasarkan data media sosial. Fokus utama kami adalah menyajikan **insight yang cepat dan dapat ditindaklanjuti** mengenai masalah (isu) utama yang dihadapi oleh maskapai.
    """)
    st.markdown("---")

    st.subheader("📚 Panduan Penggunaan")
    st.info("""
        Gunakan menu **Filters** di sidebar (kiri) untuk menyesuaikan tampilan data:
        
        * **Date Range:** Batasi analisis pada periode waktu tertentu.
        * **Pick Airlines:** Pilih maskapai mana saja yang ingin Anda bandingkan.
        * **Pick Sentiments:** Filter berdasarkan sentimen (Negatif, Netral, Positif).
        * **Hour of Day:** Batasi analisis berdasarkan jam (waktu lokal Asia/Jakarta).
        * **Cari Kata:** Lakukan pencarian teks bebas pada kolom *text_clean* (teks yang sudah dibersihkan).
    """)

    st.subheader("📊 Maksud Metrik KPI")
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.markdown("**Total Tweets**")
        st.caption("Jumlah total *tweet* yang tersisa setelah SEMUA filter di sidebar diterapkan.")
    with col_kpi2:
        st.markdown("**% Negative**")
        st.caption("Persentase *tweet* bernada negatif dari Total Tweets yang lolos filter.")
    with col_kpi3:
        st.markdown("**Top Airline (Neg)**")
        st.caption("Maskapai dengan jumlah *tweet* negatif paling banyak (bukan persentase).")
    with col_kpi4:
        st.markdown("**Delay in Negative**")
        st.caption("Persentase *tweet* negatif yang secara spesifik membahas isu **'delay'**.")

    st.markdown("---")
    st.subheader("🎨 Isi Tiap Tab")
    st.markdown("""
    * **Overview:** Distribusi sentimen keseluruhan (Bar + Pie) dan contoh tweet acak.
    * **Time Trend:** Tren jumlah *tweet* per jam, dipecah berdasarkan sentimen.
    * **Map:** Visualisasi lokasi *tweet* (jika koordinat tersedia).
    * **Topics:** Word Cloud + grafik Issue vs Sentiment.
    * **Data Quality:** Ringkasan kualitas data dan preview.
    """)

# =============== OVERVIEW ===============
with tab_overview:
    left, right = st.columns(2)
    s = agg_sentiment(df) if len(df) else pd.DataFrame(columns=["sentiment", "tweets"])
    left.subheader("Tweets by Sentiment")
    if len(s):
        left.plotly_chart(fig_sentiment_bar(s), use_container_width=True)
    else:
        left.info("Tidak ada data untuk ditampilkan.")

    right.subheader("Sentiment Share")
    if len(s):
        right.plotly_chart(fig_sentiment_pie(s), use_container_width=True)
    else:
        right.info("Tidak ada data untuk ditampilkan.")

    st.markdown("**Random tweet (berdasarkan filter saat ini):**")
    if len(df) > 0:
        if "rand_idx" not in st.session_state or st.session_state.get("rand_idx") >= len(df):
            st.session_state.rand_idx = np.random.randint(0, len(df))
        try:
            st.info(df.iloc[st.session_state.rand_idx]["text"])
        except Exception:
            st.info(df.sample(1)["text"].iat[0])
    else:
        st.warning("Data kosong setelah filter.")

# =============== TIME TREND ===============
with tab_time:
    st.subheader("Tweets per Hour")
    trend = agg_hour_trend(df) if len(df) else pd.DataFrame(columns=["hour", "airline_sentiment", "tweets"])
    if len(trend):
        st.plotly_chart(fig_hour_trend(trend), use_container_width=True)
        with st.expander("Show aggregated table"):
            st.dataframe(trend.sort_values(["hour", "airline_sentiment"]))
    else:
        st.info("Kolom waktu tidak tersedia / data kosong.")

# =============== MAP ===============
with tab_geo:
    st.subheader("Tweet Locations")
    lat_col = "lat" if "lat" in df.columns else ("latitude" if "latitude" in df.columns else None)
    lon_col = "lon" if "lon" in df.columns else ("longitude" if "longitude" in df.columns else None)
    if lat_col and lon_col:
        geo_df = df[[lat_col, lon_col]].dropna().rename(columns={lat_col: "lat", lon_col: "lon"})
        if len(geo_df):
            st.map(geo_df, use_container_width=True)
            st.caption("Menampilkan titik dengan koordinat valid (lat/lon).")
        else:
            st.info("Tidak ada baris dengan koordinat valid.")
    else:
        st.info("Kolom koordinat tidak ditemukan. Pastikan ada 'latitude/longitude' atau 'lat/lon'.")

# =============== TOPICS / WORDCLOUD ===============
with tab_topics:
    st.subheader("Word Cloud & Topic Analysis")
    c1, c2 = st.columns(2)
    pick_airline = c1.selectbox(
        "Airline for word cloud",
        ["(all)"] + (sorted(df["airline"].dropna().unique().tolist()) if "airline" in df.columns else []),
    )
    pick_sent = c2.selectbox(
        "Sentiment",
        ["(all)"] + (sorted(df["airline_sentiment"].dropna().unique().tolist()) if "airline_sentiment" in df.columns else []),
    )

    wc_df = df.copy()
    if pick_airline != "(all)" and "airline" in wc_df.columns:
        wc_df = wc_df[wc_df["airline"] == pick_airline]
    if pick_sent != "(all)" and "airline_sentiment" in wc_df.columns:
        wc_df = wc_df[wc_df["airline_sentiment"] == pick_sent]

    # Generate the word cloud if there is text to process
    if "text_clean" in wc_df.columns and len(wc_df) > 0:
        text_blob = " ".join(wc_df["text_clean"].astype(str).tolist()).strip()
        if text_blob:
            wc = WordCloud(
                stopwords=STOPWORDS,
                background_color=THEME["wc_bg"],
                width=1000,
                height=400
            ).generate(text_blob)
            fig, ax = plt.subplots(figsize=(10, 4), facecolor=THEME["mpl_face"])
            ax.imshow(wc)
            ax.axis("off")
            fig.patch.set_facecolor(THEME["mpl_face"])
            st.pyplot(fig)
        else:
            st.info("Teks kosong setelah pembersihan.")
    else:
        st.info("Tidak ada teks untuk wordcloud.")

    # Topic vs Sentiment
    st.subheader("Topic vs Sentiment")
    topic_df = agg_topic_vs_sentiment(df) if len(df) else pd.DataFrame(columns=["issue", "airline_sentiment", "tweets"])
    if len(topic_df):
        st.plotly_chart(fig_topic_stack(topic_df), use_container_width=True)
    else:
        st.info("Data topic kosong.")

    st.markdown("**Top Issues (simple keyword buckets)**")
    if len(df):
        issue_tab = (
            df.groupby(["issue", "airline_sentiment"])
              .size()
              .reset_index(name="tweets")
              .sort_values("tweets", ascending=False)
        )
        st.dataframe(issue_tab, use_container_width=True)
    else:
        st.info("Tidak ada data untuk dihitung.")

# =============== DATA QUALITY ===============
with tab_quality:
    st.subheader("Data Quality Summary")
    q_info = {
        "shape": f"{data.shape[0]} x {data.shape[1]}",
        "duplicates_total": int(data.duplicated().sum()),
        "avg_missing_%": round(100 * data.isna().mean().mean(), 2),
    }
    if "airline_sentiment" in data.columns:
        class_bal = data["airline_sentiment"].value_counts(normalize=True).mul(100).round(1).to_dict()
        q_info["class_balance_%"] = class_bal
    if "lang" in data.columns:
        lang_share = data["lang"].value_counts(normalize=True).mul(100).round(1).to_dict()
        q_info["language_share_%"] = lang_share

    st.write(q_info)

    if "airline" in data.columns and len(data):
        share = data["airline"].value_counts(normalize=True).max()
        if share > 0.6:
            st.warning("⚠️ Sampling bias: distribusi tweet sangat timpang antar maskapai (>60% pada satu maskapai).")

    st.markdown("**Preview (cleaned)**")
    st.dataframe(data.head(10), use_container_width=True)

st.divider()
st.caption("© Tim 2 — Program Studi Sains Data Terapan | Proyek Akhir MK Pengembangan Aplikasi")
