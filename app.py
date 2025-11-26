# app.py
import os
import io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

from src.ingest import load_csv
from src.clean import clean_pipeline
from src.features import (
    kpi_metrics, agg_sentiment, agg_hour_trend, agg_topic_vs_sentiment, sentiment_analysis
)
from src.viz import (
    apply_plotly_theme, fig_sentiment_bar, fig_sentiment_pie, fig_hour_trend, fig_topic_stack
)

st.set_page_config(page_title="Airline Tweet Sentiment", layout="wide")
st.title("✈️ Airline Tweet Sentiment Dashboard")
st.caption("Fokus: manajemen data (clean/transform) + visual interaktif + insight.")

apply_plotly_theme()

DEFAULT_FILE = os.path.join(os.path.dirname(__file__), "Tweets.csv")
data_raw = None
try:
    data_raw = load_csv(DEFAULT_FILE)
except Exception as e:
    st.warning(f"Catatan: {e}")

with st.sidebar:
    st.header("Data Source")
    up = st.file_uploader("Upload CSV (opsional)", type=["csv"])
    if up is not None:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(up.read()); tmp.flush()
        data_raw = load_csv(tmp.name)

if data_raw is None or data_raw.empty:
    st.error("Tidak ada data yang bisa dimuat. Pastikan `Tweets.csv` ada di root, atau upload CSV.")
    st.stop()

data = clean_pipeline(data_raw)
data = sentiment_analysis(data)

# Filter Sidebar
with st.sidebar:
    st.header("Filters")
    airlines = sorted(data["airline"].dropna().unique().tolist())
    sentiments = sorted(data["airline_sentiment"].dropna().unique().tolist())
    flt_airline = st.multiselect("Pick airlines", airlines, default=airlines[:3])
    flt_sent = st.multiselect("Pick sentiments", sentiments, default=sentiments)
    flt_hour = st.slider("Hour of day", 0, 23, (0, 23))

df = data.copy()
if flt_airline:
    df = df[df["airline"].isin(flt_airline)]
if flt_sent:
    df = df[df["airline_sentiment"].isin(flt_sent)]
if "hour" in df.columns and isinstance(flt_hour, tuple):
    df = df[(df["hour"] >= flt_hour[0]) & (df["hour"] <= flt_hour[1])]

kpi = kpi_metrics(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Tweets", f"{kpi['total_tweets']:,}")
c2.metric("% Negative", f"{kpi['neg_pct']:.1f}%")
c3.metric("Top Airline (Neg)", kpi["top_neg_airline"])
c4.metric("Delay in Negative", f"{kpi['delay_share_in_negative_pct']:.1f}%")

st.markdown("### ⬇️ Download")
col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)

clean_csv = data.to_csv(index=False).encode("utf-8")
col_dl1.download_button(
    label="Download CLEANED CSV",
    data=clean_csv,
    file_name=f"tweets_clean_{datetime.now().date()}.csv",
    mime="text/csv",
)

filtered_csv = df.to_csv(index=False).encode("utf-8")
col_dl2.download_button(
    label="Download FILTERED CSV",
    data=filtered_csv,
    file_name=f"tweets_filtered_{datetime.now().date()}.csv",
    mime="text/csv",
)

sent_df = agg_sentiment(df)
trend_df = agg_hour_trend(df)
col_dl3.download_button(
    label="Download agg_sentiment.csv",
    data=sent_df.to_csv(index=False).encode("utf-8"),
    file_name="agg_sentiment.csv",
    mime="text/csv",
)
col_dl4.download_button(
    label="Download agg_hourly_trend.csv",
    data=trend_df.to_csv(index=False).encode("utf-8"),
    file_name="agg_hourly_trend.csv",
    mime="text/csv",
)

st.divider()

# ---- Tabs ----
tab_overview, tab_time, tab_geo, tab_topics, tab_quality = st.tabs(
    ["Overview", "Time Trend", "Map", "Topics", "Data Quality"]
)

# =============== OVERVIEW ===============
with tab_overview:
    left, right = st.columns(2)
    s = agg_sentiment(df)
    left.subheader("Tweets by Sentiment")
    left.plotly_chart(fig_sentiment_bar(s), use_container_width=True)
    right.subheader("Sentiment Share")
    right.plotly_chart(fig_sentiment_pie(s), use_container_width=True)

    st.markdown("**Random tweet (berdasarkan filter saat ini):**")
    if len(df) > 0:
        st.info(df.sample(1)["text"].iat[0])
    else:
        st.warning("Data kosong setelah filter.")
