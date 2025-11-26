# src/viz.py (FINAL)

import plotly.express as px
import plotly.io as pio
import pandas as pd

# Warna untuk sentimen (positive, neutral, negative)
SENTI_COLORS = {"negative": "red", "neutral": "blue", "positive": "green"} 

PALETTE = px.colors.qualitative.Safe 

def fig_sentiment_bar(sent_count: pd.DataFrame):
    """Buat grafik batang untuk distribusi sentimen."""
    fig = px.bar(
        sent_count, 
        x="sentiment", 
        y="tweets", 
        text="tweets", 
        height=420, 
        color="sentiment", 
        color_discrete_map=SENTI_COLORS
    )
    fig.update_traces(textposition="outside") 
    fig.update_layout(xaxis_title="Sentiment", yaxis_title="Number of Tweets")
    return fig

def fig_sentiment_pie(sent_count: pd.DataFrame):
    """Buat grafik pie untuk distribusi sentimen."""
    return px.pie(
        sent_count, 
        names="sentiment", 
        values="tweets", 
        height=420, 
        hole=0.35, 
        color="sentiment", 
        color_discrete_map=SENTI_COLORS, 
        title="Distribution by Sentiment"
    )

def fig_hour_trend(trend: pd.DataFrame):
    """Buat grafik garis untuk tren tweet berdasarkan jam."""
    return px.line(
        trend, 
        x="hour", 
        y="tweets", 
        color="airline_sentiment", 
        markers=True, 
        height=460, 
        color_discrete_map=SENTI_COLORS,
        labels={"tweets": "Number of Tweets", "hour": "Hour of Day (Asia/Jakarta)"}
    )

def fig_topic_stack(df_topic: pd.DataFrame):
    """Buat grafik batang bertumpuk untuk analisis topik dan sentimen."""
    return px.bar(
        df_topic, 
        x="issue", 
        y="tweets", 
        color="airline_sentiment", 
        barmode="stack", 
        height=420, 
        color_discrete_map=SENTI_COLORS,
        labels={"tweets": "Number of Tweets", "issue": "Identified Topic/Issue"},
        title="Issue Breakdown by Sentiment"
    )
