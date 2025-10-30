# src/features.py
import pandas as pd

def kpi_metrics(df: pd.DataFrame) -> dict:
    total = len(df)
    neg_pct = round(100 * df["is_negative"].mean(), 1) if total else 0.0

    if (df["is_negative"] == 1).any():
        top_neg = df.loc[df["is_negative"] == 1, "airline"].value_counts().idxmax()
    else:
        top_neg = "-"

    if (df["is_negative"] == 1).any():
        sub = df.loc[df["is_negative"] == 1, "topic_delay"]
        delay_in_neg = round(100 * sub.mean(), 1)
    else:
        delay_in_neg = 0.0

    return {
        "total_tweets": total,
        "neg_pct": neg_pct,
        "top_neg_airline": top_neg,
        "delay_share_in_negative_pct": delay_in_neg,
    }

def agg_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    s = df["airline_sentiment"].value_counts().reset_index()
    s.columns = ["sentiment", "tweets"]
    return s

def agg_hour_trend(df: pd.DataFrame) -> pd.DataFrame:
    if "hour" not in df.columns:
        return df.iloc[0:0].copy()
    return df.groupby(["hour", "airline_sentiment"]).size().reset_index(name="tweets")

def agg_topic_vs_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    x = df.assign(topic=df["topic_delay"].map({1: "Delay-related", 0: "Other"}))
    return x.groupby(["topic", "airline_sentiment"]).size().reset_index(name="tweets")
