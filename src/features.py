# src/features.py (KODE FINAL)
import pandas as pd

SENTIMENT_ORDER = ["negative", "neutral", "positive"]

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "airline_sentiment" not in out.columns: out["airline_sentiment"] = "neutral"
    if "airline" not in out.columns: out["airline"] = "-"
    if "is_negative" not in out.columns: out["is_negative"] = (out.get("airline_sentiment", "neutral") == "negative").astype(int)
    if "topic_delay" not in out.columns: out["topic_delay"] = 0
    if "hour" not in out.columns: out["hour"] = pd.NA
    return out


def kpi_metrics(df: pd.DataFrame) -> dict:
    df = _ensure_cols(df)
    total = int(len(df))

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
    """Output kolom: sentiment, tweets"""
    df = _ensure_cols(df)

    s = (
        df["airline_sentiment"]
        .value_counts(dropna=False)
        .rename_axis("sentiment")
        .reset_index(name="tweets")
    )

    full = (
        pd.DataFrame({"sentiment": SENTIMENT_ORDER})
        .merge(s, on="sentiment", how="left")
        .fillna({"tweets": 0})
    )
    full["tweets"] = full["tweets"].astype(int)
    return full


def agg_hour_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Output kolom: hour, airline_sentiment, tweets"""
    df = _ensure_cols(df)
    empty_df_schema = pd.DataFrame(columns=["hour", "airline_sentiment", "tweets"])
    
    if "hour" not in df.columns or df["hour"].isna().all():
        return empty_df_schema

    g = (
        df.dropna(subset=["hour"])
          .groupby(["hour", "airline_sentiment"])
          .size()
          .reset_index(name="tweets")
    )
    
    if g.empty:
        return empty_df_schema

    g["hour"] = g["hour"].astype(int) 
    g["airline_sentiment"] = pd.Categorical(g["airline_sentiment"], SENTIMENT_ORDER, ordered=True)
    g = g.sort_values(["hour", "airline_sentiment"]).reset_index(drop=True)
    return g


def agg_topic_vs_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Output kolom: issue, airline_sentiment, tweets"""
    df = _ensure_cols(df)
    
    if 'issue' not in df.columns:
         df['issue'] = '(other)' 
         
    g = (
        df.groupby(["issue", "airline_sentiment"])
          .size()
          .reset_index(name="tweets") 
    )

    g["airline_sentiment"] = pd.Categorical(g["airline_sentiment"], SENTIMENT_ORDER, ordered=True)
    g = g.sort_values(["issue", "airline_sentiment"]).reset_index(drop=True)
    return g
