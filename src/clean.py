# src/clean.py
import re
import numpy as np
import pandas as pd

def parse_datetime(df: pd.DataFrame, col: str = "tweet_created") -> pd.DataFrame:
    """Parse kolom datetime + turunkan jam ke kolom 'hour'."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        df["hour"] = df[col].dt.hour
    return df

def normalize_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Samakan nama kolom lat/lon dan validasi rentang koordinat."""
    cols_lower = {c.lower(): c for c in df.columns}
    if "latitude" in cols_lower:
        df.rename(columns={cols_lower["latitude"]: "lat"}, inplace=True)
    if "longitude" in cols_lower:
        df.rename(columns={cols_lower["longitude"]: "lon"}, inplace=True)
    if "long" in df.columns and "lon" not in df.columns:
        df.rename(columns={"long": "lon"}, inplace=True)
    if "lat" in df.columns:
        df.loc[~df["lat"].between(-90, 90), "lat"] = np.nan
    if "lon" in df.columns:
        df.loc[~df["lon"].between(-180, 180), "lon"] = np.nan
    return df

def normalize_text_series(s: pd.Series) -> pd.Series:
    """Lowercase + hilangkan URL, mention, hashtag, simbol; rapikan spasi."""
    s = s.astype(str).str.lower()
    s = s.str.replace(r"http\S+|www\.\S+", " ", regex=True)
    s = s.str.replace(r"@\w+|#\w+", " ", regex=True)
    s = s.str.replace(r"[^a-z\s]", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()

def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pembersihan & feature engineering:
    - drop dupe & mandatory cols
    - parse datetime & hour
    - normalisasi geo & text
    - fitur: is_negative, topic_delay (rule keywords)
    """
    df = df.copy()

    # pastikan kolom inti ada
    for col in ["airline_sentiment", "airline", "text"]:
        if col not in df.columns:
            df[col] = np.nan

    # bersihkan basic
    df = df.dropna(subset=["airline_sentiment", "airline", "text"])
    df = df.drop_duplicates()

    # waktu, lokasi, dan teks
    df = parse_datetime(df, "tweet_created")
    df = normalize_geo(df)
    df["text_clean"] = normalize_text_series(df["text"])

    # fitur
    df["is_negative"] = (df["airline_sentiment"] == "negative").astype(int)
    delay_words = r"\b(delay|late|cancel|gate|boarding|resched|overbook|divert)\b"
    df["topic_delay"] = df["text_clean"].str.contains(delay_words).astype(int)

    return df
