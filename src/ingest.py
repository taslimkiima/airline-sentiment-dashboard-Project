# src/ingest.py
import os
from typing import Optional
import pandas as pd
import streamlit as st

# Default file: cari Tweets.csv di root project (satu level di atas folder src)
DEFAULT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Tweets.csv"))

@st.cache_data(show_spinner=True)
def load_csv(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load CSV ke DataFrame, jaga agar tweet_id string (tidak jadi notasi ilmiah),
    dan fallback encoding jika UTF-8 gagal.
    """
    path = os.path.abspath(path or DEFAULT_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV tidak ditemukan: {path}")
    try:
        df = pd.read_csv(path, dtype={"tweet_id": "string"})
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype={"tweet_id": "string"}, encoding="latin-1")
    return df
