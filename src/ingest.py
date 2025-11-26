# src/ingest.py (KODE FINAL)
import os
import io
from typing import Optional, Union
import pandas as pd
import streamlit as st

# Default file: cari Tweets.csv di root project (satu level di atas folder src)
DEFAULT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Tweets.csv"))

# delimiter & encoding fallback
_CSV_SEPS = [",", ";", "\t", "|"]
_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1"]


def coerce_columns_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Trim + lowercase semua nama kolom supaya deteksi kolom lebih mudah dalam pipeline."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _read_csv_from_bytes(raw: bytes) -> pd.DataFrame:
    """Coba baca CSV dari bytes dengan beberapa delimiter & encoding."""
    last_err = None
    for sep in _CSV_SEPS:
        for enc in _ENCODINGS:
            try:
                bio = io.BytesIO(raw)  # reset pointer tiap percobaan
                df = pd.read_csv(bio, sep=sep, encoding=enc, engine="python", dtype={"tweet_id": "string"})
                if df.shape[1] > 1:
                    return df
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Gagal membaca CSV dari bytes. Terakhir: {last_err}")


@st.cache_data(show_spinner=True)
def read_any_csv(obj: Optional[Union[str, bytes, bytearray, "UploadedFile"]]) -> pd.DataFrame:
    """
    Baca CSV dari: path string, UploadedFile, atau bytes.
    """
    # 1) None → pakai default file di root
    if obj is None:
        if not os.path.exists(DEFAULT_FILE):
            raise FileNotFoundError(f"CSV tidak ditemukan: {DEFAULT_FILE}")
        try:
            return pd.read_csv(DEFAULT_FILE, dtype={"tweet_id": "string"})
        except UnicodeDecodeError:
            return pd.read_csv(DEFAULT_FILE, dtype={"tweet_id": "string"}, encoding="latin-1")

    # 2) UploadedFile (Streamlit) → punya .read()
    if hasattr(obj, "read"):
        raw = obj.read()
        return _read_csv_from_bytes(raw)

    # 3) bytes / bytearray
    if isinstance(obj, (bytes, bytearray)):
        return _read_csv_from_bytes(obj)

    # 4) path string (termasuk "Tweets.csv" di root GitHub)
    if isinstance(obj, str):
        try:
            return pd.read_csv(obj, dtype={"tweet_id": "string"})
        except UnicodeDecodeError:
            return pd.read_csv(obj, dtype={"tweet_id": "string"}, encoding="latin-1")
        except FileNotFoundError as e:
             # Coba path absolut sebagai fallback
             abs_path = os.path.abspath(obj)
             if os.path.exists(abs_path):
                 return pd.read_csv(abs_path, dtype={"tweet_id": "string"})
             raise e

    # 5) tipe lain tidak didukung
    raise TypeError(f"Tipe objek tidak didukung untuk read_any_csv: {type(obj)}")


@st.cache_data(show_spinner=True)
def load_csv(path: Optional[Union[str, bytes, bytearray, "UploadedFile"]] = None) -> pd.DataFrame:
    """Backward compatible loader (nama fungsi lama)."""
    return read_any_csv(path)
