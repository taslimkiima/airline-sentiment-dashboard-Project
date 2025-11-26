# src/clean.py (KODE FINAL)
import re
import numpy as np
import pandas as pd

# ===== Alias maskapai → nama resmi (boleh kamu tambah) =====
AIRLINE_ALIASES = {
    "garuda": "Garuda Indonesia", "garuda indonesia": "Garuda Indonesia",
    "citilink": "Citilink", "lion": "Lion Air", "lion air": "Lion Air",
    "batik": "Batik Air", "batik air": "Batik Air", "airasia": "Indonesia AirAsia",
    "airasia indonesia": "Indonesia AirAsia", "super air jet": "Super Air Jet",
    "pelita": "Pelita Air", "pelita air": "Pelita Air", "wings": "Wings Air",
    "wings air": "Wings Air", "transnusa": "TransNusa",
    "virgin america": "Virgin America", "virgin": "Virgin America",
    "united": "United", "delta": "Delta", "southwest": "Southwest", "american": "American",
}

# ===== Kandidat nama kolom standar (tahan skema beda) =====
CAND_AIRLINE    = ["airline", "airline_name", "maskapai", "carrier", "brand"]
CAND_TEXT     = ["text", "tweet", "full_text", "content"]
CAND_SENT     = ["airline_sentiment", "sentiment", "label"]
CAND_DATETIME = ["tweet_created", "created_at", "date", "datetime", "time"]
CAND_LAT      = ["latitude", "lat", "y"]
CAND_LON      = ["longitude", "lon", "lng", "long", "x"]

def _first_match(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Ambil nama kolom pertama yang match (case-insensitive, fuzzy ringan)."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    norm = {re.sub(r"[\s_]+", "", c.lower()): c for c in df.columns}
    for name in candidates:
        nm = re.sub(r"[\s_]+", "", name.lower())
        if nm in norm:
            return norm[nm]
    return None

def _ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Deteksi & rename ke kolom standar yang dipakai app (tanpa ubah UI)."""
    out = df.copy()
    col_airline = _first_match(out, CAND_AIRLINE)
    col_text    = _first_match(out, CAND_TEXT)
    col_sent    = _first_match(out, CAND_SENT)
    col_dt      = _first_match(out, CAND_DATETIME)

    ren = {}
    if col_airline and col_airline != "airline": ren[col_airline] = "airline"
    if col_text    and col_text    != "text": ren[col_text] = "text"
    if col_sent    and col_sent    != "airline_sentiment": ren[col_sent] = "airline_sentiment"
    if col_dt      and col_dt      != "tweet_created": ren[col_dt] = "tweet_created"
    if ren:
        out = out.rename(columns=ren)

    for c in ["airline", "text", "airline_sentiment"]:
        if c not in out.columns: out[c] = np.nan
    return out

# ====== Normalisasi nilai ======
def normalize_airline_values(df: pd.DataFrame) -> pd.DataFrame:
    """Map alias maskapai → nama resmi; title-case utk yang tidak di-mapping."""
    if "airline" not in df.columns: return df
    s = df["airline"].astype(str).str.strip().str.lower()
    df["airline"] = s.map(AIRLINE_ALIASES).fillna(s.str.title())
    return df

def normalize_text_series(s: pd.Series) -> pd.Series:
    """
    Lowercase + buang URL/mention/hashtag/simbol; jaga huruf beraksen & angka.
    """
    s = s.astype(str).str.lower()
    s = s.str.replace(r"http\S+|www\.\S+", " ", regex=True)  # URL
    s = s.str.replace(r"@\w+|#\w+", " ", regex=True)         # mention/hashtag
    s = s.str.replace(r"[^A-Za-zÀ-ÿ0-9\s]", " ", regex=True) # simbol/emoji
    return s.str.replace(r"\s+", " ", regex=True).str.strip()

def standardize_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Samakan label → positive / neutral / negative (support id/en)."""
    s = df["airline_sentiment"].astype(str).str.lower().str.strip()
    s = s.replace({
        "pos": "positive", "positif": "positive",
        "neg": "negative", "negatif": "negative",
        "neu": "neutral",  "netral": "neutral"
    })
    df["airline_sentiment"] = s.where(s.isin(["positive", "neutral", "negative"]), other="neutral")
    return df

def parse_datetime(df: pd.DataFrame, col: str = "tweet_created") -> pd.DataFrame:
    """
    Parse kolom waktu + turun jam WIB ke kolom 'hour'.
    """
    out = df.copy()
    if col in out.columns:
        dt = pd.to_datetime(out[col], errors="coerce", utc=True)
        try:
            dt = dt.dt.tz_convert("Asia/Jakarta")
        except Exception:
            dt = pd.to_datetime(out[col], errors="coerce").dt.tz_localize("Asia/Jakarta", nonexistent="shift_forward", ambiguous="NaT")
        out[col] = dt
        
        # Gunakan Int64 (nullable integer) untuk kolom hour
        out["hour"] = dt.dt.hour.astype('Int64', errors='ignore') 
        
        out["date"] = dt.dt.date
    else:
        out["hour"] = pd.NA
        out["date"] = pd.NaT
    return out

def normalize_geo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Samakan nama kolom lat/lon → 'lat','lon' + validasi rentang & tipe numeric.
    """
    out = df.copy()
    cols_lower = {c.lower(): c for c in out.columns}

    if "latitude" not in out.columns:
        for k in ["lat", "y"]:
            if k in cols_lower:
                out.rename(columns={cols_lower[k]: "latitude"}, inplace=True)
                break
    if "longitude" not in out.columns:
        for k in ["lon", "lng", "long", "x"]:
            if k in cols_lower:
                out.rename(columns={cols_lower[k]: "longitude"}, inplace=True)
                break

    if "latitude" not in out.columns: out["latitude"] = np.nan
    if "longitude" not in out.columns: out["longitude"] = np.nan

    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out.loc[~out["latitude"].between(-90, 90), "latitude"] = np.nan
    out.loc[~out["longitude"].between(-180, 180), "longitude"] = np.nan

    out["lat"] = out["latitude"]
    out["lon"] = out["longitude"]
    return out

TOPIC_DELAY_REGEX = re.compile(
    r"\b(delay|telat|late|ngaret|otp|on time|resched|cancel|refund|bagasi|koper|service|komplain)\b",
    flags=re.IGNORECASE
)

def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Pembersihan & feature engineering."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    out = _ensure_standard_columns(out)
    out = out.dropna(how="all").drop_duplicates()

    out = normalize_airline_values(out)
    out["text_clean"] = normalize_text_series(out["text"])
    out = standardize_sentiment(out)
    out = parse_datetime(out, "tweet_created")
    out = normalize_geo(out)

    out["is_negative"] = (out["airline_sentiment"] == "negative").astype(int)
    out["topic_delay"] = out["text_clean"].str.contains(TOPIC_DELAY_REGEX, na=False).astype(int)

    out = out.dropna(subset=["airline", "airline_sentiment", "text_clean"])
    return out
