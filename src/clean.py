# src/clean.py

import re
import numpy as np
import pandas as pd

# ===== Alias maskapai → nama resmi (boleh kamu tambah) =====
AIRLINE_ALIASES = {
    # ======================
    # Maskapai Indonesia
    # ======================
    # Garuda
    "garuda": "Garuda Indonesia",
    "garuda indonesia": "Garuda Indonesia",
    "ga": "Garuda Indonesia",

    # Citilink
    "citilink": "Citilink",
    "qg": "Citilink",

    # Lion Group
    "lion": "Lion Air",
    "lion air": "Lion Air",
    "jt": "Lion Air",

    "batik": "Batik Air",
    "batik air": "Batik Air",
    "id": "Batik Air",  # kode IATA Batik Air

    "wings": "Wings Air",
    "wings air": "Wings Air",

    # AirAsia Indonesia
    "airasia": "AirAsia Indonesia",
    "air asia": "AirAsia Indonesia",
    "airasia indonesia": "AirAsia Indonesia",
    "air asia indonesia": "AirAsia Indonesia",

    # Lain-lain (kalau nanti muncul di data Indo)
    "super air jet": "Super Air Jet",
    "pelita": "Pelita Air",
    "pelita air": "Pelita Air",
    "transnusa": "TransNusa",
    "sriwijaya": "Sriwijaya Air",
    "sriwijaya air": "Sriwijaya Air",
    "nam air": "NAM Air",

    # ======================
    # Maskapai US (dataset Kaggle asli)
    # ======================
    "virgin america": "Virgin America",
    "virgin": "Virgin America",
    "united": "United",
    "delta": "Delta",
    "southwest": "Southwest",
    "american": "American",
}

# ===== Kandidat nama kolom standar (tahan skema beda) =====
CAND_AIRLINE    = ["airline", "airline_name", "maskapai", "carrier", "brand"]
CAND_TEXT       = ["text", "tweet", "full_text", "content"]
CAND_SENT       = ["airline_sentiment", "sentiment", "label"]
CAND_DATETIME   = ["tweet_created", "created_at", "date", "datetime", "time"]
CAND_LAT        = ["latitude", "lat", "y"]
CAND_LON        = ["longitude", "lon", "lng", "long", "x"]


def _first_match(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Ambil nama kolom pertama yang match (case-insensitive, fuzzy ringan)."""
    lower_map = {c.lower(): c for c in df.columns}
    # match langsung (case-insensitive)
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    # fuzzy ringan: buang spasi & underscore
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

    ren: dict[str, str] = {}
    if col_airline and col_airline != "airline":
        ren[col_airline] = "airline"
    if col_text and col_text != "text":
        ren[col_text] = "text"
    if col_sent and col_sent != "airline_sentiment":
        ren[col_sent] = "airline_sentiment"
    if col_dt and col_dt != "tweet_created":
        ren[col_dt] = "tweet_created"
    if ren:
        out = out.rename(columns=ren)

    # pastikan selalu ada kolom inti
    for c in ["airline", "text", "airline_sentiment"]:
        if c not in out.columns:
            out[c] = np.nan

    return out


# ====== Normalisasi nilai ======
def normalize_airline_values(df: pd.DataFrame) -> pd.DataFrame:
    """Map alias maskapai → nama resmi; title-case utk yang tidak di-mapping."""
    if "airline" not in df.columns:
        return df
    s = df["airline"].astype(str).str.strip().str.lower()
    df["airline"] = s.map(AIRLINE_ALIASES).fillna(s.str.title())
    return df


def normalize_text_series(s: pd.Series) -> pd.Series:
    """
    Lowercase + buang URL/mention/hashtag/simbol; jaga huruf beraksen & angka.
    """
    s = s.astype(str).str.lower()
    # buang URL
    s = s.str.replace(r"http\S+|www\.\S+", " ", regex=True)
    # buang mention & hashtag
    s = s.str.replace(r"@\w+|#\w+", " ", regex=True)
    # buang simbol/emoji, sisakan huruf beraksen & angka
    s = s.str.replace(r"[^A-Za-zÀ-ÿ0-9\s]", " ", regex=True)
    # rapikan spasi
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def standardize_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Samakan label → positive / neutral / negative (support id/en)."""
    s = df["airline_sentiment"].astype(str).str.lower().str.strip()
    s = s.replace({
        "pos": "positive",
        "positif": "positive",
        "neg": "negative",
        "negatif": "negative",
        "neu": "neutral",
        "netral": "neutral",
    })
    df["airline_sentiment"] = s.where(
        s.isin(["positive", "neutral", "negative"]),
        other="neutral",
    )
    return df


def parse_datetime(df: pd.DataFrame, col: str = "tweet_created") -> pd.DataFrame:
    """
    Parse kolom waktu + turunkan jam WIB ke kolom 'hour' dan tanggal ke 'date'.
    """
    out = df.copy()
    if col in out.columns:
        # coba asumsikan UTC dulu
        dt = pd.to_datetime(out[col], errors="coerce", utc=True)
        try:
            dt = dt.dt.tz_convert("Asia/Jakarta")
        except Exception:
            # fallback: parse lokal lalu set tz Asia/Jakarta
            dt = pd.to_datetime(out[col], errors="coerce").dt.tz_localize(
                "Asia/Jakarta",
                nonexistent="shift_forward",
                ambiguous="NaT",
            )
        out[col] = dt

        # nullable integer untuk kolom hour
        out["hour"] = dt.dt.hour.astype("Int64")
        out["date"] = dt.dt.date
    else:
        out["hour"] = pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"))
        out["date"] = pd.NaT
    return out


def normalize_geo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Samakan nama kolom lat/lon → 'lat','lon' + validasi rentang & tipe numeric.
    """
    out = df.copy()
    cols_lower = {c.lower(): c for c in out.columns}

    # deteksi latitude
    if "latitude" not in out.columns:
        for k in ["lat", "y"]:
            if k in cols_lower:
                out.rename(columns={cols_lower[k]: "latitude"}, inplace=True)
                break
    # deteksi longitude
    if "longitude" not in out.columns:
        for k in ["lon", "lng", "long", "x"]:
            if k in cols_lower:
                out.rename(columns={cols_lower[k]: "longitude"}, inplace=True)
                break

    if "latitude" not in out.columns:
        out["latitude"] = np.nan
    if "longitude" not in out.columns:
        out["longitude"] = np.nan

    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")

    # validasi rentang
    out.loc[~out["latitude"].between(-90, 90), "latitude"] = np.nan
    out.loc[~out["longitude"].between(-180, 180), "longitude"] = np.nan

    # alias pendek
    out["lat"] = out["latitude"]
    out["lon"] = out["longitude"]
    return out


TOPIC_DELAY_REGEX = re.compile(
    r"\b("
    r"delay|telat|late|ngaret|otp|on time|"
    r"resched|reschedule|cancel|canceled|cancelled|"
    r"refund|refundnya|bagasi|koper|baggage|"
    r"service|layanan|komplain|complain"
    r")\b",
    flags=re.IGNORECASE,
)


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Pembersihan & feature engineering utama."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # 1. Standarisasi nama kolom inti
    out = _ensure_standard_columns(out)

    # 2. Drop baris kosong total & duplikat
    out = out.dropna(how="all").drop_duplicates()

    # 3. Normalisasi nama maskapai
    out = normalize_airline_values(out)

    # 4. Bersihkan teks
    out["text_clean"] = normalize_text_series(out["text"])

    # 5. Standarisasi label sentimen
    out = standardize_sentiment(out)

    # 6. Parse waktu ke WIB + jam & tanggal
    out = parse_datetime(out, "tweet_created")

    # 7. Normalisasi geolokasi
    out = normalize_geo(out)

    # 8. Feature engineering
    out["is_negative"] = (out["airline_sentiment"] == "negative").astype(int)
    out["topic_delay"] = out["text_clean"].str.contains(
        TOPIC_DELAY_REGEX,
        na=False,
    ).astype(int)

    # 9. Pastikan hanya baris yang punya airline, sentiment, dan text_clean
    out = out.dropna(subset=["airline", "airline_sentiment", "text_clean"])

    return out
