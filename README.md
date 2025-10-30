# ✈️ Airline Tweet Sentiment Dashboard  


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airline-sentiment-dashboard-project-dcjiqw474kzy2gsvozciyx.streamlit.app/)

🎯 **Live Demo:**  
👉 [https://airline-sentiment-dashboard-project-dcjiqw474kzy2gsvozciyx.streamlit.app/](https://airline-sentiment-dashboard-project-dcjiqw474kzy2gsvozciyx.streamlit.app/)

---

## 📌 Deskripsi Umum
Proyek ini merupakan tugas akhir mata kuliah **Pengembangan Aplikasi 2025**, dengan fokus pada pembuatan **dashboard interaktif berbasis data dunia nyata**.  
Dashboard ini menampilkan analisis sentimen publik terhadap maskapai penerbangan berdasarkan data tweet, lengkap dengan pembersihan data, transformasi fitur, dan visualisasi yang interaktif.

Fokus utama proyek bukan hanya tampilan visual, tetapi juga bagaimana **data dikelola, diproses, dan divisualisasikan secara efektif** untuk menghasilkan insight yang bermakna.

---

## 🎯 Tujuan
1. Mengimplementasikan pipeline manajemen data (load → clean → transform → aggregate).  
2. Mendesain visualisasi yang informatif dan interaktif menggunakan Streamlit & Plotly.  
3. Menggali insight perilaku pelanggan terhadap maskapai penerbangan dari pola tweet.  
4. Menunjukkan kolaborasi tim dalam pengembangan dashboard berbasis data nyata.

---

## 🧩 Dataset
- **Nama File:** `Tweets.csv`  
- **Sumber:** Dataset publik *US Airline Sentiment (Kaggle)*  
- **Jumlah Data:** ±14.000 tweet  
- **Fitur Utama:**
  - `tweet_id` → ID unik setiap tweet  
  - `airline_sentiment` → label sentimen (`positive`, `negative`, `neutral`)  
  - `airline` → nama maskapai  
  - `text` → isi tweet  
  - `tweet_created` → waktu tweet dibuat  
  - `latitude` & `longitude` → lokasi pengguna  

---

## ⚙️ Pipeline Data yang digunakan
### Tahapan:
| Tahap | Deskripsi | Hasil |
|-------|------------|--------|
| **Ingest** | Membaca dataset mentah (`Tweets.csv`) menggunakan `pandas` | DataFrame awal |
| **Validate** | Cek missing values, duplikat, tipe data, dan range koordinat | Laporan kualitas data |
| **Clean** | - Hapus duplikat & baris kosong<br>- Parsing waktu (`tweet_created`)<br>- Normalisasi teks (hapus URL, mention, simbol) | Data bersih & konsisten |
| **Transform** | Tambahkan fitur baru: `hour`, `is_negative`, `topic_delay` (berdasarkan keyword “delay”, “late”, dll.) | Data siap analisis |
| **Aggregate** | Hitung distribusi sentimen, tren waktu, dan proporsi keyword per sentimen | Tabel dan metrik KPI |
| **Cache** | Gunakan `@st.cache_data` untuk efisiensi load ulang | Performa stabil |

---

## 📊 Fitur Dashboard
| Fitur | Keterangan |
|-------|-------------|
| **KPI Cards** | Total tweet, % negatif, top airline dengan keluhan terbanyak |
| **Filters** | Airline, Sentiment, Hour (jam) |
| **Charts** | Bar chart, Pie chart, Line trend per jam, Word cloud |
| **Geo Map** | Sebaran lokasi tweet berdasarkan koordinat (`lat`, `lon`) |
| **Topic Analysis** | Proporsi komplain delay vs non-delay |
| **Upload CSV** | User bisa upload dataset lain untuk analisis berbeda |
| **Data Validation Box** | Menampilkan ringkasan kualitas data (missing, duplikat, dtypes) |
| **Insights Box** | Menampilkan kesimpulan dan rekomendasi otomatis |

---

## 🧠 Insight Utama (contoh)
- Tweet negatif mendominasi pada jam **18.00–21.00**, terutama maskapai **United Airlines**.  
- Sekitar **65%** tweet negatif mengandung kata terkait *delay* atau *cancel*.  
- Volume tweet positif cenderung naik di pagi hari (06.00–10.00).  

---

## 🖥️ Teknologi yang Digunakan
| Komponen | Library/Tools |
|-----------|---------------|
| Bahasa | Python 3.11 |
| Framework Dashboard | Streamlit |
| Visualisasi | Plotly, Matplotlib, WordCloud |
| Analisis Data | Pandas, Numpy |
| Lingkungan | Virtual Environment (venv) |
| Version Control | Git & GitHub |

---

## 🧪 Cara Menjalankan
```bash
# 1. Clone repository
git clone https://github.com/<username-kamu>/AirlineSentimentDashboard.git
cd AirlineSentimentDashboard

# 2. Buat environment & aktifkan
python -m venv .venv
.venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan dashboard
streamlit run app.py


