# 📋 Penilaian Makalah BPOM (LightRAG)

Aplikasi Penilaian Makalah otomatis berbasis kecerdasan buatan (AI) menggunakan arsitektur **LightRAG**. Proyek ini dikembangkan untuk memfasilitasi evaluasi makalah secara objektif dan sistematis, berdasarkan kriteria penilaian standar, konteks jabatan (SKJ), serta standar kualitas yang diharapkan dalam lingkungan BPOM.

Aplikasi ini dibalut dengan antarmuka **Streamlit** yang interaktif, memungkinkan proses pengelolaan _knowledge base_ hingga _assessment_ (penilaian makalah) secara end-to-end.

---

## ✨ Fitur Utama

- **Penilaian Makalah Otomatis:** Memanfaatkan arsitektur LightRAG untuk memberikan evaluasi objektif dan berbasis bukti.
- **Kriteria Terstruktur:** Menilai makalah berdasarkan: Kesesuaian Judul, Kesesuaian Isi, Sistematika Penulisan, Ketajaman Analisis, dan Penggunaan Bahasa.
- **Manajemen Konteks:** Memuat Standar Kompetensi Jabatan (SKJ) untuk menyesuaikan standar penilaian jabatan terkait.
- **Dukungan Format File:** Mampu membaca ekstensi `.docx`, `.pdf`, dan `.txt`.
- **Dashboard Riwayat Penilaian:** Hasil evaluasi disimpan dan dapat difilter dengan mudah di tab riwayat.

---

## 🛠️ Tech Stack & Arsitektur

- **Frontend:** [Streamlit](https://streamlit.io/)
- **RAG Framework:** [LightRAG](https://github.com/HKUDS/LightRAG) (lightrag-hku)
- **Database Relasional & Vector:** [PostgreSQL](https://www.postgresql.org/) dengan ekstensi [pgvector](https://github.com/pgvector/pgvector)
- **Graph Database:** [Neo4j](https://neo4j.com/)
- **Object Storage:** [MinIO](https://min.io/)
- **Package Manager:** `uv` (Fast Python Package Installer & Resolver)

---

## 🚀 Prasyarat Sistem

- **Python >= 3.10**
- **Docker & Docker Compose**
- **uv** (Package manager yang sangat direkomendasikan)

---

## ⚙️ Panduan Instalasi dan Setup

### 1. Clone Repositori

```bash
git clone <repository_url>
cd py310_lightRAG
```

### 2. Jalankan Layanan Infrastruktur (Docker)

Aplikasi membutuhkan MinIO, Neo4j, dan PostgreSQL yang telah dikonfigurasi melalui `docker-compose.yml`.

```bash
docker-compose up -d
```
_Perintah ini akan mengunduh image dan menjalankan kontainer database di background._

### 3. Setup Lingkungan Python dengan `uv sync`

Proyek ini menggunakan `uv` untuk memastikan instalasi dependensi berjalan dengan sangat cepat dan sinkron dengan `uv.lock`.

Cukup jalankan perintah berikut di dalam direktori proyek:

```bash
uv sync
```

**Apa yang dilakukan `uv sync`?**
- Secara otomatis membuat virtual environment (v-env) jika belum ada.
- Membaca file `pyproject.toml` dan `uv.lock`.
- Mengunduh dan menginstal versi library yang sama persis seperti yang direkam di `uv.lock`, sehingga mencegah terjadinya masalah inkompatibilitas (dependency hell).

Setelah sinkronisasi selesai, Anda bisa mengaktifkan virtual environment (opsional jika menggunakan `uv run`):
- **Windows:** `.venv\Scripts\activate`
- **Mac/Linux:** `source .venv/bin/activate`

### 4. Konfigurasi Environment Variables

Buat file bernama `.env` di direktori utama (sejajar dengan file `docker-compose.yml`) dengan isi berikut:

```env
# Konfigurasi LLM & Embedding (Sesuaikan dengan Provider Anda - OpenAI / Azure / Local)
LLM_MODEL="gpt-4o-mini"
LLM_BINDING_API_KEY="sk-..."
LLM_BINDING_HOST="https://api.openai.com/v1"

EMBEDDING_MODEL="text-embedding-3-small"
EMBEDDING_DIM=1536
EMBEDDING_TOKEN_LIMIT=8192

# Konfigurasi MinIO
MINIO_ENDPOINT="http://localhost:9000"
MINIO_ACCESS_KEY="admin"
MINIO_SECRET_KEY="password123"

# Konfigurasi PostgreSQL
POSTGRES_HOST="localhost"
POSTGRES_PORT="5452"
POSTGRES_DATABASE="lightrag"
POSTGRES_USER="light_postgres"
POSTGRES_PASSWORD="light_postgres_root"

# Konfigurasi Neo4j
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="lightrag_neo4j_root"

# Storage Direktori untuk LightRAG
LIGHTRAG_WORKING_DIR="./rag_storage"
```

---

## 💻 Menjalankan Aplikasi

Anda dapat menggunakan `uv run` agar otomatis dieksekusi di dalam virtual environment proyek Anda:

```bash
uv run streamlit run penilaian_makalah/notebooks/streamlit.py
```

Aplikasi akan terbuka otomatis di web browser pada alamat `http://localhost:8501`.

---

## 📂 Struktur Repositori

```text
py310_lightRAG/
├── .streamlit/                 # Pengaturan tema dan server UI Streamlit
├── penilaian_makalah/          # Modul dan aplikasi
│   ├── notebooks/
│   │   └── streamlit.py        # Entry point utama aplikasi UI
│   └── data/
│       └── skj_documents_json/ # Data JSON Standar Kompetensi Jabatan
├── storage/                    # Tempat penyimpanan data persisten Docker
├── docker-compose.yml          # Setup layanan backend
├── pyproject.toml              # Konfigurasi proyek Python
├── uv.lock                     # Kunci versi dependensi package
└── README.md                   # Panduan proyek
```
