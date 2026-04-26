"""
app.py — Streamlit Application: Penilaian Makalah dengan LightRAG
Menggabungkan Tab 1 (Ingestion), Tab 2 (Evaluasi), Tab 3 (Riwayat)
"""

import nest_asyncio
nest_asyncio.apply()

import os
import asyncio
import json
import tempfile
import zipfile
import logging
import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import streamlit as st
from dotenv import load_dotenv

import tempfile

# ── Env & Config ──────────────────────────────────────────────────────────────
load_dotenv(".env", override=True)

WORKING_DIR        = os.getenv("LIGHTRAG_WORKING_DIR", "./rag_storage")
MINIO_ENDPOINT     = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY   = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY   = os.getenv("MINIO_SECRET_KEY", "password123")
BUCKET_MAKALAH     = os.getenv("BUCKET_MAKALAH", "makalah")
BUCKET_KNOWLEDGE   = os.getenv("BUCKET_KNOWLEDGE", "knowledge")
SKJ_JSON_FOLDER    = os.getenv("SKJ_JSON_FOLDER", "../data/skj_documents_json")
PG_HOST            = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT            = os.getenv("POSTGRES_PORT", "5452")
PG_DB              = os.getenv("POSTGRES_DATABASE", "lightrag")
PG_USER            = os.getenv("POSTGRES_USER", "light_postgres")
PG_PASS            = os.getenv("POSTGRES_PASSWORD", "light_postgres_root")

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("penilaian_makalah")

# ── Score Config ───────────────────────────────────────────────────────────────
SCORE_LABELS = {
    "n1_kesesuaian_judul":   "Kesesuaian Judul dengan Tema",
    "n2_kesesuaian_isi":     "Kesesuaian Isi dengan Judul & Tema",
    "n3_sistematika":        "Sistematika Penulisan",
    "n4_ketajaman_analisis": "Ketajaman Analisis",
    "n5_penggunaan_bahasa":  "Penggunaan Bahasa",
}
SCORE_WEIGHTS = {
    "n1_kesesuaian_judul":   1,
    "n2_kesesuaian_isi":     1,
    "n3_sistematika":        1,
    "n4_ketajaman_analisis": 2,
    "n5_penggunaan_bahasa":  1,
}
SCORE_KEYS = list(SCORE_LABELS.keys())

# ── Prompts ────────────────────────────────────────────────────────────────────
QUERY_KEYWORDS = """
Penilaian Penulisan Makalah
Form. 1 Penilaian Penulisan Makalah
{selected_jabatan}
Kesesuaian Judul dengan Tema
Kesesuaian Isi Makalah dengan Judul dan Tema
Sistematika Penulisan
Ketajaman Analisis
Penggunaan Bahasa dalam Penulisan Makalah
Bobot Penilaian Penulisan Makalah
Format Penulisan Makalah
Struktur Makalah (Pendahuluan, Analisis dan Sintesis, Rencana Strategis, Plan Of Action, Konklusi)
Penilaian Kompetensi Teknis/Bidang
Kompetensi Bidang
Panitia Seleksi
"""

PROMPT_KONTEKS = """
Anda adalah asisten yang bertugas mengumpulkan konteks relevan untuk penilaian makalah.

Berdasarkan jabatan '{selected_jabatan}', berikan ringkasan singkat tentang:
1. Deskripsi jabatan dan kompetensi yang diperlukan
2. Kriteria penilaian utama untuk posisi ini
3. Standar kualitas yang diharapkan dalam penulisan makalah
4. Rencana Strategis BPOM yang relevan untuk menilai kedalaman analisis dalam makalah

Berikan jawaban dalam format paragraf singkat, fokus pada poin-poin penting yang akan membantu dalam evaluasi makalah.
"""

PROMPT_PENILAIAN = """
---Role---

Anda adalah evaluator akademik sebagai Panitia Seleksi yang bertugas menilai kualitas substansi makalah secara objektif dan sistematis. Penilaian harus didasarkan hanya pada isi makalah yang tersedia, dengan mempertimbangkan konteks jabatan yang dituju.

---Goal---

Melakukan penilaian terhadap makalah berdasarkan kriteria penilaian yang telah ditentukan, memberikan skor numerik untuk setiap kriteria, serta menyusun justifikasi yang jelas dan berbasis bukti dari isi makalah.

---Konteks Jabatan---

{assessment_context}

---Instructions---

1. Baca dan pahami isi makalah secara menyeluruh.
2. Tinjau konteks jabatan di atas sebagai acuan penilaian.
3. Lakukan penilaian terhadap setiap kriteria dengan memberikan skor antara 40 sampai 100.
4. Setiap skor harus disertai justifikasi yang menjelaskan alasan pemberian skor.
5. Penilaian harus objektif, sistematis, dan berbasis isi makalah.
6. Gunakan bahasa formal dan akademik.
7. Jangan menggunakan informasi di luar isi makalah.
8. Jika informasi dalam makalah terbatas, tetap berikan skor dengan menjelaskan keterbatasan informasi tersebut.
9. Hitung nilai akhir menggunakan rumus yang telah ditentukan.
10. Output harus dalam format JSON yang valid dan tidak boleh mengandung teks tambahan di luar JSON.

---Assessment Criteria---

A. Penulisan Makalah

1. Kesesuaian judul dengan tema
Menilai kesesuaian antara judul dan tema yang dibahas dalam makalah.

2. Kesesuaian isi makalah dengan judul dan tema
Menilai kesesuaian antara isi makalah dengan judul dan tema.

3. Sistematika penulisan
Menilai keteraturan struktur penulisan dan alur pembahasan.

4. Ketajaman analisis
Menilai kedalaman pemikiran, argumentasi, dan kemampuan analisis terhadap permasalahan.

5. Penggunaan bahasa dalam penulisan makalah
Menilai kejelasan, ketepatan, dan konsistensi penggunaan bahasa.

---Scoring Rules---

- Skor minimum: 40
- Skor maksimum: 100
- Semua skor harus berupa bilangan bulat
- Ketajaman analisis memiliki bobot dua kali lipat dalam nilai akhir

---Makalah---

{makalah_text}

---Output Format---

Hasil harus dalam format JSON berikut:
{{
  "Ringkasan": "ringkasan isi makalah yang menjelaskan tentang keseluruhan makalah",

  "scores": {{
    "n1_kesesuaian_judul": 0,
    "n2_kesesuaian_isi": 0,
    "n3_sistematika": 0,
    "n4_ketajaman_analisis": 0,
    "n5_penggunaan_bahasa": 0
  }},

  "justification": {{
    "n1_kesesuaian_judul": "",
    "n2_kesesuaian_isi": "",
    "n3_sistematika": "",
    "n4_ketajaman_analisis": "",
    "n5_penggunaan_bahasa": ""
  }},

  "evidence": {{
    "n1_kesesuaian_judul": "",
    "n2_kesesuaian_isi": "",
    "n3_sistematika": "",
    "n4_ketajaman_analisis": "",
    "n5_penggunaan_bahasa": ""
  }},

  "final_score": 0
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS (dari notebook)
# ══════════════════════════════════════════════════════════════════════════════

NS     = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
VMERGE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val'


def _read_docx(filepath: str) -> str:
    with zipfile.ZipFile(filepath) as z:
        root = ET.fromstring(z.read('word/document.xml'))
    lines = []
    for child in root.find('.//w:body', NS):
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            text = ''.join(
                r.find('w:t', NS).text for r in child.findall('.//w:r', NS)
                if r.find('w:t', NS) is not None and r.find('w:t', NS).text
            ).strip()
            if text:
                lines.append(text)
        elif tag == 'tbl':
            for row in child.findall('w:tr', NS):
                cells = []
                for cell in row.findall('w:tc', NS):
                    vm = cell.find('.//w:vMerge', NS)
                    if vm is not None and vm.get(VMERGE) != 'restart':
                        continue
                    text = ''.join(
                        r.find('w:t', NS).text
                        for p in cell.findall('.//w:p', NS)
                        for r in p.findall('.//w:r', NS)
                        if r.find('w:t', NS) is not None and r.find('w:t', NS).text
                    ).strip()
                    if text:
                        cells.append(text)
                if cells:
                    lines.append(' | '.join(cells))
    return '\n'.join(lines)


def _read_pdf(filepath: str) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            return '\n'.join(
                page.extract_text() for page in pdf.pages
                if page.extract_text()
            )
    except ImportError:
        return "[pdfplumber not installed — install with: pip install pdfplumber]"


def extract_text_from_uploaded(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        if suffix == ".docx":
            return _read_docx(tmp_path)
        elif suffix == ".pdf":
            return _read_pdf(tmp_path)
        elif suffix == ".txt":
            with open(tmp_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return ""
    finally:
        os.unlink(tmp_path)


def read_folder(folder_path: str) -> list[dict]:
    results = []
    for path in Path(folder_path).rglob('*'):
        suffix = path.suffix.lower()
        if suffix == '.docx':
            text = _read_docx(str(path))
        elif suffix == '.pdf':
            text = _read_pdf(str(path))
        else:
            continue
        results.append({'filename': path.name, 'text': text})
    return results


def compute_final_score(scores: dict) -> float:
    total_weight = sum(SCORE_WEIGHTS.values())
    weighted_sum = sum(scores.get(k, 0) * w for k, w in SCORE_WEIGHTS.items())
    return round(weighted_sum / total_weight, 1)


def parse_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def load_all_skj() -> dict:
    skj_dict = {}
    folder = Path(SKJ_JSON_FOLDER)
    if not folder.exists():
        return skj_dict
    for json_file in folder.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "profil_jabatan" in data:
                jabatan = data["profil_jabatan"].get("nama_jabatan", json_file.stem)
            elif "nama_jabatan" in data:
                jabatan = data["nama_jabatan"]
            else:
                jabatan = json_file.stem
            skj_dict[jabatan] = data
        except Exception as e:
            log.warning(f"Gagal load {json_file.name}: {e}")
    return skj_dict


# ══════════════════════════════════════════════════════════════════════════════
# RAG ASYNC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_async(coro):
    """Run async coroutine safely inside Streamlit (nest_asyncio applied)."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


async def initialize_rag():
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
        return await openai_complete_if_cache(
            os.getenv("LLM_MODEL"),
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=os.getenv("LLM_BINDING_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST"),
            **kwargs,
        )

    embedding_dim  = int(os.getenv("EMBEDDING_DIM", 1536))
    token_limit    = int(os.getenv("EMBEDDING_TOKEN_LIMIT", 8192))
    model_name     = os.getenv("EMBEDDING_MODEL")

    async def raw_embedding_func(texts):
        return await openai_embed.func(
            texts,
            api_key=os.getenv("LLM_BINDING_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST"),
            model=model_name,
        )

    embedding_func = wrap_embedding_func_with_attrs(
        embedding_dim=embedding_dim,
        max_token_size=token_limit,
        model_name=model_name,
    )(raw_embedding_func)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="Neo4JStorage",
    )
    await rag.initialize_storages()
    return rag


async def retrieve_assessment_context(rag, selected_jabatan: str, query_mode: str) -> str:
    from lightrag import QueryParam
    try:
        context_response = await rag.aquery(
            query=QUERY_KEYWORDS.format(selected_jabatan=selected_jabatan),
            param=QueryParam(
                mode=query_mode,
                user_prompt=PROMPT_KONTEKS.format(selected_jabatan=selected_jabatan),
                only_need_context=True,
            ),
        )
        return context_response if context_response else "Konteks jabatan tidak ditemukan dalam knowledge base."
    except Exception as e:
        return f"Error retrieving context: {str(e)}"


async def evaluate_paper_with_context(rag, makalah_text: str, assessment_context: str, query_mode: str) -> dict:
    from lightrag import QueryParam
    evaluation_prompt = PROMPT_PENILAIAN.format(
        assessment_context=assessment_context,
        makalah_text=makalah_text
    )
    eval_response = await rag.aquery(
        query="Penilaian Makalah: Kesesuaian Isi, Sistematika, Ketajaman Analisis, Penggunaan Bahasa",
        param=QueryParam(
            mode=query_mode,
            user_prompt=evaluation_prompt,
        ),
    )
    return parse_response(eval_response)


# ══════════════════════════════════════════════════════════════════════════════
# MINIO HELPER
# ══════════════════════════════════════════════════════════════════════════════

def get_minio_client():
    try:
        import boto3
        from botocore.client import Config
        return boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
        )
    except Exception as e:
        log.error(f"MinIO connection failed: {e}")
        return None


def ensure_bucket(client, bucket_name: str):
    try:
        existing = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
        if bucket_name not in existing:
            client.create_bucket(Bucket=bucket_name)
    except Exception as e:
        log.warning(f"Could not ensure bucket '{bucket_name}': {e}")


def list_minio_files(bucket: str) -> list[str]:
    client = get_minio_client()
    if not client:
        return []
    try:
        resp = client.list_objects_v2(Bucket=bucket)
        return [obj["Key"] for obj in resp.get("Contents", [])]
    except Exception:
        return []


def upload_to_minio(bucket: str, key: str, data: bytes):
    client = get_minio_client()
    if not client:
        return False
    try:
        ensure_bucket(client, bucket)
        client.put_object(Bucket=bucket, Key=key, Body=data)
        return True
    except Exception as e:
        log.error(f"MinIO upload failed: {e}")
        return False


def download_from_minio(bucket: str, key: str) -> bytes | None:
    client = get_minio_client()
    if not client:
        return None
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except Exception as e:
        log.error(f"MinIO download failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# POSTGRESQL HELPER
# ══════════════════════════════════════════════════════════════════════════════

def get_pg_conn():
    try:
        import psycopg2
        return psycopg2.connect(
            host=PG_HOST, port=PG_PORT,
            dbname=PG_DB, user=PG_USER, password=PG_PASS,
        )
    except Exception as e:
        log.error(f"PostgreSQL connection failed: {e}")
        return None


def init_db():
    conn = get_pg_conn()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id SERIAL PRIMARY KEY,
                    paper_filename TEXT,
                    jabatan TEXT,
                    scores JSONB,
                    final_score FLOAT,
                    justification JSONB,
                    evidence JSONB,
                    ringkasan TEXT,
                    query_mode TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_log (
                    id SERIAL PRIMARY KEY,
                    filename TEXT,
                    status TEXT,
                    error_message TEXT,
                    ingested_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        conn.commit()
        return True
    except Exception as e:
        log.error(f"DB init failed: {e}")
        return False
    finally:
        conn.close()


def save_evaluation(paper_filename, jabatan, result_dict, query_mode):
    conn = get_pg_conn()
    if not conn:
        return False
    try:
        scores        = result_dict.get("scores", {})
        final_score   = result_dict.get("final_score", compute_final_score(scores))
        justification = result_dict.get("justification", {})
        evidence      = result_dict.get("evidence", {})
        ringkasan     = result_dict.get("Ringkasan", "")
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO evaluation_results
                  (paper_filename, jabatan, scores, final_score, justification, evidence, ringkasan, query_mode)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                paper_filename, jabatan,
                json.dumps(scores), final_score,
                json.dumps(justification), json.dumps(evidence),
                ringkasan, query_mode,
            ))
        conn.commit()
        return True
    except Exception as e:
        log.error(f"Save evaluation failed: {e}")
        return False
    finally:
        conn.close()


def get_evaluation_history(jabatan_filter=None, limit=100):
    conn = get_pg_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            if jabatan_filter and jabatan_filter != "Semua":
                cur.execute("""
                    SELECT id, paper_filename, jabatan, scores, final_score,
                           justification, evidence, ringkasan, query_mode, created_at
                    FROM evaluation_results
                    WHERE jabatan = %s
                    ORDER BY created_at DESC LIMIT %s
                """, (jabatan_filter, limit))
            else:
                cur.execute("""
                    SELECT id, paper_filename, jabatan, scores, final_score,
                           justification, evidence, ringkasan, query_mode, created_at
                    FROM evaluation_results
                    ORDER BY created_at DESC LIMIT %s
                """, (limit,))
            rows = cur.fetchall()
            cols = ["id","paper_filename","jabatan","scores","final_score",
                    "justification","evidence","ringkasan","query_mode","created_at"]
            return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        log.error(f"Get history failed: {e}")
        return []
    finally:
        conn.close()


def log_ingestion(filename, status, error_message=None):
    conn = get_pg_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ingestion_log (filename, status, error_message)
                VALUES (%s, %s, %s)
            """, (filename, status, error_message))
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
 
def init_session():
    defaults = {
        "rag": None,
        "rag_initialized": False,
        "rag_error": None,
        "skj_data": None,
        "ingestion_progress": {},
        "eval_result": None,
        "eval_context": None,
        "eval_paper_text": None,
        "eval_paper_name": None,
        "eval_jabatan": None,
        "db_ok": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
 
 
# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
 
def score_color(score: float) -> str:
    if score >= 85:   return "#22c55e"
    elif score >= 70: return "#f59e0b"
    elif score >= 55: return "#f97316"
    else:             return "#ef4444"
 
 
def render_score_card(result: dict):
    scores      = result.get("scores", {})
    final       = result.get("final_score", compute_final_score(scores))
    justif      = result.get("justification", {})
    evidence    = result.get("evidence", {})
    ringkasan   = result.get("Ringkasan", "")
 
    color = score_color(final)
 
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}11);
        border: 2px solid {color};
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 24px;
    ">
        <div style="font-size: 14px; color: #94a3b8; font-weight: 600; letter-spacing: 2px; text-transform: uppercase;">NILAI AKHIR</div>
        <div style="font-size: 64px; font-weight: 800; color: {color}; line-height: 1.1;">{final}</div>
        <div style="font-size: 12px; color: #64748b;">dari 100</div>
    </div>
    """, unsafe_allow_html=True)
 
    if ringkasan:
        with st.expander("📄 Ringkasan Makalah"):
            st.write(ringkasan)
 
    st.markdown("**Rincian Skor per Kriteria**")
    for key, label in SCORE_LABELS.items():
        score = scores.get(key, 0)
        weight = SCORE_WEIGHTS[key]
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"{'⭐ ' if weight > 1 else ''}{label}")
            st.progress(max(0, min(100, int(score))) / 100)
        with col2:
            st.metric("Skor", score)
        with col3:
            st.metric("Bobot", f"×{weight}")
 
        j = justif.get(key, "")
        e = evidence.get(key, "")
        if j or e:
            with st.expander(f"Detail: {label}"):
                if j:
                    st.markdown(f"**Justifikasi:** {j}")
                if e:
                    st.markdown(f"**Bukti:** {e}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════
 
st.set_page_config(
    page_title="Penilaian Makalah — LightRAG",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
 
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    padding: 4px 0;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    background: #1e293b;
    border-radius: 10px;
    color: #94a3b8;
    font-weight: 600;
    font-size: 14px;
    padding: 0 20px;
    border: 1px solid #334155;
}
.stTabs [aria-selected="true"] {
    background: #3b82f6 !important;
    color: white !important;
    border-color: #3b82f6 !important;
}
.metric-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# INIT SESSION STATE — harus sebelum sidebar & tabs
# ══════════════════════════════════════════════════════════════════════════════
 
init_session()
 
# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
 
with st.sidebar:
    st.markdown("## 📋 Penilaian Makalah")
    st.markdown("*Powered by LightRAG*")
    st.divider()
 
    # RAG Status
    st.markdown("### ⚙️ Status Sistem")
 
    rag_status = st.empty()
    if st.session_state.rag_initialized:
        rag_status.success("✅ RAG Aktif")
    elif st.session_state.rag_error:
        rag_status.error(f"❌ RAG Error")
        st.caption(st.session_state.rag_error[:100])
    else:
        rag_status.warning("⏳ RAG belum diinisialisasi")
 
    if st.button("🚀 Inisialisasi RAG", use_container_width=True, disabled=st.session_state.rag_initialized):
        with st.spinner("Menginisialisasi LightRAG..."):
            try:
                rag = run_async(initialize_rag())
                st.session_state.rag = rag
                st.session_state.rag_initialized = True
                st.session_state.rag_error = None
                rag_status.success("✅ RAG Aktif")
                st.success("RAG berhasil diinisialisasi!")
            except Exception as e:
                st.session_state.rag_error = str(e)
                st.session_state.rag_initialized = False
                rag_status.error("❌ RAG Error")
                st.error(f"Gagal: {e}")
 
    st.divider()
 
    # DB Status
    st.markdown("### 🗄️ Database")
    if not st.session_state.db_ok:
        if st.button("Inisialisasi DB", use_container_width=True):
            ok = init_db()
            st.session_state.db_ok = ok
            if ok:
                st.success("Database siap!")
            else:
                st.error("Gagal connect ke PostgreSQL")
    else:
        st.success("✅ PostgreSQL Terhubung")
 
    st.divider()
 
    # Query Mode
    st.markdown("### 🔍 Query Mode")
    query_mode = st.selectbox(
        "Mode RAG",
        ["mix", "hybrid", "local", "global", "naive"],
        index=0,
        help="Mode query LightRAG yang akan digunakan",
        label_visibility="collapsed",
    )
    st.session_state.query_mode = query_mode
 
    st.divider()
    st.caption(f"LightRAG Working Dir: `{WORKING_DIR}`")
    st.caption(f"MinIO: `{MINIO_ENDPOINT}`")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
 
tab1, tab2, tab3 = st.tabs([
    "📥 Tab 1 — Ingestion Dokumen",
    "📝 Tab 2 — Evaluasi Makalah",
    "📊 Tab 3 — Riwayat Evaluasi",
])
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DOCUMENT INGESTION
# ══════════════════════════════════════════════════════════════════════════════
 
with tab1:
    st.markdown("## 📥 Ingestion Dokumen ke Knowledge Base")
    st.markdown("Upload dokumen SKJ, peraturan, atau renstra untuk dimasukkan ke LightRAG.")
 
    col_upload, col_info = st.columns([2, 1])
 
    with col_upload:
        uploaded_docs = st.file_uploader(
            "Upload dokumen (PDF / DOCX / TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="doc_uploader",
        )
 
        doc_type = st.selectbox(
            "Tipe Dokumen",
            ["SKJ", "Peraturan Pemerintah", "Renstra BPOM", "Lainnya"],
        )
 
        also_upload_minio = st.checkbox("Upload juga ke MinIO (bucket: knowledge)", value=True)
 
        ingest_btn = st.button(
            "⚡ Mulai Ingestion",
            disabled=(not st.session_state.rag_initialized or not uploaded_docs),
            type="primary",
            use_container_width=True,
        )
 
    with col_info:
        st.markdown("**📌 Catatan:**")
        st.info("""
- Format: PDF, DOCX, TXT
- Bisa upload banyak file sekaligus
- Pastikan RAG sudah diinisialisasi
- Proses bisa memakan waktu beberapa menit
        """)
 
        # MinIO file list
        st.markdown("**📦 File di MinIO (knowledge):**")
        minio_files = list_minio_files(BUCKET_KNOWLEDGE)
        if minio_files:
            for f in minio_files[:10]:
                st.markdown(f"  - `{f}`")
            if len(minio_files) > 10:
                st.caption(f"... dan {len(minio_files)-10} file lainnya")
        else:
            st.caption("(belum ada file)")
 
    if ingest_btn and uploaded_docs:
        rag = st.session_state.rag
        progress_bar = st.progress(0, text="Memulai ingestion...")
        results_log = []
 
        for i, up_file in enumerate(uploaded_docs):
            fname = up_file.name
            progress_bar.progress((i) / len(uploaded_docs), text=f"Membaca {fname}...")
 
            try:
                # Read text
                up_file.seek(0)
                text = extract_text_from_uploaded(up_file)
                if not text.strip():
                    results_log.append({"file": fname, "status": "⚠️ Skip", "info": "Teks kosong"})
                    log_ingestion(fname, "skip", "Teks kosong")
                    continue
 
                # Upload to MinIO if requested
                if also_upload_minio:
                    up_file.seek(0)
                    ok = upload_to_minio(BUCKET_KNOWLEDGE, fname, up_file.read())
                    if not ok:
                        results_log.append({"file": fname, "status": "⚠️ MinIO gagal", "info": "Teks tetap diingest"})
 
                # Ingest to RAG
                progress_bar.progress((i + 0.5) / len(uploaded_docs), text=f"Mengingest {fname} ke RAG...")
                run_async(rag.ainsert([text], file_paths=[fname]))
 
                results_log.append({"file": fname, "status": "✅ Berhasil", "info": f"{len(text)} karakter"})
                log_ingestion(fname, "success")
 
            except Exception as e:
                results_log.append({"file": fname, "status": "❌ Error", "info": str(e)[:80]})
                log_ingestion(fname, "error", str(e))
 
        progress_bar.progress(1.0, text="Selesai!")
        st.success(f"Ingestion selesai! {sum(1 for r in results_log if '✅' in r['status'])}/{len(results_log)} berhasil.")
 
        st.markdown("**Log Ingestion:**")
        for r in results_log:
            st.markdown(f"{r['status']} `{r['file']}` — {r['info']}")
 
    st.divider()
 
    # Manual folder ingestion
    with st.expander("📂 Ingestion dari Folder (path di server)"):
        folder_path_input = st.text_input("Path folder dokumen", placeholder="/app/data/skj_documents")
        folder_ingest_btn = st.button(
            "Ingest Folder",
            disabled=not st.session_state.rag_initialized,
        )
        if folder_ingest_btn and folder_path_input:
            if not os.path.exists(folder_path_input):
                st.error(f"Folder tidak ditemukan: `{folder_path_input}`")
            else:
                with st.spinner("Membaca dan mengingest folder..."):
                    docs = read_folder(folder_path_input)
                    if not docs:
                        st.warning("Tidak ada file PDF/DOCX di folder tersebut.")
                    else:
                        rag = st.session_state.rag
                        run_async(rag.ainsert(
                            [d["text"] for d in docs],
                            file_paths=[d["filename"] for d in docs],
                        ))
                        st.success(f"✅ {len(docs)} dokumen berhasil diingest dari `{folder_path_input}`")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: EVALUASI MAKALAH
# ══════════════════════════════════════════════════════════════════════════════
 
with tab2:
    st.markdown("## 📝 Evaluasi Makalah")
 
    if not st.session_state.rag_initialized:
        st.warning("⚠️ Inisialisasi RAG terlebih dahulu di sidebar kiri sebelum melakukan evaluasi.")
        st.stop()
 
    # Load SKJ
    if st.session_state.skj_data is None:
        st.session_state.skj_data = load_all_skj()
 
    skj_data = st.session_state.skj_data
    jabatan_list = list(skj_data.keys()) if skj_data else []
 
    col_left, col_right = st.columns([1, 1])
 
    with col_left:
        st.markdown("### 1️⃣ Pilih Jabatan")
 
        if jabatan_list:
            selected_jabatan = st.selectbox(
                "Jabatan yang dilamar",
                jabatan_list,
                key="jabatan_selector",
            )
        else:
            selected_jabatan = st.text_input(
                "Jabatan yang dilamar (input manual — file SKJ JSON tidak ditemukan)",
                placeholder="contoh: Sekretaris Utama",
            )
 
        st.markdown("### 2️⃣ Upload Makalah")
 
        source_option = st.radio(
            "Sumber makalah",
            ["Upload file baru", "Pilih dari MinIO"],
            horizontal=True,
        )
 
        paper_text = ""
        paper_name = ""
 
        if source_option == "Upload file baru":
            uploaded_paper = st.file_uploader(
                "Upload makalah (PDF / DOCX / TXT)",
                type=["pdf", "docx", "txt"],
                key="paper_uploader",
            )
            if uploaded_paper:
                paper_name = uploaded_paper.name
                with st.spinner("Membaca makalah..."):
                    uploaded_paper.seek(0)
                    paper_text = extract_text_from_uploaded(uploaded_paper)
                    # Also upload to MinIO
                    uploaded_paper.seek(0)
                    upload_to_minio(BUCKET_MAKALAH, paper_name, uploaded_paper.read())
 
                if paper_text:
                    st.success(f"✅ Makalah dibaca: {len(paper_text)} karakter")
                else:
                    st.error("Gagal membaca teks dari file.")
 
        else:
            minio_papers = list_minio_files(BUCKET_MAKALAH)
            if minio_papers:
                selected_paper_key = st.selectbox("Pilih makalah dari MinIO", minio_papers)
                if st.button("📥 Load dari MinIO"):
                    data = download_from_minio(BUCKET_MAKALAH, selected_paper_key)
                    if data:
                        suffix = Path(selected_paper_key).suffix.lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(data)
                            tmp_path = tmp.name
                        if suffix == ".docx":
                            paper_text = _read_docx(tmp_path)
                        elif suffix == ".pdf":
                            paper_text = _read_pdf(tmp_path)
                        elif suffix == ".txt":
                            paper_text = open(tmp_path, "r", encoding="utf-8").read()
                        os.unlink(tmp_path)
                        paper_name = selected_paper_key
                        st.session_state.eval_paper_text = paper_text
                        st.session_state.eval_paper_name = paper_name
                        st.success(f"✅ Loaded: {len(paper_text)} karakter")
                    else:
                        st.error("Gagal download dari MinIO.")
            else:
                st.info("Belum ada makalah di MinIO bucket 'makalah'.")
 
        # Use previously loaded paper text if available
        if not paper_text and st.session_state.eval_paper_text:
            paper_text = st.session_state.eval_paper_text
            paper_name = st.session_state.eval_paper_name or ""
 
        if paper_text:
            with st.expander("👁️ Preview Teks Makalah (500 karakter pertama)"):
                st.text(paper_text[:500] + ("..." if len(paper_text) > 500 else ""))
 
    with col_right:
        st.markdown("### 3️⃣ Jalankan Evaluasi")
 
        qmode = st.session_state.get("query_mode", "mix")
        st.caption(f"Query mode aktif: **{qmode}**")
 
        can_eval = bool(selected_jabatan and paper_text and st.session_state.rag_initialized)
 
        # Stage 1: Retrieve context
        ctx_btn = st.button(
            "🔍 Tahap 1: Ambil Konteks Jabatan",
            disabled=not can_eval,
            use_container_width=True,
        )
 
        if ctx_btn:
            with st.spinner(f"Mengambil konteks untuk '{selected_jabatan}'..."):
                try:
                    ctx = run_async(retrieve_assessment_context(
                        st.session_state.rag, selected_jabatan, qmode
                    ))
                    st.session_state.eval_context = ctx
                    st.session_state.eval_jabatan = selected_jabatan
                    st.success("✅ Konteks berhasil diambil!")
                except Exception as e:
                    st.error(f"Gagal ambil konteks: {e}")
 
        if st.session_state.eval_context:
            with st.expander("📋 Lihat Konteks Jabatan"):
                st.write(st.session_state.eval_context)
 
        st.divider()
 
        # Stage 2: Evaluate
        eval_btn = st.button(
            "⚡ Tahap 2: Evaluasi Makalah",
            disabled=(not can_eval or not st.session_state.eval_context),
            type="primary",
            use_container_width=True,
        )
 
        if eval_btn:
            if not paper_text:
                st.error("Belum ada teks makalah yang dimuat.")
            else:
                with st.spinner("Mengevaluasi makalah... (ini bisa memakan waktu 1-2 menit)"):
                    try:
                        result = run_async(evaluate_paper_with_context(
                            st.session_state.rag,
                            paper_text,
                            st.session_state.eval_context,
                            qmode,
                        ))
                        # Recompute final score to ensure accuracy
                        result["final_score"] = compute_final_score(result.get("scores", {}))
                        st.session_state.eval_result = result
                        st.session_state.eval_paper_name = paper_name
                        st.session_state.eval_jabatan = selected_jabatan
                        st.success("✅ Evaluasi selesai!")
                    except Exception as e:
                        st.error(f"Gagal evaluasi: {e}")
 
        # Quick eval (single stage)
        st.divider()
        st.caption("Atau jalankan langsung tanpa ambil konteks dulu:")
        quick_btn = st.button(
            "🚀 Evaluasi Langsung (tanpa konteks RAG)",
            disabled=not can_eval,
            use_container_width=True,
        )
        if quick_btn:
            with st.spinner("Mengevaluasi..."):
                try:
                    quick_ctx = f"Jabatan: {selected_jabatan}. Evaluasi makalah berdasarkan kriteria penulisan yang berlaku."
                    result = run_async(evaluate_paper_with_context(
                        st.session_state.rag, paper_text, quick_ctx, qmode
                    ))
                    result["final_score"] = compute_final_score(result.get("scores", {}))
                    st.session_state.eval_result = result
                    st.session_state.eval_paper_name = paper_name
                    st.session_state.eval_jabatan = selected_jabatan
                    st.success("✅ Evaluasi selesai!")
                except Exception as e:
                    st.error(f"Gagal evaluasi: {e}")
 
    # Show result
    if st.session_state.eval_result:
        st.divider()
        st.markdown(f"### 🎯 Hasil Evaluasi — {st.session_state.eval_paper_name or 'Makalah'}")
        st.caption(f"Jabatan: **{st.session_state.eval_jabatan}** | Mode: **{qmode}**")
 
        render_score_card(st.session_state.eval_result)
 
        col_save, col_download = st.columns(2)
 
        with col_save:
            save_btn = st.button("💾 Simpan ke Database", use_container_width=True)
            if save_btn:
                if not st.session_state.db_ok:
                    st.warning("Database belum diinisialisasi. Klik 'Inisialisasi DB' di sidebar.")
                else:
                    ok = save_evaluation(
                        st.session_state.eval_paper_name or "unknown",
                        st.session_state.eval_jabatan,
                        st.session_state.eval_result,
                        qmode,
                    )
                    if ok:
                        # Also save JSON to MinIO
                        result_json = json.dumps(st.session_state.eval_result, ensure_ascii=False, indent=2)
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        minio_key = f"eval_{ts}_{st.session_state.eval_paper_name or 'result'}.json"
                        upload_to_minio(BUCKET_MAKALAH, minio_key, result_json.encode("utf-8"))
                        st.success("✅ Hasil evaluasi disimpan ke database dan MinIO!")
                    else:
                        st.error("Gagal menyimpan ke database.")
 
        with col_download:
            result_json = json.dumps(st.session_state.eval_result, ensure_ascii=False, indent=2)
            st.download_button(
                "⬇️ Download Hasil (JSON)",
                data=result_json,
                file_name=f"evaluasi_{st.session_state.eval_paper_name or 'hasil'}.json",
                mime="application/json",
                use_container_width=True,
            )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: RIWAYAT EVALUASI
# ══════════════════════════════════════════════════════════════════════════════
 
with tab3:
    st.markdown("## 📊 Riwayat Evaluasi")
 
    if not st.session_state.db_ok:
        st.warning("⚠️ Inisialisasi database terlebih dahulu di sidebar untuk melihat riwayat.")
    else:
        col_filter1, col_filter2, col_filter3 = st.columns(3)
 
        with col_filter1:
            skj_data = st.session_state.skj_data or {}
            jabatan_options = ["Semua"] + list(skj_data.keys())
            filter_jabatan = st.selectbox("Filter Jabatan", jabatan_options)
 
        with col_filter2:
            filter_limit = st.number_input("Jumlah data", min_value=10, max_value=500, value=50, step=10)
 
        with col_filter3:
            filter_score_min = st.number_input("Nilai minimum", min_value=0, max_value=100, value=0)
 
        refresh_btn = st.button("🔄 Muat Riwayat", use_container_width=False)
 
        if refresh_btn or "history_data" not in st.session_state:
            history = get_evaluation_history(
                jabatan_filter=filter_jabatan if filter_jabatan != "Semua" else None,
                limit=int(filter_limit),
            )
            # Filter by min score
            if filter_score_min > 0:
                history = [h for h in history if (h.get("final_score") or 0) >= filter_score_min]
            st.session_state.history_data = history
 
        history = st.session_state.get("history_data", [])
 
        if not history:
            st.info("Belum ada riwayat evaluasi. Lakukan evaluasi di Tab 2 dan simpan hasilnya.")
        else:
            st.markdown(f"**{len(history)} hasil ditemukan**")
 
            # Summary metrics
            scores_list = [h.get("final_score", 0) for h in history if h.get("final_score")]
            if scores_list:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Evaluasi", len(history))
                c2.metric("Rata-rata Nilai", f"{sum(scores_list)/len(scores_list):.1f}")
                c3.metric("Nilai Tertinggi", f"{max(scores_list):.1f}")
                c4.metric("Nilai Terendah", f"{min(scores_list):.1f}")
 
            st.divider()
 
            # Table view
            import pandas as pd
            table_data = []
            for h in history:
                table_data.append({
                    "ID": h["id"],
                    "File Makalah": h["paper_filename"],
                    "Jabatan": h["jabatan"],
                    "Nilai Akhir": h["final_score"],
                    "Mode": h["query_mode"],
                    "Tanggal": str(h["created_at"])[:16] if h["created_at"] else "-",
                })
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, height=300)
 
            st.divider()
            st.markdown("**Detail Evaluasi**")
 
            for h in history:
                final = h.get("final_score", 0)
                color = score_color(final)
                with st.expander(
                    f"[{final}] {h['paper_filename']} — {h['jabatan']} ({str(h['created_at'])[:10]})"
                ):
                    scores = h.get("scores") or {}
                    if isinstance(scores, str):
                        try:
                            scores = json.loads(scores)
                        except Exception:
                            scores = {}
 
                    justif = h.get("justification") or {}
                    if isinstance(justif, str):
                        try:
                            justif = json.loads(justif)
                        except Exception:
                            justif = {}
 
                    ringkasan = h.get("ringkasan", "")
                    if ringkasan:
                        st.markdown(f"**Ringkasan:** {ringkasan[:300]}...")
 
                    st.markdown("**Skor per Kriteria:**")
                    for key, label in SCORE_LABELS.items():
                        score = scores.get(key, "-")
                        j = justif.get(key, "")
                        st.markdown(f"- **{label}**: `{score}` {'⭐' if SCORE_WEIGHTS.get(key,1)>1 else ''}")
                        if j:
                            st.caption(f"  {j[:150]}")
 
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        result_export = {
                            "paper_filename": h["paper_filename"],
                            "jabatan": h["jabatan"],
                            "final_score": h["final_score"],
                            "scores": scores,
                            "justification": justif,
                            "evidence": h.get("evidence") or {},
                            "ringkasan": h.get("ringkasan", ""),
                            "created_at": str(h["created_at"]),
                        }
                        st.download_button(
                            "⬇️ Download JSON",
                            data=json.dumps(result_export, ensure_ascii=False, indent=2),
                            file_name=f"eval_{h['id']}_{h['paper_filename']}.json",
                            mime="application/json",
                            key=f"dl_{h['id']}",
                        )
