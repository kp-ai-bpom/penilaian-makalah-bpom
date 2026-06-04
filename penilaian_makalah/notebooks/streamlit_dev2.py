import nest_asyncio
nest_asyncio.apply()

import os
import sys
import asyncio
import json
import zipfile
import tempfile
import logging
import datetime
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("penilaian_makalah")

# ── Constants ─────────────────────────────────────────────────────────────────
WORKING_DIR      = os.getenv("LIGHTRAG_WORKING_DIR", "/rag_storage")
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
BUCKET_MAKALAH   = os.getenv("BUCKET_MAKALAH", "makalah")
BUCKET_KNOWLEDGE = os.getenv("BUCKET_KNOWLEDGE", "knowledge")
BUCKET_RIWAYAT   = os.getenv("BUCKET_RIWAYAT", "riwayat-penilaian-makalah")
BUCKET_TEMA      = os.getenv("BUCKET_TEMA", "ketentuan-penulisan-makalah")
SKJ_JSON_FOLDER  = os.getenv("SKJ_JSON_FOLDER", os.path.join(os.path.dirname(__file__), "../data/skj_documents_json"))
PG_HOST  = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT  = os.getenv("POSTGRES_PORT", "5432")
PG_DB    = os.getenv("POSTGRES_DATABASE", "lightrag")
PG_USER  = os.getenv("POSTGRES_USER", "postgres")
PG_PASS  = os.getenv("POSTGRES_PASSWORD", "")

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

# ── Uncertainty-Aware Evaluation Constants ────────────────────────────────────
M_SAMPLES           = 7              # Number of evaluation samples (default)
TEMPERATURE         = 1.0            # Fixed temperature for uncertainty sampling
NSV_THRESHOLD       = 0.1            # Threshold for "acceptable uncertainty"
SAMPLE_RETRY_LIMIT  = 2              # Max retries for failed samples
MIN_SCORE           = 40
MAX_SCORE           = 100
SCORE_RANGE         = MAX_SCORE - MIN_SCORE

# Criteria mapping
CRITERIA_KEYS = ["n1_kesesuaian_judul", "n2_kesesuaian_isi", "n3_sistematika", 
                 "n4_ketajaman_analisis", "n5_penggunaan_bahasa"]
CRITERIA_SHORT = ["n1", "n2", "n3", "n4", "n5"]
CRITERIA_LABELS = {
    "n1": "Kesesuaian Judul",
    "n2": "Kesesuaian Isi",
    "n3": "Sistematika",
    "n4": "Ketajaman Analisis",
    "n5": "Penggunaan Bahasa",
}

# ── Prompts ───────────────────────────────────────────────────────────────────
QUERY_KEYWORDS = """
Penilaian Penulisan Makalah Form. 1 Penilaian Penulisan Makalah {selected_jabatan}
Kesesuaian Judul dengan Tema Kesesuaian Isi Makalah dengan Judul dan Tema
Sistematika Penulisan Ketajaman Analisis Penggunaan Bahasa dalam Penulisan Makalah
Bobot Penilaian Format Penulisan Struktur Makalah Penilaian Kompetensi Teknis
"""

PROMPT_KONTEKS = """
Anda adalah asisten yang bertugas mengumpulkan konteks relevan untuk penilaian makalah.
Berdasarkan jabatan '{selected_jabatan}', berikan ringkasan singkat tentang:
1. Deskripsi jabatan dan kompetensi yang diperlukan
2. Kriteria penilaian utama untuk posisi ini
3. Standar kualitas yang diharapkan dalam penulisan makalah
4. Rencana Strategis BPOM yang relevan untuk menilai kedalaman analisis
Berikan jawaban dalam format paragraf singkat, fokus pada poin-poin penting.
"""

PROMPT_PENILAIAN = """
---Role---
Anda adalah evaluator akademik sebagai Panitia Seleksi yang bertugas menilai kualitas substansi makalah secara objektif dan sistematis.

---Goal---
Melakukan penilaian terhadap makalah berdasarkan kriteria penilaian yang telah ditentukan, memberikan skor numerik untuk setiap kriteria, serta menyusun justifikasi yang jelas dan berbasis bukti dari isi makalah.

---Konteks Jabatan---
{assessment_context}

---Ketentuan Penulisan Makalah (Tema)---
{tema_text}

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
1. Kesesuaian judul dengan tema (berdasarkan Ketentuan Penulisan Makalah di atas)
2. Kesesuaian isi makalah dengan judul dan tema (berdasarkan Ketentuan Penulisan Makalah di atas)
3. Sistematika penulisan
4. Ketajaman analisis (bobot 2x)
5. Penggunaan bahasa dalam penulisan makalah

---Scoring Rules---
- Skor minimum: 40, maksimum: 100, harus bilangan bulat
- Ketajaman analisis memiliki bobot dua kali lipat dalam nilai akhir

---Makalah---
{makalah_text}

---Output Format---
Output MUST be a valid JSON format. All property names and string values MUST be enclosed in double quotes. Do not use trailing commas. Do not wrap the JSON in markdown blocks.
Example output format: 
{{
  "Ringkasan": "ringkasan isi makalah secara keseluruhan",
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

# ── Document Parsers ──────────────────────────────────────────────────────────
NS     = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
VMERGE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"


def _read_docx(filepath: str) -> str:
    with zipfile.ZipFile(filepath) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    lines = []
    for child in root.find(".//w:body", NS):
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text = "".join(
                r.find("w:t", NS).text for r in child.findall(".//w:r", NS)
                if r.find("w:t", NS) is not None and r.find("w:t", NS).text
            ).strip()
            if text:
                lines.append(text)
        elif tag == "tbl":
            for row in child.findall("w:tr", NS):
                cells = []
                for cell in row.findall("w:tc", NS):
                    vm = cell.find(".//w:vMerge", NS)
                    if vm is not None and vm.get(VMERGE) != "restart":
                        continue
                    text = "".join(
                        r.find("w:t", NS).text
                        for p in cell.findall(".//w:p", NS)
                        for r in p.findall(".//w:r", NS)
                        if r.find("w:t", NS) is not None and r.find("w:t", NS).text
                    ).strip()
                    if text:
                        cells.append(text)
                if cells:
                    lines.append(" | ".join(cells))
    return "\n".join(lines)


def _read_pdf(filepath: str) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            return "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
    except ImportError:
        return "[pdfplumber tidak terinstall]"


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
        return ""
    finally:
        os.unlink(tmp_path)


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        if suffix == ".docx":
            return _read_docx(tmp_path)
        elif suffix == ".pdf":
            return _read_pdf(tmp_path)
        elif suffix == ".txt":
            with open(tmp_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""
    finally:
        os.unlink(tmp_path)


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
    
    raw = raw.strip()
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1:
        raw = raw[start:end+1]
        
    raw = re.sub(r',\s*([\]}])', r'\1', raw)
    
    return json.loads(raw)


def score_color(score: float) -> str:
    if score >= 85:   return "#22c55e"
    elif score >= 70: return "#f59e0b"
    elif score >= 55: return "#f97316"
    else:             return "#ef4444"


# ── Uncertainty Sampling Functions ────────────────────────────────────────────

def calculate_uncertainty_metrics(scores_list: list) -> dict:
    """
    Calculate NSV, WAU, and uncertainty metrics from M samples.
    """
    score_distributions = {k: [] for k in CRITERIA_SHORT}
    
    for res in scores_list:
        if not res:
            continue
        for i, key_short in enumerate(CRITERIA_SHORT):
            key_full = CRITERIA_KEYS[i]
            if key_full in res:
                score_distributions[key_short].append(res[key_full])
    
    evaluation_results = {
        "consensus_scores": {},
        "uncertainty": {"per_criteria": {}},
        "valid_samples": len([s for s in scores_list if s])
    }
    
    nsv_dict = {}
    
    for i, (criteria_short, criteria_full) in enumerate(zip(CRITERIA_SHORT, CRITERIA_KEYS)):
        values = score_distributions[criteria_short]
        
        if not values:
            evaluation_results["consensus_scores"][criteria_full] = 0
            evaluation_results["uncertainty"]["per_criteria"][criteria_short] = {
                "mean": 0, "std": 0, "nsv": 0, "status": "❌ NO DATA", "raw_samples": []
            }
            nsv_dict[criteria_short] = 0
            continue
        
        mean_score = np.mean(values)
        std_dev = np.std(values, ddof=1) if len(values) > 1 else 0.0
        nsv = std_dev / SCORE_RANGE
        nsv_dict[criteria_short] = nsv
        
        status = "⚠️ PERLU REVIEW" if nsv > NSV_THRESHOLD else "✅ YAKIN"
        
        evaluation_results["consensus_scores"][criteria_full] = round(mean_score, 2)
        evaluation_results["uncertainty"]["per_criteria"][criteria_short] = {
            "mean": round(mean_score, 2),
            "std": round(std_dev, 2),
            "nsv": round(nsv, 3),
            "status": status,
            "raw_samples": [round(v, 1) for v in values]
        }
    
    # Calculate WAU (Weighted Aggregate Uncertainty)
    if all(k in nsv_dict for k in CRITERIA_SHORT):
        wau = (nsv_dict["n1"] + nsv_dict["n2"] + nsv_dict["n3"] + 
               (2 * nsv_dict["n4"]) + nsv_dict["n5"]) / 6
        
        evaluation_results["uncertainty"]["weighted_aggregate"] = round(wau, 3)
        evaluation_results["uncertainty"]["overall_status"] = (
            "⚠️ BUTUH REVIEW HUMAN" if wau > NSV_THRESHOLD else "✅ YAKIN (Konsisten)"
        )
        evaluation_results["uncertainty"]["most_uncertain_criteria"] = max(nsv_dict, key=nsv_dict.get)
    
    return evaluation_results


def render_uncertainty_visualization(uncertainty_data: dict):
    """
    Render enhanced uncertainty metrics with professional visualizations.
    Better than image.png - includes range plots, WAU score card, and detailed table.
    """
    if not uncertainty_data:
        st.info("Tidak ada data ketidakpastian untuk ditampilkan.")
        return
    
    per_criteria = uncertainty_data.get("per_criteria", {})
    wau = uncertainty_data.get("weighted_aggregate", 0)
    overall_status = uncertainty_data.get("overall_status", "")
    most_uncertain = uncertainty_data.get("most_uncertain_criteria", "-")
    
    # WAU Score Card - Professional design
    wau_color = "#22c55e" if "YAKIN" in overall_status else "#ef4444"
    bg_color = "#f0fdf4" if "YAKIN" in overall_status else "#fef2f2"
    
    st.markdown(f"""
    <div style="background:{bg_color}; border:2px solid {wau_color}; border-radius:16px; padding:20px; margin-bottom:24px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
                <div style="font-size:11px; color:#64748b; letter-spacing:2px; text-transform:uppercase;">Kalkulator Ketidakpastian LLM</div>
                <div style="font-size:12px; color:#64748b; margin-top:4px;">Weighted Aggregate Uncertainty (WAU)</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:48px; font-weight:800; color:{wau_color};">{wau:.4f}</div>
                <div style="font-size:11px; color:#64748b;">Threshold: {NSV_THRESHOLD}</div>
            </div>
            <div style="text-align:right;">
                <div style="background:{wau_color}; color:white; padding:6px 12px; border-radius:20px; font-size:12px; font-weight:600;">
                    {overall_status}
                </div>
                <div style="font-size:11px; color:#64748b; margin-top:4px;">Kriteria terumit: {most_uncertain.upper()}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Visualisasi Range", "📋 Tabel Ringkasan", "🔍 Detail Sampel"])
    
    with tab1:
        st.markdown("**Analisis Ketidakpastian per Kriteria**")
        
        for short_key in CRITERIA_SHORT:
            if short_key in per_criteria:
                data = per_criteria[short_key]
                mean_val = data.get("mean", 0)
                std_val = data.get("std", 0)
                nsv_val = data.get("nsv", 0)
                raw_samples = data.get("raw_samples", [])
                status = data.get("status", "")
                
                nsv_color = "#22c55e" if nsv_val <= NSV_THRESHOLD else "#f97316"
                status_icon = "✅" if "YAKIN" in status else "⚠️"
                
                # Header with metrics
                col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])
                with col1:
                    st.markdown(f"**{status_icon} {CRITERIA_LABELS.get(short_key, short_key.upper())}**")
                with col2:
                    st.metric("Mean", f"{mean_val:.1f}", delta=None)
                with col3:
                    st.metric("Std Dev", f"{std_val:.2f}", delta=None)
                with col4:
                    st.metric("NSV", f"{nsv_val:.3f}", delta_color="off")
                with col5:
                    st.markdown(f"<span style='color:{nsv_color}; font-weight:600;'>{status}</span>", unsafe_allow_html=True)
                
                # Range visualization with matplotlib
                if raw_samples and len(raw_samples) > 1:
                    fig, ax = plt.subplots(figsize=(12, 2.5), dpi=100)
                    
                    min_val = min(raw_samples)
                    max_val = max(raw_samples)
                    x_min = max(MIN_SCORE, min_val - 5)
                    x_max = min(MAX_SCORE, max_val + 5)
                    
                    # Draw confidence interval band
                    ax.axvspan(mean_val - std_val, mean_val + std_val, alpha=0.2, color=nsv_color, label=f'±1 Std Dev ({std_val:.1f})')
                    
                    # Draw min-max range line
                    ax.hlines(y=0, xmin=min_val, xmax=max_val, colors=nsv_color, linewidth=3, alpha=0.5, label=f'Range: {min_val:.0f}-{max_val:.0f}')
                    
                    # Plot sample points with jitter
                    y_positions = np.random.normal(0, 0.05, len(raw_samples))
                    ax.scatter(raw_samples, y_positions, s=120, alpha=0.8, color=nsv_color, 
                              edgecolors='black', linewidth=1.5, zorder=3, label=f'{len(raw_samples)} samples')
                    
                    # Plot mean line
                    ax.axvline(mean_val, color='#3b82f6', linestyle='--', linewidth=2.5, label=f'Mean: {mean_val:.1f}')
                    
                    # Formatting
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(-0.2, 0.2)
                    ax.set_xlabel('Skor', fontsize=10, fontweight='bold')
                    ax.set_yticks([])
                    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
                    ax.grid(axis='x', alpha=0.3, linestyle=':')
                    ax.set_title(f'Distribusi Skor {CRITERIA_LABELS.get(short_key, short_key.upper())}', fontsize=11, fontweight='bold', pad=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
    
    with tab2:
        # Create summary table
        table_data = []
        for short_key in CRITERIA_SHORT:
            if short_key in per_criteria:
                data = per_criteria[short_key]
                raw_samples = data.get("raw_samples", [])
                table_data.append({
                    "Kriteria": CRITERIA_LABELS.get(short_key, short_key.upper()),
                    "Rata-rata": f"{data.get('mean', 0):.1f}",
                    "Std Dev": f"{data.get('std', 0):.2f}",
                    "NSV": f"{data.get('nsv', 0):.3f}",
                    "Status": data.get('status', '-'),
                    "Min": f"{min(raw_samples) if raw_samples else 0:.0f}",
                    "Max": f"{max(raw_samples) if raw_samples else 0:.0f}",
                })
        
        if table_data:
            df_table = pd.DataFrame(table_data)
            st.dataframe(
                df_table, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Kriteria": st.column_config.TextColumn("Kriteria", width="medium"),
                }
            )
    
    with tab3:
        # Detailed raw samples with distribution
        st.markdown("**Distribusi Skor Mentah per Kriteria**")
        
        for short_key in CRITERIA_SHORT:
            if short_key in per_criteria:
                samples = per_criteria[short_key].get("raw_samples", [])
                if samples:
                    sample_counts = Counter(samples)
                    sample_str = " | ".join([f"{val:.0f} (×{count})" if count > 1 else f"{val:.0f}" 
                                            for val, count in sorted(sample_counts.items())])
                    
                    with st.expander(f"{CRITERIA_LABELS.get(short_key, short_key.upper())} - {len(samples)} samples"):
                        st.code(sample_str, language="text")
                        
                        # Simple distribution bar
                        unique_vals = sorted(set(samples))
                        if len(unique_vals) > 1:
                            counts = [samples.count(v) for v in unique_vals]
                            fig, ax = plt.subplots(figsize=(8, 3))
                            bars = ax.bar([str(v) for v in unique_vals], counts, color='#3b82f6', edgecolor='black')
                            ax.set_xlabel('Skor')
                            ax.set_ylabel('Frekuensi')
                            ax.set_title(f'Distribusi {CRITERIA_LABELS.get(short_key, short_key.upper())}')
                            for bar, count in zip(bars, counts):
                                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                       str(count), ha='center', va='bottom', fontweight='bold')
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)


# ── MinIO Helper ─────────────────────────────────────────────────────────────
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


def list_minio_files(bucket: str) -> list:
    client = get_minio_client()
    if not client:
        return []
    try:
        resp = client.list_objects_v2(Bucket=bucket)
        return [obj["Key"] for obj in resp.get("Contents", [])]
    except Exception:
        return []


def list_minio_files_detailed(bucket: str) -> list:
    client = get_minio_client()
    if not client:
        return []
    try:
        resp = client.list_objects_v2(Bucket=bucket)
        results = []
        for obj in resp.get("Contents", []):
            results.append({
                "Nama File": obj["Key"],
                "Ukuran": f"{obj['Size']:,} bytes",
                "Terakhir Diubah": obj["LastModified"].strftime("%Y-%m-%d %H:%M") if obj.get("LastModified") else "-",
            })
        return results
    except Exception:
        return []


def upload_to_minio(bucket: str, key: str, data: bytes) -> bool:
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


def download_from_minio(bucket: str, key: str):
    client = get_minio_client()
    if not client:
        return None
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except Exception as e:
        log.error(f"MinIO download failed: {e}")
        return None


# ── PostgreSQL Helper ─────────────────────────────────────────────────────────
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


def init_db() -> bool:
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
                    uncertainty_metrics JSONB,
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


def save_evaluation(paper_filename, jabatan, result_dict, query_mode) -> bool:
    conn = get_pg_conn()
    if not conn:
        return False
    try:
        scores        = result_dict.get("scores", {})
        final_score   = result_dict.get("final_score", compute_final_score(scores))
        justification = result_dict.get("justification", {})
        evidence      = result_dict.get("evidence", {})
        ringkasan     = result_dict.get("Ringkasan", "")
        uncertainty   = result_dict.get("uncertainty_metrics", {})
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO evaluation_results
                  (paper_filename, jabatan, scores, final_score, justification, evidence, ringkasan, query_mode, uncertainty_metrics)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                paper_filename, jabatan,
                json.dumps(scores), final_score,
                json.dumps(justification), json.dumps(evidence),
                ringkasan, query_mode,
                json.dumps(uncertainty),
            ))
        conn.commit()
        return True
    except Exception as e:
        log.error(f"Save evaluation failed: {e}")
        return False
    finally:
        conn.close()


def get_evaluation_history(jabatan_filter=None, limit=100, score_min=0) -> list:
    conn = get_pg_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            if jabatan_filter and jabatan_filter != "Semua":
                cur.execute("""
                    SELECT id, paper_filename, jabatan, scores, final_score,
                           justification, evidence, ringkasan, query_mode, uncertainty_metrics, created_at
                    FROM evaluation_results
                    WHERE jabatan = %s AND final_score >= %s
                    ORDER BY created_at DESC LIMIT %s
                """, (jabatan_filter, score_min, limit))
            else:
                cur.execute("""
                    SELECT id, paper_filename, jabatan, scores, final_score,
                           justification, evidence, ringkasan, query_mode, uncertainty_metrics, created_at
                    FROM evaluation_results
                    WHERE final_score >= %s
                    ORDER BY created_at DESC LIMIT %s
                """, (score_min, limit))
            rows = cur.fetchall()
            cols = ["id", "paper_filename", "jabatan", "scores", "final_score",
                    "justification", "evidence", "ringkasan", "query_mode", "uncertainty_metrics", "created_at"]
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


# ── RAG Async Functions ───────────────────────────────────────────────────────
import threading

@st.cache_resource(show_spinner=False)
def get_async_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop

def run_async(coro):
    loop = get_async_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


async def _initialize_rag():
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

    embedding_dim = int(os.getenv("EMBEDDING_DIM", 1536))
    token_limit   = int(os.getenv("EMBEDDING_TOKEN_LIMIT", 8192))
    model_name    = os.getenv("EMBEDDING_MODEL")

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
        enable_llm_cache=False, 
        enable_llm_cache_for_entity_extract=False,
    )
    await rag.initialize_storages()
    return rag


@st.cache_resource(show_spinner=False)
def get_rag():
    return run_async(_initialize_rag())


async def retrieve_assessment_context(rag, selected_jabatan: str, query_mode: str) -> str:
    from lightrag import QueryParam
    try:
        ctx = await rag.aquery(
            query=QUERY_KEYWORDS.format(selected_jabatan=selected_jabatan),
            param=QueryParam(
                mode=query_mode,
                user_prompt=PROMPT_KONTEKS.format(selected_jabatan=selected_jabatan),
            ),
        )
        return ctx if ctx else "Konteks jabatan tidak ditemukan dalam knowledge base."
    except Exception as e:
        return f"Error retrieving context: {str(e)}"


async def evaluate_paper_with_uncertainty(rag, makalah_text: str, assessment_context: str, 
                                         tema_text: str, m_samples: int = M_SAMPLES,
                                         temperature: float = TEMPERATURE) -> dict:
    """
    Evaluate paper with uncertainty sampling (M samples).
    - UI menampilkan Ringkasan, Justification, Evidence dari LAST sample
    - Download JSON menyimpan ALL samples
    """
    evaluation_prompt = PROMPT_PENILAIAN.format(
        assessment_context=assessment_context,
        makalah_text=makalah_text,
        tema_text=tema_text,
    )
    
    log.info(f"🚀 Starting {m_samples} sampling for uncertainty analysis...")
    
    # Create LLM function for sampling
    async def llm_sample(prompt: str, sample_id: int) -> dict | None:
        llm = ChatOpenAI(
            # model=os.getenv("LLM_PENILAI", "anthropic/claude-sonnet-4.6"),
            model=os.getenv("LLM_PENILAI", "openai/gpt-4o-mini"),
            api_key=os.getenv("LLM_BINDING_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST"),
            temperature=temperature,
        )
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                log.warning(f"Sample {sample_id}: No JSON found")
                return None
            
            clean_json = json_match.group(0)
            clean_json = re.sub(r',\s*([\]}])', r'\1', clean_json)
            data = json.loads(clean_json)
            
            # Extract scores
            scores = data.get("scores", {})
            valid_scores = {}
            for i, key_full in enumerate(CRITERIA_KEYS):
                short_key = CRITERIA_SHORT[i]
                val = scores.get(key_full, scores.get(short_key))
                if isinstance(val, (int, float)) and MIN_SCORE <= val <= MAX_SCORE:
                    valid_scores[key_full] = float(val)
            
            if valid_scores and len(valid_scores) == len(CRITERIA_KEYS):
                return {
                    "sample_id": sample_id,
                    "scores": valid_scores,
                    "Ringkasan": data.get("Ringkasan", ""),
                    "justification": data.get("justification", {}),
                    "evidence": data.get("evidence", {}),
                    "raw_response": content,  # Save raw response for debugging
                }
            else:
                log.warning(f"Sample {sample_id}: Invalid scores")
                return None
        except Exception as e:
            log.warning(f"Sample {sample_id}: Error - {e}")
            return None
    
    # Run M samples in parallel with retries
    async def sample_with_retry(prompt, sample_id, attempt=0):
        result = await llm_sample(prompt, sample_id)
        if result is None and attempt < SAMPLE_RETRY_LIMIT:
            log.warning(f"Sample {sample_id}: Retry attempt {attempt + 1}")
            return await sample_with_retry(prompt, sample_id, attempt + 1)
        return result
    
    tasks = [sample_with_retry(evaluation_prompt, i) for i in range(m_samples)]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    log.info(f"✅ Valid samples: {len(valid_results)}/{m_samples}")
    
    if not valid_results:
        # Return empty result if no valid samples
        return {
            "scores": {k: 0 for k in CRITERIA_KEYS},
            "final_score": 0,
            "uncertainty_metrics": {},
            "valid_samples": 0,
            "total_samples": m_samples,
            "raw_samples": [],
            "full_samples": [],
            "Ringkasan": "Gagal mendapatkan sampel yang valid",
            "justification": {},
            "evidence": {},
            "temperature_used": temperature,
            "m_samples_used": m_samples,
            "error": "No valid samples"
        }
    
    # Extract scores only for uncertainty calculation
    scores_list = [r["scores"] for r in valid_results]
    
    # Calculate uncertainty metrics
    metrics = calculate_uncertainty_metrics(scores_list)
    
    # Get consensus scores
    consensus = metrics["consensus_scores"]
    final_score = compute_final_score(consensus)
    
    # --- IMPORTANT: Use LAST sample for UI display ---
    # Sort by sample_id to ensure we get the last one
    valid_results_sorted = sorted(valid_results, key=lambda x: x.get("sample_id", 0))
    last_sample = valid_results_sorted[-1] if valid_results_sorted else None
    
    # Also find sample closest to consensus (for reference in JSON)
    best_sample = None
    if scores_list and consensus:
        best_score_diff = float('inf')
        for sample in valid_results:
            sample_scores = sample.get("scores", {})
            diff = sum((sample_scores.get(k, 0) - consensus.get(k, 0)) ** 2 for k in CRITERIA_KEYS)
            if diff < best_score_diff:
                best_score_diff = diff
                best_sample = sample
    
    # Prepare result
    result = {
        # Consensus scores (for final score calculation)
        "scores": consensus,
        "final_score": final_score,
        
        # Uncertainty metrics
        "uncertainty_metrics": metrics["uncertainty"],
        "valid_samples": len(valid_results),
        "total_samples": m_samples,
        
        # Raw data for download (ALL samples)
        "raw_samples": scores_list,  # Just scores from all samples
        "full_samples": valid_results,  # Complete samples with ringkasan, justification, evidence
        
        # For UI display: use LAST sample
        "Ringkasan": last_sample.get("Ringkasan", "") if last_sample else "",
        "justification": last_sample.get("justification", {}) if last_sample else {},
        "evidence": last_sample.get("evidence", {}) if last_sample else {},
        
        # For reference: sample closest to consensus (stored in JSON only)
        "best_sample_by_consensus": {
            "sample_id": best_sample.get("sample_id") if best_sample else None,
            "scores": best_sample.get("scores") if best_sample else {},
            "Ringkasan": best_sample.get("Ringkasan", "") if best_sample else "",
        } if best_sample else None,
        
        # Metadata
        "temperature_used": temperature,
        "m_samples_used": m_samples,
        "last_sample_id": last_sample.get("sample_id") if last_sample else None,
    }
    
    return result
# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
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

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.nav-btn { width: 100%; text-align: left; padding: 12px 16px; margin: 4px 0;
  border-radius: 10px; border: none; background: #1e293b; color: #94a3b8;
  font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
.nav-btn:hover { background: #334155; color: #e2e8f0; }
.nav-btn.active { background: linear-gradient(135deg, #3b82f6, #6366f1);
  color: white; box-shadow: 0 4px 12px rgba(59,130,246,0.3); }

.score-card { border-radius: 16px; padding: 24px; text-align: center; margin-bottom: 24px; }
.score-num { font-size: 72px; font-weight: 800; line-height: 1; }
.score-label { font-size: 12px; color: #64748b; letter-spacing: 2px; text-transform: uppercase; }

.step-row { display: flex; align-items: center; gap: 8px; margin-bottom: 24px; }
.step-circle { width: 32px; height: 32px; border-radius: 50%; display: flex;
  align-items: center; justify-content: center; font-weight: 700; font-size: 14px; flex-shrink: 0; }
.step-active { background: #3b82f6; color: white; }
.step-done { background: #22c55e; color: white; }
.step-idle { background: #1e293b; color: #64748b; border: 1px solid #334155; }
.step-line { flex: 1; height: 2px; background: #1e293b; }

.metric-box { background: #1e293b; border: 1px solid #334155; border-radius: 12px;
  padding: 16px; text-align: center; }

code, pre { font-family: 'JetBrains Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "nav_page": "📝 Penilaian Makalah",
        "skj_data": None,
        "db_ok": False,
        "rag_error": None,
        "eval_jabatan": None,
        "eval_paper_text": None,
        "eval_paper_name": None,
        "eval_context": None,
        "eval_result": None,
        "eval_saved": False,
        "minio_selected_files": [],
        "query_mode": "hybrid",
        "history_data": None,
        "m_samples": M_SAMPLES,
        "temperature": TEMPERATURE,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# Auto-init DB silently
if not st.session_state.db_ok:
    ok = init_db()
    st.session_state.db_ok = ok

# Auto-init RAG silently
rag_instance = None
rag_ready = False
try:
    rag_instance = get_rag()
    rag_ready = True
except Exception as e:
    st.session_state.rag_error = str(e)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px;">
      <div style="font-size:32px;">📋</div>
      <div style="font-size:18px; font-weight:800; color:#e2e8f0;">Penilaian Makalah</div>
      <div style="font-size:11px; color:#64748b; letter-spacing:1px;">Powered by LightRAG · BPOM</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    nav_options = ["📝 Penilaian Makalah", "📊 Riwayat Penilaian", "⚙️ Docs & Settings"]
    selected_nav = st.radio("Navigasi", nav_options,
                            index=nav_options.index(st.session_state.nav_page),
                            label_visibility="collapsed")
    st.session_state.nav_page = selected_nav

    st.divider()

    # System status
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(f"{'🟢' if rag_ready else '🔴'}")
        st.caption("RAG")
    with col_s2:
        st.markdown(f"{'🟢' if st.session_state.db_ok else '🟡'}")
        st.caption("DB")
    with col_s3:
        minio_ok = bool(get_minio_client())
        st.markdown(f"{'🟢' if minio_ok else '🟡'}")
        st.caption("MinIO")

    if st.session_state.rag_error:
        with st.expander("⚠️ RAG Error"):
            st.error(st.session_state.rag_error[:200])

    st.divider()

    # Query mode
    qmode = st.selectbox("Query Mode", ["hybrid", "mix", "local", "global", "naive"],
                         index=0, help="Mode query LightRAG")
    st.session_state.query_mode = qmode
    
    st.divider()
    
    # Uncertainty Sampling Settings
    st.markdown("### 🎲 Uncertainty Settings")
    m_samples_val = st.slider("Jumlah Sampel (M)", min_value=3, max_value=15, value=st.session_state.m_samples, step=1,
                              help="Semakin banyak sampel, semakin akurat estimasi ketidakpastian namun lebih lambat dan mahal")
    st.session_state.m_samples = m_samples_val
    
    temp_val = st.slider("Temperature Sampling", min_value=0.5, max_value=2.0, value=st.session_state.temperature, step=0.1,
                         help="Temperature tinggi = variasi skor lebih besar (untuk mengukur ketidakpastian)")
    st.session_state.temperature = temp_val
    
    st.caption(f"NSV Threshold: {NSV_THRESHOLD}")

    st.divider()
    st.caption(f"LLM: `{os.getenv('LLM_MODEL','?')}`")
    st.caption(f"Embed: `{os.getenv('EMBEDDING_MODEL','?')}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PENILAIAN MAKALAH
# ══════════════════════════════════════════════════════════════════════════════

def render_score_card(result: dict):
    scores    = result.get("scores", {})
    final     = result.get("final_score", compute_final_score(scores))
    justif    = result.get("justification", {})
    evidence  = result.get("evidence", {})
    ringkasan = result.get("Ringkasan", "")
    color     = score_color(final)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{color}22,{color}11);
      border:2px solid {color};border-radius:20px;padding:32px;text-align:center;margin-bottom:24px;">
      <div style="font-size:13px;color:#94a3b8;font-weight:600;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;">NILAI AKHIR</div>
      <div style="font-size:80px;font-weight:800;color:{color};line-height:1;">{final}</div>
      <div style="font-size:12px;color:#64748b;margin-top:4px;">dari 100</div>
    </div>
    """, unsafe_allow_html=True)

    if ringkasan:
        st.markdown("**📄 Ringkasan Makalah**")
        st.info(ringkasan)

    # Display uncertainty metrics
    if result.get("uncertainty_metrics"):
        st.divider()
        st.markdown("### 🎲 Analisis Ketidakpastian Skor (M-Sampling)")
        st.caption(f"Berdasarkan {result.get('valid_samples', 0)} sampel valid dari {st.session_state.m_samples} total sampling dengan temperature={st.session_state.temperature}")
        render_uncertainty_visualization(result.get("uncertainty_metrics"))
        st.divider()

    st.markdown("**Rincian Skor per Kriteria**")
    for key, label in SCORE_LABELS.items():
        score  = scores.get(key, 0)
        weight = SCORE_WEIGHTS[key]
        sc     = score_color(score)
        with st.expander(f"{label}  —  **{score}**"):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.markdown(f"""
                <div style="background:{sc}22;border:2px solid {sc};border-radius:12px;
                  padding:16px;text-align:center;">
                  <div style="font-size:36px;font-weight:800;color:{sc};">{score}</div>
                  <div style="font-size:11px;color:#64748b;">bobot ×{weight}</div>
                </div>""", unsafe_allow_html=True)
                st.progress(max(0, min(100, int(score))) / 100)
            with c2:
                j = justif.get(key, "")
                e = evidence.get(key, "")
                if j:
                    st.markdown(f"**📝 Justifikasi:**\n{j}")
                if e:
                    st.markdown(f"**🔎 Bukti:**\n_{e}_")


def page_penilaian():
    st.markdown("## 📝 Penilaian Makalah")

    if not rag_ready:
        st.error("❌ LightRAG gagal diinisialisasi. Periksa konfigurasi di tab Settings.")
        if st.session_state.rag_error:
            st.code(st.session_state.rag_error)
        return

    # Load SKJ
    if st.session_state.skj_data is None:
        st.session_state.skj_data = load_all_skj()
    skj_data    = st.session_state.skj_data or {}
    jabatan_list = list(skj_data.keys())

    # Step 1: Pilih Jabatan & Tema
    with st.container(border=True):
        st.markdown("""<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
          <div style="background:#3b82f6;color:white;width:28px;height:28px;border-radius:50%;
            display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0;">1</div>
          <div style="font-size:16px;font-weight:700;color:#090088;">Pilih Jabatan & Tema Penulisan Makalah</div>
        </div>""", unsafe_allow_html=True)

        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.markdown("**1A. Pilih Jabatan yang Dituju**")
            if jabatan_list:
                sel_jabatan = st.selectbox("Jabatan", jabatan_list, label_visibility="collapsed")
            else:
                sel_jabatan = st.text_input("Jabatan", placeholder="contoh: Sekretaris Utama", label_visibility="collapsed")
            st.session_state.eval_jabatan = sel_jabatan

        with col1_2:
            st.markdown("**1B. Pilih Ketentuan Tema**")
            tema_files = list_minio_files(BUCKET_TEMA)
            if tema_files:
                sel_tema = st.selectbox("Ketentuan Tema (PDF/DOCX)", tema_files, label_visibility="collapsed")
            else:
                sel_tema = st.selectbox("Ketentuan Tema (PDF/DOCX)", ["(Tidak ada file di MinIO)"], disabled=True, label_visibility="collapsed")
                st.warning(f"⚠️ Bucket `{BUCKET_TEMA}` kosong atau tidak dapat diakses.")
            st.session_state.eval_tema_file = sel_tema if tema_files else None

    st.divider()

    # Step 2: Pilih / Upload Makalah
    with st.container(border=True):
        st.markdown("""<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
          <div style="background:#3b82f6;color:white;width:28px;height:28px;border-radius:50%;
            display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0;">2</div>
          <div style="font-size:16px;font-weight:700;color:#090088;">Pilih Makalah yang Akan Dinilai</div>
        </div>""", unsafe_allow_html=True)

        src_tab1, src_tab2 = st.tabs(["📂 Pilih dari MinIO", "⬆️ Upload File Baru"])

        with src_tab1:
            minio_files_detail = list_minio_files_detailed(BUCKET_MAKALAH)
            if minio_files_detail:
                df_minio = pd.DataFrame(minio_files_detail)
                df_minio.insert(0, "Pilih", False)
                
                st.markdown(f"**{len(minio_files_detail)} file tersedia di MinIO (bucket: `{BUCKET_MAKALAH}`)**")
                st.markdown("Centang file yang ingin dinilai pada tabel di bawah:")

                edited_df = st.data_editor(
                    df_minio,
                    column_config={
                        "Pilih": st.column_config.CheckboxColumn(
                            "Pilih",
                            help="Pilih makalah untuk dinilai",
                            default=False,
                        )
                    },
                    disabled=["Nama File", "Ukuran", "Terakhir Diubah"],
                    hide_index=True,
                    use_container_width=True,
                )

                sel_keys = edited_df[edited_df["Pilih"]]["Nama File"].tolist()

                if sel_keys:
                    if st.button(f"📥 Load {len(sel_keys)} File Terpilih", type="secondary", use_container_width=True):
                        for key in sel_keys:
                            data = download_from_minio(BUCKET_MAKALAH, key)
                            if data:
                                txt = extract_text_from_bytes(data, key)
                                if txt.strip():
                                    st.session_state.setdefault("loaded_papers", {})[key] = txt
                                    st.toast(f"✅ Berhasil load: {key}")
                                else:
                                    st.toast(f"⚠️ Kosong: {key}")
                            else:
                                st.toast(f"❌ Gagal: {key}")
            else:
                st.info("Belum ada file di MinIO bucket 'makalah'. Upload melalui tab Upload File Baru.")

        with src_tab2:
            uploaded_files = st.file_uploader(
                "Upload makalah (PDF / DOCX / TXT) — bisa multiple",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="paper_uploader",
            )
            if uploaded_files:
                for uf in uploaded_files:
                    if uf.name not in st.session_state.get("loaded_papers", {}):
                        uf.seek(0)
                        txt = extract_text_from_uploaded(uf)
                        if txt.strip():
                            st.session_state.setdefault("loaded_papers", {})[uf.name] = txt
                            uf.seek(0)
                            upload_to_minio(BUCKET_MAKALAH, uf.name, uf.read())
                            st.toast(f"✅ Berhasil unggah: {uf.name}")
                        else:
                            st.toast(f"❌ Gagal baca: {uf.name}")

        # Show loaded papers
        loaded_papers = st.session_state.get("loaded_papers", {})
        if loaded_papers:
            st.markdown(f"**{len(loaded_papers)} makalah siap dinilai:**")
            for fn, txt in loaded_papers.items():
                with st.expander(f"👁️ Preview: {fn}"):
                    st.text(txt[:400] + ("..." if len(txt) > 400 else ""))
            if st.button("🗑️ Bersihkan Daftar Makalah & Hasil Penilaian", key="clear_papers", use_container_width=True):
                st.session_state.pop("loaded_papers", None)
                st.session_state.pop("eval_results_batch", None)
                st.session_state.pop("eval_context", None)
                st.rerun()

    st.divider()

    # Step 3: Mulai Penilaian
    with st.container(border=True):
        st.markdown("""<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
          <div style="background:#3b82f6;color:white;width:28px;height:28px;border-radius:50%;
            display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0;">3</div>
          <div style="font-size:16px;font-weight:700;color:#090088;">Mulai Penilaian</div>
        </div>""", unsafe_allow_html=True)

        qmode    = st.session_state.query_mode
        sel_tema = st.session_state.get("eval_tema_file")
        can_eval = bool(sel_jabatan and sel_tema and loaded_papers and rag_ready)

        if not can_eval:
            if not sel_jabatan or not sel_tema:
                st.info("ℹ️ Selesaikan Langkah 1: Pilih jabatan dan ketentuan tema terlebih dahulu.")
            elif not loaded_papers:
                st.info("ℹ️ Selesaikan Langkah 2: Load minimal 1 makalah terlebih dahulu.")
        else:
            st.info(f"Siap menilai **{len(loaded_papers)}** makalah untuk jabatan **{sel_jabatan}** dengan tema **{sel_tema}** (mode: `{qmode}`, M-Samples: {st.session_state.m_samples}, Temperature: {st.session_state.temperature})")

        eval_btn = st.button("🚀 Mulai Penilaian Otomatis (M-Sampling)", type="primary",
                             disabled=not can_eval, use_container_width=True)

        if eval_btn:
            results = {}
            with st.status("Memproses Penilaian dengan Uncertainty Sampling...", expanded=True) as status_box:
                try:
                    # Stage 1: retrieve context once
                    st.write(f"⏳ Mengambil konteks jabatan '{sel_jabatan}'...")
                    ctx = run_async(retrieve_assessment_context(rag_instance, sel_jabatan, qmode))
                    st.session_state.eval_context = ctx

                    # Stage 1.5: retrieve tema text
                    st.write(f"⏳ Mengunduh file ketentuan tema: {sel_tema}...")
                    tema_data = download_from_minio(BUCKET_TEMA, sel_tema)
                    tema_text = extract_text_from_bytes(tema_data, sel_tema) if tema_data else "Tidak ada teks tema."
                    if len(tema_text) > 5000:
                        tema_text = tema_text[:5000] + "\n\n... (teks dipotong karena terlalu panjang)"

                    # Stage 2: evaluate each paper with M-Sampling
                    total = len(loaded_papers)
                    for idx, (fname, txt) in enumerate(loaded_papers.items(), 1):
                        st.write(f"⏳ Menilai dengan {st.session_state.m_samples} sampel: {fname} ({idx}/{total})...")
                        result = run_async(evaluate_paper_with_uncertainty(
                            rag_instance, txt, ctx, tema_text, 
                            m_samples=st.session_state.m_samples,
                            temperature=st.session_state.temperature
                        ))
                        results[fname] = result

                        # Auto-save to DB & MinIO
                        if st.session_state.db_ok:
                            save_evaluation(fname, sel_jabatan, result, qmode)
                        result_json = json.dumps(result, ensure_ascii=False, indent=2)
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        upload_to_minio(BUCKET_RIWAYAT,
                                        f"eval_{ts}_{Path(fname).stem}.json",
                                        result_json.encode("utf-8"))

                    status_box.update(label=f"✅ Penilaian selesai! {len(results)} makalah telah dinilai dengan {st.session_state.m_samples} sampel masing-masing.", state="complete", expanded=False)
                    st.session_state["eval_results_batch"] = results

                except Exception as e:
                    status_box.update(label=f"❌ Error: {e}", state="error")

    # Step 4: Tampilkan Hasil
    batch = st.session_state.get("eval_results_batch", {})
    if batch:
        st.divider()
        st.markdown("## 📊 Hasil Penilaian")

        # Summary table
        rows = []
        for fn, r in batch.items():
            sc = r.get("scores", {})
            unc = r.get("uncertainty_metrics", {})
            wau = unc.get("weighted_aggregate", 0)
            rows.append({
                "Makalah": fn,
                "Judul": sc.get("n1_kesesuaian_judul", None),
                "Isi": sc.get("n2_kesesuaian_isi", None),
                "Sistematika": sc.get("n3_sistematika", None),
                "Analisis": sc.get("n4_ketajaman_analisis", None),
                "Bahasa": sc.get("n5_penggunaan_bahasa", None),
                "Nilai Akhir": r.get("final_score", 0),
                "WAU": wau,
                "Status": unc.get("overall_status", "-")[:20],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Detail per makalah
        st.markdown("### 🔍 Detail per Makalah")
        doc_tabs = st.tabs([f"📄 {fn}" for fn in batch.keys()])
        for tab, (fn, r) in zip(doc_tabs, batch.items()):
            with tab:
                col_left, col_right = st.columns([1, 1.2])
                with col_left:
                    st.markdown("#### 📄 Teks Makalah")
                    paper_text = st.session_state.get("loaded_papers", {}).get(fn, "Teks makalah tidak ditemukan di session.")
                    st.markdown(
                        f'<div style="height:800px;overflow-y:auto;padding:16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;font-size:14px;white-space:pre-wrap;">{paper_text}</div>',
                        unsafe_allow_html=True
                    )
                with col_right:
                    render_score_card(r)
                    st.divider()
                    col_dl, col_js = st.columns(2)
                    with col_dl:
                            # Prepare full export with ALL samples
                        export_data = {
                            "metadata": {
                                "paper_filename": fn,
                                "jabatan": sel_jabatan,
                                "query_mode": qmode,
                                "evaluation_date": datetime.datetime.now().isoformat(),
                                "temperature_used": r.get("temperature_used", st.session_state.temperature),
                                "m_samples": r.get("total_samples", st.session_state.m_samples),
                                "valid_samples": r.get("valid_samples", 0),
                            },
                            "consensus_results": {
                                "final_score": r.get("final_score"),
                                "scores": r.get("scores"),
                                "uncertainty_metrics": r.get("uncertainty_metrics"),
                            },
                            "ui_display_sample": {
                                "sample_id": r.get("last_sample_id"),
                                "Ringkasan": r.get("Ringkasan"),
                                "justification": r.get("justification"),
                                "evidence": r.get("evidence"),
                            },
                            "all_samples": r.get("full_samples", []),  # ALL M samples with full details
                            "scores_only": r.get("raw_samples", []),   # Just scores from all samples
                            "best_sample_by_consensus": r.get("best_sample_by_consensus"),
                        }

                        st.download_button(
                            "⬇️ Download JSON (Lengkap - Semua Sample)",
                            data=json.dumps(export_data, ensure_ascii=False, indent=2),
                            file_name=f"penilaian_{Path(fn).stem}_all_{st.session_state.m_samples}samples.json",
                            mime="application/json",
                            use_container_width=True,
                            key=f"dl_batch_{fn}",
                        )
                    with col_js:
                        with st.expander("🔧 Raw JSON"):
                            st.json(r)

        if st.button("🔄 Reset / Nilai Makalah Baru", key="reset_all", type="secondary", use_container_width=True):
            for k in ["loaded_papers", "eval_results_batch", "eval_context"]:
                st.session_state.pop(k, None)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: RIWAYAT PENILAIAN
# ══════════════════════════════════════════════════════════════════════════════

def page_riwayat():
    st.markdown("## 📊 Riwayat Penilaian Makalah")

    if not st.session_state.db_ok:
        st.warning("⚠️ Database belum terhubung. Pastikan PostgreSQL berjalan dan konfigurasi benar.")
        if st.button("🔄 Coba Hubungkan DB"):
            ok = init_db()
            st.session_state.db_ok = ok
            if ok:
                st.success("✅ Database terhubung!")
                st.rerun()
            else:
                st.error("Gagal terhubung ke PostgreSQL.")
        return

    # Filters
    if st.session_state.skj_data is None:
        st.session_state.skj_data = load_all_skj()
    skj_data = st.session_state.skj_data or {}

    with st.container():
        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 1, 1, 1])
        with col_f1:
            jabatan_opts = ["Semua"] + list(skj_data.keys())
            filter_jabatan = st.selectbox("Filter Jabatan", jabatan_opts, key="hist_jabatan")
        with col_f2:
            filter_limit = st.number_input("Maks. Data", min_value=10, max_value=500, value=100, step=10)
        with col_f3:
            filter_score_min = st.number_input("Nilai Min.", min_value=0, max_value=100, value=0)
        with col_f4:
            st.write("")
            st.write("")
            refresh_btn = st.button("🔄 Muat Riwayat", use_container_width=True)

    if refresh_btn or st.session_state.history_data is None:
        with st.spinner("Memuat riwayat..."):
            st.session_state.history_data = get_evaluation_history(
                jabatan_filter=filter_jabatan if filter_jabatan != "Semua" else None,
                limit=int(filter_limit),
                score_min=filter_score_min,
            )

    history = st.session_state.history_data or []

    if not history:
        st.info("Belum ada riwayat penilaian. Lakukan penilaian di menu 'Penilaian Makalah' terlebih dahulu.")
        return

    # Summary metrics
    scores_list = [h.get("final_score", 0) for h in history if h.get("final_score") is not None]
    if scores_list:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Evaluasi", len(history))
        m2.metric("Rata-rata Nilai", f"{sum(scores_list)/len(scores_list):.1f}")
        m3.metric("Nilai Tertinggi", f"{max(scores_list):.1f}")
        m4.metric("Nilai Terendah", f"{min(scores_list):.1f}")

    st.divider()

    # Table
    table_data = []
    for h in history:
        table_data.append({
            "ID": h["id"],
            "Makalah": h["paper_filename"],
            "Jabatan": h["jabatan"],
            "Nilai Akhir": h["final_score"],
            "Mode": h["query_mode"],
            "Tanggal": str(h["created_at"])[:16] if h["created_at"] else "-",
        })
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, height=280, hide_index=True)

    # Download all
    all_export = json.dumps(
        {"total": len(history), "data": [
            {**h, "created_at": str(h["created_at"]),
             "scores": h["scores"] if not isinstance(h["scores"], str) else json.loads(h["scores"] or "{}"),
             "justification": h["justification"] if not isinstance(h["justification"], str) else json.loads(h["justification"] or "{}"),
            } for h in history
        ]},
        ensure_ascii=False, indent=2
    )
    st.download_button("📥 Export Semua Riwayat (JSON)", data=all_export,
                       file_name=f"riwayat_penilaian_{len(history)}data.json",
                       mime="application/json", use_container_width=False)

    st.divider()
    st.markdown("### 🔍 Detail per Evaluasi")

    for h in history:
        final = h.get("final_score", 0) or 0
        color = score_color(final)
        date_str = str(h["created_at"])[:10] if h["created_at"] else "-"
        with st.expander(
            f"[{final:.1f}]  {h['paper_filename']}  ·  {h['jabatan']}  ·  {date_str}"
        ):
            scores = h.get("scores") or {}
            if isinstance(scores, str):
                try: scores = json.loads(scores)
                except: scores = {}
            justif = h.get("justification") or {}
            if isinstance(justif, str):
                try: justif = json.loads(justif)
                except: justif = {}
            uncertainty = h.get("uncertainty_metrics") or {}
            if isinstance(uncertainty, str):
                try: uncertainty = json.loads(uncertainty)
                except: uncertainty = {}

            # Score card compact
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{color}22,{color}11);
              border:1px solid {color};border-radius:12px;padding:16px;text-align:center;margin-bottom:16px;">
              <div style="font-size:11px;color:#94a3b8;letter-spacing:2px;">NILAI AKHIR</div>
              <div style="font-size:48px;font-weight:800;color:{color};line-height:1.1;">{final:.1f}</div>
            </div>""", unsafe_allow_html=True)

            # Show uncertainty if available
            if uncertainty:
                wau = uncertainty.get("weighted_aggregate", 0)
                status = uncertainty.get("overall_status", "")
                st.markdown(f"""
                <div style="background:#1e293b;border-radius:12px;padding:12px;margin-bottom:16px;">
                  <div style="display:flex;justify-content:space-between;">
                    <span style="font-size:12px;color:#94a3b8;">WAU (Ketidakpastian):</span>
                    <span style="font-size:16px;font-weight:700;color:{"#22c55e" if "YAKIN" in status else "#f97316"}">{wau:.4f}</span>
                  </div>
                  <div style="font-size:11px;color:#64748b;">{status}</div>
                </div>
                """, unsafe_allow_html=True)

            ringkasan = h.get("ringkasan", "")
            if ringkasan:
                st.markdown(f"**Ringkasan:** {ringkasan[:300]}{'...' if len(ringkasan) > 300 else ''}")

            st.markdown("**Skor per Kriteria:**")
            for key, label in SCORE_LABELS.items():
                score = scores.get(key, "-")
                j = justif.get(key, "")
                w = SCORE_WEIGHTS.get(key, 1)
                st.markdown(f"- {'⭐ ' if w > 1 else ''}**{label}**: `{score}`")
                if j:
                    st.caption(f"  → {j[:150]}")

            # Download
            result_export = {
                "id": h["id"], "paper_filename": h["paper_filename"],
                "jabatan": h["jabatan"], "final_score": h["final_score"],
                "scores": scores, "justification": justif,
                "evidence": h.get("evidence") or {},
                "ringkasan": h.get("ringkasan", ""),
                "query_mode": h.get("query_mode", ""),
                "uncertainty_metrics": uncertainty,
                "created_at": str(h["created_at"]),
            }
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(result_export, ensure_ascii=False, indent=2),
                file_name=f"eval_{h['id']}_{h['paper_filename']}.json",
                mime="application/json",
                key=f"dl_hist_{h['id']}",
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: DOCS & SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

def page_settings():
    st.markdown("## ⚙️ Docs Ingestion & Settings")

    dtab1, dtab2 = st.tabs(["📥 Ingestion Dokumen Knowledge Base", "🔧 Konfigurasi Sistem"])

    # Tab Ingestion
    with dtab1:
        st.markdown("Upload dokumen SKJ, peraturan, atau renstra untuk dimasukkan ke LightRAG knowledge base.")

        if not rag_ready:
            st.error("❌ LightRAG belum aktif. Periksa konfigurasi sistem.")
            return

        col_up, col_info = st.columns([2, 1])

        with col_up:
            ing_files = st.file_uploader(
                "Upload dokumen (PDF / DOCX / TXT) — bisa multiple",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="ing_uploader",
            )
            doc_type = st.selectbox("Tipe Dokumen",
                ["SKJ", "Peraturan Pemerintah", "Renstra BPOM", "Lainnya"])
            also_minio = st.checkbox("Simpan juga ke MinIO (bucket: knowledge)", value=True)

            ingest_btn = st.button(
                "⚡ Mulai Ingestion ke LightRAG",
                disabled=not ing_files,
                type="primary",
                use_container_width=True,
            )

        with col_info:
            st.markdown("**📌 Catatan:**")
            st.info("- Format: PDF, DOCX, TXT\n- Bisa upload banyak file\n- Proses bisa memakan waktu beberapa menit\n- Dokumen diindeks ke graph knowledge base")

            st.markdown("**📦 File di Knowledge Base (MinIO):**")
            kb_files = list_minio_files(BUCKET_KNOWLEDGE)
            if kb_files:
                for f in kb_files[:15]:
                    st.markdown(f"  - `{f}`")
                if len(kb_files) > 15:
                    st.caption(f"... dan {len(kb_files)-15} file lainnya")
                if st.button("🔄 Refresh Daftar"):
                    st.rerun()
            else:
                st.caption("(belum ada file)")

        if ingest_btn and ing_files:
            prog = st.progress(0, text="Memulai ingestion...")
            log_rows = []
            for i, uf in enumerate(ing_files):
                prog.progress(i / len(ing_files), text=f"Memproses: {uf.name}")
                try:
                    uf.seek(0)
                    txt = extract_text_from_uploaded(uf)
                    if not txt.strip():
                        log_rows.append({"File": uf.name, "Status": "⚠️ Skip", "Info": "Teks kosong"})
                        log_ingestion(uf.name, "skip", "Teks kosong")
                        continue

                    if also_minio:
                        uf.seek(0)
                        ok = upload_to_minio(BUCKET_KNOWLEDGE, uf.name, uf.read())
                        if not ok:
                            log_rows.append({"File": uf.name, "Status": "⚠️ MinIO gagal", "Info": "Ingest tetap dilanjutkan"})

                    run_async(rag_instance.ainsert([txt], file_paths=[uf.name]))
                    log_rows.append({"File": uf.name, "Status": "✅ Berhasil", "Info": f"{len(txt):,} karakter"})
                    log_ingestion(uf.name, "success")

                except Exception as e:
                    log_rows.append({"File": uf.name, "Status": "❌ Error", "Info": str(e)[:100]})
                    log_ingestion(uf.name, "error", str(e))

            prog.progress(1.0, text="Selesai!")
            ok_count = sum(1 for r in log_rows if "✅" in r["Status"])
            st.success(f"Ingestion selesai! {ok_count}/{len(log_rows)} berhasil.")
            st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

        st.divider()
        with st.expander("📂 Ingestion dari Folder (path di server)"):
            folder_path = st.text_input("Path folder", placeholder="/path/to/documents")
            if st.button("Ingest Folder", disabled=not folder_path):
                if not os.path.exists(folder_path):
                    st.error(f"Folder tidak ditemukan: `{folder_path}`")
                else:
                    docs = []
                    for path in Path(folder_path).rglob("*"):
                        suffix = path.suffix.lower()
                        if suffix == ".docx":
                            txt = _read_docx(str(path))
                        elif suffix == ".pdf":
                            txt = _read_pdf(str(path))
                        elif suffix == ".txt":
                            txt = open(str(path), "r", encoding="utf-8").read()
                        else:
                            continue
                        if txt.strip():
                            docs.append({"filename": path.name, "text": txt})
                    if not docs:
                        st.warning("Tidak ada file valid di folder tersebut.")
                    else:
                        with st.spinner(f"Mengingest {len(docs)} dokumen..."):
                            run_async(rag_instance.ainsert(
                                [d["text"] for d in docs],
                                file_paths=[d["filename"] for d in docs],
                            ))
                        st.success(f"✅ {len(docs)} dokumen berhasil diingest dari `{folder_path}`")

    # Tab Settings
    with dtab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### 🤖 Konfigurasi Model")
            st.markdown("**LLM Model**")
            st.code(os.getenv("LLM_MODEL", "(tidak ditemukan)"))
            st.markdown("**LLM Binding**")
            st.code(os.getenv("LLM_BINDING", "(tidak ditemukan)"))
            st.markdown("**LLM Host**")
            st.code(os.getenv("LLM_BINDING_HOST", "(tidak ditemukan)"))
            st.markdown("**LLM Penilai**")
            st.code(os.getenv("LLM_PENILAI", "(tidak ditemukan)"))
            st.markdown("**Embedding Model**")
            st.code(os.getenv("EMBEDDING_MODEL", "(tidak ditemukan)"))
            st.markdown("**Embedding Dim**")
            st.code(os.getenv("EMBEDDING_DIM", "(tidak ditemukan)"))

        with col_b:
            st.markdown("### 🗄️ Status Koneksi")

            st.markdown("**LightRAG**")
            if rag_ready:
                st.success("✅ RAG Aktif")
            else:
                st.error("❌ RAG Error")
                if st.session_state.rag_error:
                    st.code(st.session_state.rag_error[:300])

            st.markdown("**PostgreSQL**")
            if st.session_state.db_ok:
                st.success(f"✅ Terhubung ({PG_HOST}:{PG_PORT}/{PG_DB})")
            else:
                st.warning("⚠️ Tidak terhubung")
                if st.button("🔄 Hubungkan DB"):
                    ok = init_db()
                    st.session_state.db_ok = ok
                    st.rerun()

            st.markdown("**MinIO**")
            mc = get_minio_client()
            if mc:
                st.success(f"✅ Terhubung ({MINIO_ENDPOINT})")
            else:
                st.warning(f"⚠️ Tidak terhubung ({MINIO_ENDPOINT})")

            st.divider()
            st.markdown("### 🎲 Uncertainty Configuration")
            st.markdown(f"**Default M-Samples:** {M_SAMPLES}")
            st.markdown(f"**Default Temperature:** {TEMPERATURE}")
            st.markdown(f"**NSV Threshold:** {NSV_THRESHOLD}")
            st.markdown(f"**Score Range:** {MIN_SCORE} - {MAX_SCORE}")

            st.divider()
            st.markdown("**Storage LightRAG**")
            st.code(f"Working Dir: {WORKING_DIR}")
            st.code(f"KV: {os.getenv('LIGHTRAG_KV_STORAGE','?')}")
            st.code(f"Vector: {os.getenv('LIGHTRAG_VECTOR_STORAGE','?')}")
            st.code(f"Graph: {os.getenv('LIGHTRAG_GRAPH_STORAGE','?')}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

page = st.session_state.nav_page

if page == "📝 Penilaian Makalah":
    page_penilaian()
elif page == "📊 Riwayat Penilaian":
    page_riwayat()
elif page == "⚙️ Docs & Settings":
    page_settings()