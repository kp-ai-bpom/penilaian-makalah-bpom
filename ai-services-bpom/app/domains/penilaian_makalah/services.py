import json
import re
import tempfile
import os
import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from .repositories import MinioRepository, EvaluationRepository
from .schemas import PenilaianRequest, PenilaianResponse

# ── Constants & Prompts ───────────────────────────────────────────────────────
SCORE_WEIGHTS = {
    "n1_kesesuaian_judul":   1,
    "n2_kesesuaian_isi":     1,
    "n3_sistematika":        1,
    "n4_ketajaman_analisis": 2,
    "n5_penggunaan_bahasa":  1,
}

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

# ── Extractor Helper ─────────────────────────────────────────────────────────
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

class DocumentExtractor:
    @staticmethod
    def read_docx(filepath: str) -> str:
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
                if text: lines.append(text)
        return "\n".join(lines)

    @staticmethod
    def read_pdf(filepath: str) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                return "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
        except ImportError:
            return "[pdfplumber tidak terinstall]"

    @staticmethod
    def extract_from_bytes(data: bytes, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            if suffix == ".docx":
                import zipfile
                return DocumentExtractor.read_docx(tmp_path)
            elif suffix == ".pdf":
                return DocumentExtractor.read_pdf(tmp_path)
            elif suffix == ".txt":
                with open(tmp_path, "r", encoding="utf-8") as f:
                    return f.read()
            return ""
        finally:
            os.unlink(tmp_path)

# ── Service Class ────────────────────────────────────────────────────────────
class PenilaianService:
    def __init__(self, db_repo: EvaluationRepository, minio_repo: MinioRepository, rag_instance):
        self.db_repo = db_repo
        self.minio_repo = minio_repo
        self.rag = rag_instance

    def compute_final_score(self, scores: dict) -> float:
        total_weight = sum(SCORE_WEIGHTS.values())
        weighted_sum = sum(scores.get(k, 0) * w for k, w in SCORE_WEIGHTS.items())
        return round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0

    def parse_llm_json(self, raw: str) -> dict:
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

    async def get_assessment_context(self, jabatan: str, query_mode: str) -> str:
        from lightrag import QueryParam
        if not self.rag:
            return f"Mock Context for Jabatan: {jabatan}"
        try:
            ctx = await self.rag.aquery(
                query=QUERY_KEYWORDS.format(selected_jabatan=jabatan),
                param=QueryParam(
                    mode=query_mode,
                    user_prompt=PROMPT_KONTEKS.format(selected_jabatan=jabatan),
                ),
            )
            return ctx if ctx else "Konteks tidak ditemukan."
        except Exception as e:
            return f"Error retrieving context: {str(e)}"

    async def evaluate_with_llm(self, makalah_text: str, context: str, tema_text: str) -> dict:
        prompt = PROMPT_PENILAIAN.format(
            assessment_context=context,
            tema_text=tema_text,
            makalah_text=makalah_text
        )
        if not self.rag:
            # Mock if RAG is not injected
            return {"Ringkasan": "Mock", "scores": {}, "justification": {}, "evidence": {}, "final_score": 0.0}
            
        eval_response = await self.rag.llm_model_func(prompt)
        return self.parse_llm_json(eval_response)

    async def process_evaluation(self, request: PenilaianRequest) -> PenilaianResponse:
        # 1. Download & Extract Makalah
        makalah_data = self.minio_repo.download_file(self.minio_repo.BUCKET_MAKALAH, request.filename_makalah)
        if not makalah_data:
            raise Exception(f"File makalah {request.filename_makalah} tidak ditemukan di MinIO.")
        makalah_text = DocumentExtractor.extract_from_bytes(makalah_data, request.filename_makalah)

        # 2. Download & Extract Tema
        tema_data = self.minio_repo.download_file(self.minio_repo.BUCKET_TEMA, request.filename_tema)
        if not tema_data:
            raise Exception(f"File tema {request.filename_tema} tidak ditemukan di MinIO.")
        tema_text = DocumentExtractor.extract_from_bytes(tema_data, request.filename_tema)
        if len(tema_text) > 5000:
            tema_text = tema_text[:5000] + "\n\n... (teks dipotong karena terlalu panjang)"

        # 3. Retrieve Context Jabatan via LightRAG
        ctx = await self.get_assessment_context(request.jabatan, request.query_mode)

        # 4. Evaluate via LLM
        result_dict = await self.evaluate_with_llm(makalah_text, ctx, tema_text)
        
        # 5. Compute Score & Formatting
        scores = result_dict.get("scores", {})
        final_score = self.compute_final_score(scores)
        result_dict["final_score"] = final_score

        # 6. Save to DB & MinIO
        self.db_repo.save_evaluation(request.filename_makalah, request.jabatan, result_dict, request.query_mode)
        
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.minio_repo.upload_file(
            self.minio_repo.BUCKET_RIWAYAT,
            f"eval_{ts}_{Path(request.filename_makalah).stem}.json",
            json.dumps(result_dict, ensure_ascii=False, indent=2).encode("utf-8")
        )

        return PenilaianResponse(
            paper_filename=request.filename_makalah,
            jabatan=request.jabatan,
            final_score=final_score,
            scores=scores,
            justification=result_dict.get("justification", {}),
            evidence=result_dict.get("evidence", {}),
            ringkasan=result_dict.get("Ringkasan", ""),
            kelebihan_utama=result_dict.get("kelebihan_utama", []),
            kekurangan_utama=result_dict.get("kekurangan_utama", [])
        )
