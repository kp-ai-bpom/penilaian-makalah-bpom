# Function helper
# Used
import os
import tempfile
import zipfile
from xml.etree import ElementTree as ET
from pathlib import Path
import pdfplumber
import re
import json

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
VMERGE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val'

def _read_docx(filepath):
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

def _read_pdf(filepath):
    with pdfplumber.open(filepath) as pdf:
        return '\n'.join(
            page.extract_text() for page in pdf.pages
            if page.extract_text()
        )

def read_folder(folder_path):
    results = []
    for path in Path(folder_path).rglob('*'):
        suffix = path.suffix.lower()
        if suffix == '.docx':
            text = _read_docx(path)
        elif suffix == '.pdf':
            text = _read_pdf(path)
        else:
            continue
        results.append({'filename': path.name, 'text': text})
        print(f"✓ {path.name} ({len(text)} karakter)")
    return results


def extract_text_from_file(uploaded_file) -> str:
    """Save uploaded file to temp then read."""
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


# Load SKJ Json
def load_all_skj():
    import json
    skj_folder = "../data/skj_documents_json"

    skj_dict = {}
    for json_file in skj_folder.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

                if "profil_jabatan" in data:
                    jabatan = data["profil_jabatan"].get(
                        "nama_jabatan", json_file.stem
                    )
                elif "nama_jabatan" in data:
                    jabatan = data["nama_jabatan"]
                else:
                    jabatan = json_file.stem

                skj_dict[jabatan] = data

        except Exception as e:
            print(f"Gagal load {json_file.name}: {e}")

    return skj_dict


def parse_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    
    raw = raw.strip()
    # Ekstrak hanya blok JSON
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1:
        raw = raw[start:end+1]
        
    # Hapus trailing commas yang sering bikin json.loads error
    raw = re.sub(r',\s*([\]}])', r'\1', raw)
    
    return json.loads(raw)

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
    "n4_ketajaman_analisis": 2,   # bobot 2x
    "n5_penggunaan_bahasa":  1,
}