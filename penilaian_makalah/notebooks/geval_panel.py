# ============================================================
# G-EVAL PANEL — Penilaian Makalah BPOM
# ============================================================
# Jalankan tiap section sebagai cell di Jupyter (# %% sebagai pemisah)
# Requirements: pip install deepeval openai python-dotenv

# %% [1] Install & Imports
# !pip install deepeval openai python-dotenv -q

import os
import asyncio
import statistics
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env", override=True)

# %% [2] Konfigurasi 3 Model Panel
GEVAL_API_KEY  = os.getenv("LLM_BINDING_API_KEY")
GEVAL_BASE_URL = os.getenv("LLM_BINDING_HOST", "https://openrouter.ai/api/v1")

PANEL_MODELS = {
    "gpt-4o-mini":     os.getenv("GEVAL_MODEL_A", "openai/gpt-4o-mini"),
    "claude-haiku":    os.getenv("GEVAL_MODEL_B", "anthropic/claude-haiku-4"),
    "gemini-flash":    os.getenv("GEVAL_MODEL_C", "google/gemini-flash-1.5"),
}

SCORE_MIN, SCORE_MAX = 40, 100

print("Panel Models:", PANEL_MODELS)
print("API Base URL:", GEVAL_BASE_URL)
print("API Key set:", bool(GEVAL_API_KEY))

# %% [3] Custom DeepEval Model Wrapper (OpenRouter)
from deepeval.models.base_model import DeepEvalBaseLLM

class OpenRouterGEvalModel(DeepEvalBaseLLM):
    """Wraps any OpenRouter-compatible model for use with deepeval GEval."""

    def __init__(self, model_name: str, alias: str, api_key: str, base_url: str):
        self.model_name = model_name
        self.alias      = alias
        self.api_key    = api_key
        self.base_url   = base_url
        self._client    = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def get_model_name(self) -> str:
        return self.alias

    def load_model(self):
        return self._get_client()

    def generate(self, prompt: str, schema=None) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2048,
        )
        return resp.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2048,
        )
        return resp.choices[0].message.content


# Inisialisasi 3 model judge
panel_judges = {
    alias: OpenRouterGEvalModel(
        model_name=model_name,
        alias=alias,
        api_key=GEVAL_API_KEY,
        base_url=GEVAL_BASE_URL,
    )
    for alias, model_name in PANEL_MODELS.items()
}

print("Panel judges initialized:", list(panel_judges.keys()))

# %% [4] Definisi 5 Kriteria G-Eval (dengan CoT Steps)
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

GEVAL_CRITERIA_DEF = {
    "n1_kesesuaian_judul": {
        "label": "Kesesuaian Judul dengan Tema",
        "criteria": (
            "Nilai sejauh mana judul makalah mencerminkan tema yang telah ditetapkan "
            "dalam ketentuan penulisan makalah. Judul harus spesifik, relevan, dan "
            "mencerminkan isu strategis sesuai konteks jabatan."
        ),
        "evaluation_steps": [
            "Baca judul makalah secara seksama.",
            "Identifikasi tema yang ditetapkan dari bagian INPUT (konteks jabatan & tema).",
            "Periksa apakah kata kunci utama tema muncul atau tercermin dalam judul.",
            "Nilai apakah judul cukup spesifik (bukan terlalu umum/luas).",
            "Tentukan skor: sangat sesuai (tinggi), cukup sesuai (sedang), tidak sesuai (rendah).",
        ],
    },
    "n2_kesesuaian_isi": {
        "label": "Kesesuaian Isi dengan Judul & Tema",
        "criteria": (
            "Nilai sejauh mana isi makalah konsisten dengan judul dan tema yang ditetapkan. "
            "Seluruh pembahasan harus relevan dan tidak menyimpang dari fokus utama."
        ),
        "evaluation_steps": [
            "Identifikasi janji yang dibuat oleh judul makalah.",
            "Baca isi makalah dan identifikasi topik-topik yang dibahas.",
            "Periksa apakah setiap bagian isi relevan dengan judul dan tema.",
            "Identifikasi apakah ada penyimpangan atau pembahasan yang tidak relevan.",
            "Nilai konsistensi dan koherensi antara judul, tema, dan isi secara keseluruhan.",
        ],
    },
    "n3_sistematika": {
        "label": "Sistematika Penulisan",
        "criteria": (
            "Nilai kelengkapan dan ketepatan struktur penulisan makalah: pendahuluan, "
            "tinjauan pustaka/landasan teori, pembahasan, kesimpulan, dan daftar pustaka."
        ),
        "evaluation_steps": [
            "Identifikasi bagian-bagian struktural yang ada dalam makalah.",
            "Periksa apakah pendahuluan mencakup latar belakang dan tujuan.",
            "Periksa apakah ada landasan teori atau tinjauan pustaka yang memadai.",
            "Periksa kelengkapan bagian pembahasan dan kesimpulan.",
            "Nilai alur logis antar bagian dan ketepatan penggunaan sub-judul.",
        ],
    },
    "n4_ketajaman_analisis": {
        "label": "Ketajaman Analisis",
        "criteria": (
            "Nilai kedalaman dan kritis-tidaknya analisis yang disajikan. "
            "Penulis harus menunjukkan pemahaman mendalam tentang permasalahan, "
            "didukung data/fakta, dan dikaitkan dengan konteks BPOM/kebijakan publik."
        ),
        "evaluation_steps": [
            "Identifikasi permasalahan utama yang dianalisis dalam makalah.",
            "Periksa apakah analisis didukung data, fakta, atau referensi yang relevan.",
            "Nilai apakah penulis menunjukkan pemikiran kritis (bukan deskriptif semata).",
            "Periksa apakah ada kaitan dengan isu kebijakan atau konteks jabatan BPOM.",
            "Nilai kedalaman: apakah analisis menjawab 'mengapa' dan 'bagaimana', bukan hanya 'apa'.",
        ],
    },
    "n5_penggunaan_bahasa": {
        "label": "Penggunaan Bahasa",
        "criteria": (
            "Nilai kualitas bahasa yang digunakan: keformalan, ketepatan tata bahasa, "
            "kejelasan ekspresi, konsistensi istilah, dan keterbacaan."
        ),
        "evaluation_steps": [
            "Periksa apakah bahasa yang digunakan formal dan akademis.",
            "Identifikasi adanya kesalahan tata bahasa atau ejaan yang signifikan.",
            "Nilai kejelasan kalimat: apakah mudah dipahami atau ambigu.",
            "Periksa konsistensi penggunaan istilah teknis.",
            "Nilai secara keseluruhan: apakah bahasa mendukung atau menghambat penyampaian ide.",
        ],
    },
}

print("Kriteria G-Eval:", list(GEVAL_CRITERIA_DEF.keys()))


# %% [5] Fungsi: Buat GEval Metric per Kriteria per Model
def build_geval_metric(criteria_key: str, judge_model: OpenRouterGEvalModel) -> GEval:
    defn = GEVAL_CRITERIA_DEF[criteria_key]
    return GEval(
        name=defn["label"],
        criteria=defn["criteria"],
        evaluation_steps=defn["evaluation_steps"],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=judge_model,
        threshold=0.0,  # Kita tidak pakai threshold/pass-fail
    )


# %% [6] Fungsi: Hitung Confidence & Agregasi Panel
def compute_confidence(scores_0_1: list[float]) -> float:
    """Confidence = 1 - normalized_std. Semakin sepaham, semakin tinggi."""
    if len(scores_0_1) < 2:
        return 1.0
    std = statistics.stdev(scores_0_1)
    return round(max(0.0, 1.0 - (std / 0.5)), 3)  # 0.5 = threshold normalisasi


def scale_score(geval_score_0_1: float) -> float:
    """Rescale deepeval score (0–1) ke skala penilaian BPOM (40–100)."""
    return round(SCORE_MIN + geval_score_0_1 * (SCORE_MAX - SCORE_MIN), 1)


def aggregate_panel(per_model_results: dict) -> dict:
    """
    per_model_results: {
        "gpt-4o-mini": {"score_0_1": 0.7, "score_scaled": 82.0, "reason": "..."},
        "claude-haiku": {...},
        "gemini-flash": {...},
    }
    Returns agregasi: mean_score, confidence, consensus_reason.
    """
    scores_0_1 = [v["score_0_1"] for v in per_model_results.values()]
    scores_scaled = [v["score_scaled"] for v in per_model_results.values()]

    mean_0_1   = statistics.mean(scores_0_1)
    mean_scaled = statistics.mean(scores_scaled)
    confidence  = compute_confidence(scores_0_1)

    # Consensus reason = reason dari model dengan skor paling dekat ke mean
    closest_alias = min(
        per_model_results,
        key=lambda a: abs(per_model_results[a]["score_0_1"] - mean_0_1),
    )
    consensus_reason = per_model_results[closest_alias]["reason"]

    return {
        "mean_score_scaled": round(mean_scaled, 1),
        "mean_score_0_1":    round(mean_0_1, 3),
        "confidence":        confidence,
        "model_scores":      {a: v["score_scaled"] for a, v in per_model_results.items()},
        "model_reasons":     {a: v["reason"] for a, v in per_model_results.items()},
        "consensus_reason":  consensus_reason,
    }


# %% [7] Fungsi Utama: Jalankan Panel Evaluasi
def run_geval_panel(
    makalah_text: str,
    jabatan: str,
    tema_text: str,
    assessment_context: str = "",
) -> dict:
    """
    Jalankan G-Eval panel untuk 1 makalah.
    Returns dict lengkap per kriteria + ringkasan.
    """
    import time
    start = time.time()

    # Input context untuk judge model
    input_ctx = (
        f"Jabatan: {jabatan}\n\n"
        f"Tema Makalah:\n{tema_text[:2000]}\n\n"
        f"Konteks Jabatan:\n{assessment_context[:2000]}"
    )

    # Potong makalah jika terlalu panjang (token limit)
    actual_output = makalah_text[:8000] if len(makalah_text) > 8000 else makalah_text

    geval_results = {}

    for crit_key in GEVAL_CRITERIA_DEF:
        print(f"\n  ⏳ Mengevaluasi: {GEVAL_CRITERIA_DEF[crit_key]['label']}")
        per_model = {}

        for alias, judge in panel_judges.items():
            print(f"     [{alias}] ...", end=" ", flush=True)
            try:
                metric = build_geval_metric(crit_key, judge)
                test_case = LLMTestCase(
                    input=input_ctx,
                    actual_output=actual_output,
                )
                metric.measure(test_case)
                score_0_1 = metric.score or 0.0
                per_model[alias] = {
                    "score_0_1":    round(score_0_1, 3),
                    "score_scaled": scale_score(score_0_1),
                    "reason":       metric.reason or "",
                }
                print(f"✓ {scale_score(score_0_1)}")
            except Exception as e:
                print(f"✗ Error: {e}")
                per_model[alias] = {
                    "score_0_1": 0.0,
                    "score_scaled": SCORE_MIN,
                    "reason": f"Error: {str(e)}",
                }

        geval_results[crit_key] = {
            "label": GEVAL_CRITERIA_DEF[crit_key]["label"],
            **aggregate_panel(per_model),
            "per_model": per_model,
        }

    # Hitung final score (dengan bobot)
    WEIGHTS = {
        "n1_kesesuaian_judul":   1,
        "n2_kesesuaian_isi":     1,
        "n3_sistematika":        1,
        "n4_ketajaman_analisis": 2,
        "n5_penggunaan_bahasa":  1,
    }
    total_w = sum(WEIGHTS.values())
    weighted_sum = sum(
        geval_results[k]["mean_score_scaled"] * WEIGHTS[k]
        for k in WEIGHTS
        if k in geval_results
    )
    final_score = round(weighted_sum / total_w, 1)

    overall_confidence = round(
        statistics.mean(geval_results[k]["confidence"] for k in geval_results), 3
    )

    duration = round(time.time() - start, 1)

    return {
        "jabatan":            jabatan,
        "geval_final_score":  final_score,
        "geval_confidence":   overall_confidence,
        "panel_models":       list(PANEL_MODELS.keys()),
        "duration_sec":       duration,
        "criteria":           geval_results,
    }


# %% [8] Test dengan Makalah Sample
SAMPLE_MAKALAH = """
JUDUL: Strategi Penguatan Pengawasan Obat dan Makanan dalam Mendukung Rencana Strategis BPOM 2025–2029

BAB I PENDAHULUAN
Badan Pengawas Obat dan Makanan (BPOM) memiliki peran strategis dalam melindungi masyarakat 
dari risiko obat dan makanan yang tidak memenuhi syarat. Dalam konteks Renstra BPOM 2025–2029, 
penguatan pengawasan menjadi prioritas utama untuk meningkatkan kepercayaan publik.

BAB II TINJAUAN PUSTAKA
Pengawasan obat dan makanan di Indonesia diatur dalam UU No. 18 Tahun 2012 tentang Pangan 
dan Peraturan BPOM Nomor 2 Tahun 2024. Standar internasional seperti Codex Alimentarius 
menjadi acuan dalam penetapan batas keamanan produk.

BAB III PEMBAHASAN
Tantangan pengawasan saat ini meliputi: (1) masih tingginya peredaran produk ilegal, 
(2) keterbatasan sumber daya manusia di daerah, dan (3) perkembangan teknologi produksi 
yang pesat. Strategi yang diusulkan mencakup digitalisasi sistem pengawasan berbasis risiko, 
penguatan kapasitas Balai BPOM di seluruh Indonesia, serta kolaborasi lintas kementerian/lembaga.

BAB IV KESIMPULAN
Penguatan pengawasan obat dan makanan memerlukan pendekatan komprehensif yang melibatkan 
teknologi, regulasi, dan SDM yang kompeten. Implementasi Renstra 2025–2029 harus disertai 
evaluasi berkala untuk memastikan capaian target nasional.

DAFTAR PUSTAKA
1. UU No. 18 Tahun 2012 tentang Pangan
2. Peraturan BPOM Nomor 2 Tahun 2024
3. Renstra BPOM 2025–2029
"""

SAMPLE_JABATAN = "Kepala Biro SDM"
SAMPLE_TEMA    = "Strategi Penguatan Kelembagaan BPOM dalam Mendukung Renstra 2025–2029"
SAMPLE_CONTEXT = "Jabatan Kepala Biro SDM bertanggung jawab atas pengelolaan sumber daya manusia BPOM."

print("Sample makalah siap, panjang:", len(SAMPLE_MAKALAH), "karakter")


# %% [9] Jalankan Panel Evaluasi
print("=" * 60)
print("MEMULAI G-EVAL PANEL (3 MODEL)")
print("=" * 60)

results = run_geval_panel(
    makalah_text=SAMPLE_MAKALAH,
    jabatan=SAMPLE_JABATAN,
    tema_text=SAMPLE_TEMA,
    assessment_context=SAMPLE_CONTEXT,
)

print(f"\n{'='*60}")
print(f"G-Eval Final Score : {results['geval_final_score']}")
print(f"Overall Confidence : {results['geval_confidence']:.1%}")
print(f"Duration           : {results['duration_sec']}s")
print(f"{'='*60}")


# %% [10] Tampilkan Hasil Lengkap
try:
    import pandas as pd

    rows = []
    for k, v in results["criteria"].items():
        row = {"Kriteria": v["label"], "Confidence": f"{v['confidence']:.0%}"}
        for alias in results["panel_models"]:
            row[alias] = v["model_scores"].get(alias, "-")
        row["Mean Score"] = v["mean_score_scaled"]
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n📊 PANEL COMPARISON TABLE")
    print(df.to_string(index=False))
except ImportError:
    print(json.dumps(results, indent=2, ensure_ascii=False))


# %% [11] Tampilkan CoT Reasoning per Kriteria
print("\n📝 REASONING PER KRITERIA")
print("-" * 60)
for k, v in results["criteria"].items():
    print(f"\n🔹 {v['label']} (Score: {v['mean_score_scaled']} | Confidence: {v['confidence']:.0%})")
    print(f"   Consensus Reason: {v['consensus_reason'][:300]}...")
    for alias, reason in v["model_reasons"].items():
        print(f"   [{alias}]: {reason[:200]}...")


# %% [12] Simpan Hasil ke JSON
import datetime
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"./output/geval_result_{ts}.json"
os.makedirs("./output", exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n✅ Hasil disimpan ke: {output_path}")
