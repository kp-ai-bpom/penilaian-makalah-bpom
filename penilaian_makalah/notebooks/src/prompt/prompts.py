# ── Prompt ────────────────────────────────────────────────────────────────────
QUERY_KEYWORDS = """
Standar Kompetensi Jabatan {selected_jabatan}
Kompetensi Teknis {selected_jabatan}
Kompetensi Manajerial {selected_jabatan}
Kompetensi Sosial Kultural {selected_jabatan}
Indikator perilaku kompetensi {selected_jabatan}
Persyaratan jabatan {selected_jabatan}
Tugas pokok dan fungsi {selected_jabatan}
Penilaian Penulisan Makalah
Form. 1 Penilaian Penulisan Makalah
Kriteria penilaian makalah seleksi kompetensi bidang
Ketajaman analisis kompetensi bidang
Sistematika penulisan makalah JPT
Kesesuaian isi makalah dengan tema jabatan
PEMETAAN VISI MISI TUJUAN STRATEGI SASARAN BPOM
PEMETAAN ARAH KEBIJAKAN DAN STRATEGI BPOM
MATRIKS RINGKASAN ANALISIS SWOT BPOM
Rencana Strategis BPOM
"""

PROMPT_KONTEKS = """
---Role---
Anda adalah asisten persiapan konteks yang bertugas merangkum informasi relevan
dari knowledge graph untuk membantu evaluator menilai makalah seleksi.
 
---Goal---
Berdasarkan context yang diberikan dari knowledge graph SKJ BPOM, susun ringkasan
kontekstual yang terstruktur untuk jabatan '{selected_jabatan}' guna mendukung
penilaian makalah seleksi kompetensi bidang.
 
---Batasan Penting---
- Gunakan HANYA informasi yang tersedia dalam context yang diberikan.
- Jangan menambahkan informasi dari pengetahuan umum Anda di luar context.
- Jika informasi tertentu tidak tersedia dalam context, nyatakan secara eksplisit.
 
---Instructions---
Berdasarkan context yang tersedia, susun ringkasan dalam format berikut:
 
## 1. Profil Jabatan
Deskripsikan peran, tanggung jawab utama, dan posisi jabatan '{selected_jabatan}'
dalam struktur organisasi BPOM.
 
## 2. Kompetensi yang Dipersyaratkan
Jabarkan kompetensi teknis, manajerial, dan sosial kultural yang diperlukan,
beserta level yang dipersyaratkan (skala 1-5) jika tersedia dalam context.
 
## 3. Relevansi Kebijakan Strategis
Uraikan arah kebijakan, strategi BPOM, dan isu-isu strategis yang relevan
dengan jabatan ini berdasarkan Renstra BPOM yang tersedia di context.
 
## 4. Ekspektasi Kualitas Makalah
Berdasarkan profil jabatan dan kompetensi di atas, jelaskan apa yang diharapkan
dari makalah berkualitas tinggi untuk posisi ini — khususnya dari sisi:
- Kedalaman analisis masalah yang relevan
- Kelayakan solusi/konsep yang ditawarkan
- Keterkaitan dengan kebijakan dan strategi BPOM
 
## 5. Keterbatasan Konteks
Sebutkan aspek-aspek yang tidak ditemukan dalam context sehingga evaluator
perlu mengandalkan penilaian substantif dari teks makalah.
 
---Context dari Knowledge Graph---
{retrieved_context}
"""


PROMPT_PENILAIAN = """
---Role---

Anda adalah evaluator akademik sebagai Panitia Seleksi yang bertugas menilai kualitas substansi makalah secara objektif dan sistematis. Penilaian harus didasarkan hanya pada isi makalah yang tersedia, dengan mempertimbangkan konteks jabatan yang dituju.

---Goal---

Melakukan penilaian terhadap makalah berdasarkan kriteria penilaian yang telah ditentukan, memberikan skor numerik untuk setiap kriteria.

---Konteks Jabatan---

{assessment_context}

---Tema Makalah---

{tema_text}

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
- Nilai Akhir = (n1 + n2 + n3 + (2 × n4) + n5) / 6

---Score Criteria---
- Sangat baik: skor >= 90
- Baik: skor >= 76
- Cukup: skor < 76

---Instructions---

1. Baca dan pahami isi makalah secara menyeluruh.
2. Jika isi makalah tidak relevan dengan tema atau konteks jabatan, berikan skor rendah sesuai dengan kriteria penilaian.
3. Tinjau konteks jabatan di atas sebagai acuan penilaian.
4. Lakukan penilaian terhadap setiap kriteria dengan menggunakan panduan di atas.
5. Penilaian harus objektif, sistematis, dan berbasis isi makalah.
6. Gunakan bahasa formal dan akademik.
7. Jangan menggunakan informasi di luar isi makalah.
8. Hitung nilai akhir menggunakan dengan menggunakan panduan di atas.
9. Output harus dalam format JSON yang valid dan tidak boleh mengandung teks tambahan di luar JSON.


---Makalah---

{makalah_text}

---Output Format---

Hasil harus dalam format JSON berikut:
{{
  "Ringkasan": "ringkasan isi makalah yang menjelaskan tentang keseluruhan makalah",

  "scores": {{
    "n1": 0,
    "n2": 0,
    "n3": 0,
    "n4": 0,
    "n5": 0
  }},

  "justification": {{
    "n1": "justifikasi skor untuk kriteria n1",
    "n2": "justifikasi skor untuk kriteria n2",
    "n3": "justifikasi skor untuk kriteria n3",
    "n4": "justifikasi skor untuk kriteria n4",
    "n5": "justifikasi skor untuk kriteria n5"
  }},
  "evidence": {{
    "n1": "evidence dari makalah yang mendukung skor n1 dan justifikasi n1",
    "n2": "evidence dari makalah yang mendukung skor n2 dan justifikasi n2",
    "n3": "evidence dari makalah yang mendukung skor n3 dan justifikasi n3",
    "n4": "evidence dari makalah yang mendukung skor n4 dan justifikasi n4",
    "n5": "evidence dari makalah yang mendukung skor n5 dan justifikasi n5"
  }},

  "final_score": 0
}}
"""