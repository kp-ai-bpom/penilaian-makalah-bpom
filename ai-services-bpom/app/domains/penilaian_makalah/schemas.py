from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class PenilaianRequest(BaseModel):
    jabatan: str = Field(..., description="Jabatan yang dituju")
    filename_makalah: str = Field(..., description="Nama file makalah di MinIO (bucket: makalah)")
    filename_tema: str = Field(..., description="Nama file ketentuan tema di MinIO (bucket: ketentuan-penulisan-makalah)")
    query_mode: str = Field("hybrid", description="Mode query LightRAG (hybrid, mix, local, global, naive)")

class PenilaianResponse(BaseModel):
    paper_filename: str
    jabatan: str
    final_score: float
    scores: Dict[str, float]
    justification: Dict[str, str]
    evidence: Dict[str, str]
    ringkasan: str
    kelebihan_utama: Optional[List[str]] = []
    kekurangan_utama: Optional[List[str]] = []
