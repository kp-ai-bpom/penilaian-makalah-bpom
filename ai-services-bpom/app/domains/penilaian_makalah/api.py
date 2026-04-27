from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from sqlalchemy.orm import Session
# from app.db.database import get_db
# from app.core.llm import get_rag

from .schemas import PenilaianRequest, PenilaianResponse
from .services import PenilaianService
from .repositories import MinioRepository, EvaluationRepository

router = APIRouter(prefix="/penilaian-makalah", tags=["Penilaian Makalah"])

# MOCK DEPENDENCIES (untuk mensimulasikan integrasi dengan core)
def get_db():
    yield None

def get_rag():
    return None

def get_penilaian_service(db: Session = Depends(get_db), rag = Depends(get_rag)) -> PenilaianService:
    minio_repo = MinioRepository()
    db_repo = EvaluationRepository(db)
    return PenilaianService(db_repo, minio_repo, rag)

@router.get("/tema", response_model=List[str])
async def get_tema_list(service: PenilaianService = Depends(get_penilaian_service)):
    """Mengembalikan daftar file ketentuan tema dari MinIO"""
    # Menggunakan properti BUCKET_TEMA yang diatur di repository
    return service.minio_repo.list_files(service.minio_repo.BUCKET_TEMA)

@router.get("/makalah", response_model=List[str])
async def get_makalah_list(service: PenilaianService = Depends(get_penilaian_service)):
    """Mengembalikan daftar file makalah dari MinIO"""
    return service.minio_repo.list_files(service.minio_repo.BUCKET_MAKALAH)

@router.post("/evaluate", response_model=PenilaianResponse)
async def evaluate_makalah(request: PenilaianRequest, service: PenilaianService = Depends(get_penilaian_service)):
    """Menjalankan proses evaluasi makalah dengan LLM"""
    try:
        response = await service.process_evaluation(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_history(limit: int = 100, service: PenilaianService = Depends(get_penilaian_service)):
    """Mengambil riwayat penilaian dari database PostgreSQL"""
    history = service.db_repo.get_history(limit=limit)
    return history
