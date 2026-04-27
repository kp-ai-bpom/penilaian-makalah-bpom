import os
import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from .models import EvaluationResult

try:
    import boto3
    from botocore.client import Config
except ImportError:
    boto3 = None

log = logging.getLogger(__name__)

class MinioRepository:
    def __init__(self):
        # Di environment nyata, gunakan settings dari pydantic config (app.core.config)
        self.endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "")
        
        self.BUCKET_MAKALAH = os.getenv("BUCKET_MAKALAH", "makalah")
        self.BUCKET_TEMA = os.getenv("BUCKET_TEMA", "ketentuan-penulisan-makalah")
        self.BUCKET_RIWAYAT = os.getenv("BUCKET_RIWAYAT", "riwayat-penilaian-makalah")

        if boto3:
            self.client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=Config(signature_version="s3v4"),
            )
        else:
            self.client = None
            log.warning("boto3 is not installed. MinIO operations will fail.")

    def ensure_bucket(self, bucket_name: str):
        if not self.client: return
        try:
            existing = [b["Name"] for b in self.client.list_buckets().get("Buckets", [])]
            if bucket_name not in existing:
                self.client.create_bucket(Bucket=bucket_name)
        except Exception as e:
            log.warning(f"Could not ensure bucket '{bucket_name}': {e}")

    def list_files(self, bucket: str) -> List[str]:
        if not self.client: return []
        try:
            resp = self.client.list_objects_v2(Bucket=bucket)
            return [obj["Key"] for obj in resp.get("Contents", [])]
        except Exception:
            return []

    def download_file(self, bucket: str, key: str) -> Optional[bytes]:
        if not self.client: return None
        try:
            obj = self.client.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read()
        except Exception as e:
            log.error(f"MinIO download failed: {e}")
            return None

    def upload_file(self, bucket: str, key: str, data: bytes) -> bool:
        if not self.client: return False
        try:
            self.ensure_bucket(bucket)
            self.client.put_object(Bucket=bucket, Key=key, Body=data)
            return True
        except Exception as e:
            log.error(f"MinIO upload failed: {e}")
            return False


class EvaluationRepository:
    def __init__(self, db_session: Session):
        self.db = db_session

    def save_evaluation(self, paper_filename: str, jabatan: str, result_dict: dict, query_mode: str) -> EvaluationResult:
        if not self.db:
            log.warning("DB Session is None. Skipping save.")
            return None

        scores = result_dict.get("scores", {})
        final_score = result_dict.get("final_score", 0.0)
        justification = result_dict.get("justification", {})
        evidence = result_dict.get("evidence", {})
        ringkasan = result_dict.get("Ringkasan", "")
        
        eval_record = EvaluationResult(
            paper_filename=paper_filename,
            jabatan=jabatan,
            scores=scores,
            final_score=final_score,
            justification=justification,
            evidence=evidence,
            ringkasan=ringkasan,
            query_mode=query_mode
        )
        self.db.add(eval_record)
        self.db.commit()
        self.db.refresh(eval_record)
        return eval_record

    def get_history(self, limit: int = 100) -> List[EvaluationResult]:
        if not self.db: return []
        return self.db.query(EvaluationResult).order_by(EvaluationResult.created_at.desc()).limit(limit).all()
