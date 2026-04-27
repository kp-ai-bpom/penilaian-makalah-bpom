from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base

# Di aplikasi sesungguhnya, Base akan di-import dari app.db.database
Base = declarative_base()

class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    paper_filename = Column(String, index=True)
    jabatan = Column(String, index=True)
    scores = Column(JSON)
    final_score = Column(Float)
    justification = Column(JSON)
    evidence = Column(JSON)
    ringkasan = Column(String)
    query_mode = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class IngestionLog(Base):
    __tablename__ = "ingestion_log"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    status = Column(String)
    error_message = Column(String, nullable=True)
    ingested_at = Column(DateTime(timezone=True), server_default=func.now())
