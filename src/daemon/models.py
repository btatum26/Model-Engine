import enum
import uuid
from sqlalchemy import Column, String, Float, Enum, JSON
from .jobs_db import Base

class JobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class JobRegistry(Base):
    __tablename__ = "job_registry"

    job_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_name = Column(String, nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.QUEUED)
    progress = Column(Float, default=0.0)
    parameters = Column(JSON, default=dict)
    artifact_path = Column(String, nullable=True)
    error_log = Column(String, nullable=True)
