import os
import json
import socket
import logging
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor

from .jobs_db import SessionLocal, engine, Base
from .models import JobRegistry, JobStatus
from ..logger import logger, daemon_logger
from ..exceptions import EngineError
from ..config import config

# Initialize database schema
Base.metadata.create_all(bind=engine)

# Limit execution to one job at a time to prevent resource exhaustion
job_executor = ProcessPoolExecutor(max_workers=1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the API server."""
    # High-level status to console
    logger.info(f"API Online at {config.api_url}")
    # Detailed log to file
    daemon_logger.info("FastAPI Server starting up...")
    
    yield
    
    job_executor.shutdown(wait=False)
    logger.info("API Offline.")
    daemon_logger.info("FastAPI Server shut down.")

app = FastAPI(title="Research Engine API", lifespan=lifespan)

def get_db():
    """Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class TimeframeRequest(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None

class JobPayloadRequest(BaseModel):
    strategy: str
    assets: List[str]
    interval: str
    mode: str
    timeframe: Optional[TimeframeRequest] = None
    multi_asset_mode: Optional[str] = "BATCH"

def run_job_in_executor(job_id: str):
    """Background process for job execution."""
    import sys
    import os
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        
    from src.daemon.jobs_db import SessionLocal
    from src.daemon.models import JobRegistry, JobStatus
    from src.controller import ApplicationController
    
    db = SessionLocal()
    job = db.query(JobRegistry).filter(JobRegistry.job_id == job_id).first()
    if not job:
        db.close()
        return

    try:
        daemon_logger.info(f"Starting Job execution: {job_id}")
        job.status = JobStatus.RUNNING
        job.progress = 10.0
        db.commit()
        
        payload = job.parameters
        daemon_logger.info(f"Job payload: {payload}")
        controller = ApplicationController()
        
        job.progress = 50.0
        db.commit()
        
        result = controller.execute_job(payload)
        
        # Re-query to prevent session detach issues
        job = db.query(JobRegistry).filter(JobRegistry.job_id == job_id).first()
        job.status = JobStatus.COMPLETED
        job.progress = 100.0
        job.artifact_path = json.dumps(result) if result else None
        db.commit()
        daemon_logger.info(f"Job completed successfully: {job_id}")
        
    except Exception as e:
        db.rollback()
        daemon_logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        job = db.query(JobRegistry).filter(JobRegistry.job_id == job_id).first()
        if job:
            job.status = JobStatus.FAILED
            job.error_log = traceback.format_exc()
            db.commit()
    finally:
        db.close()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/submit")
def submit_job(payload: JobPayloadRequest, db: Session = Depends(get_db)):
    """Add a job to the queue for background execution."""
    try:
        job = JobRegistry(
            strategy_name=payload.strategy,
            status=JobStatus.QUEUED,
            progress=0.0,
            parameters=payload.model_dump()
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        
        daemon_logger.info(f"Job Queued: {job.job_id} for strategy {payload.strategy}")
        job_executor.submit(run_job_in_executor, job.job_id)
        return {"job_id": job.job_id, "status": job.status}
    except Exception as e:
        daemon_logger.error(f"Failed to submit job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/jobs")
def list_jobs(db: Session = Depends(get_db)):
    """Return all jobs in the database."""
    try:
        jobs = db.query(JobRegistry).order_by(JobRegistry.job_id.desc()).all()
        return [
            {
                "job_id": j.job_id,
                "strategy_name": j.strategy_name,
                "status": j.status.value if hasattr(j.status, 'value') else j.status,
                "progress": j.progress,
                "parameters": j.parameters,
                "artifact_path": j.artifact_path,
                "error_log": j.error_log
            }
            for j in jobs
        ]
    except Exception as e:
        daemon_logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
