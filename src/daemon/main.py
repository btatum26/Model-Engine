import os
import json
import socket
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from .jobs_db import SessionLocal, engine, Base
from .models import JobRegistry, JobStatus
from .worker import start_worker_thread

TRANSIT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../transit"))
BEACON_FILE = os.path.join(TRANSIT_DIR, "api_beacon.json")
PORT = 8000

Base.metadata.create_all(bind=engine)

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(TRANSIT_DIR, exist_ok=True)
    api_url = f"http://{get_local_ip()}:{PORT}"
    
    beacon_data = {"api_url": api_url, "status": "online"}
    with open(BEACON_FILE, "w") as f:
        json.dump(beacon_data, f)
    
    print(f"🚀 Daemon Online. Beacon written to {BEACON_FILE} -> {api_url}")
    
    worker_thread = start_worker_thread()
    
    yield
    
    beacon_data["status"] = "offline"
    with open(BEACON_FILE, "w") as f:
        json.dump(beacon_data, f)
    print("🛑 Daemon Offline. Beacon updated.")

app = FastAPI(title="Vectorized Alpha Engine API", lifespan=lifespan)

def get_db():
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

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/submit")
def submit_job(payload: JobPayloadRequest, db: Session = Depends(get_db)):
    job = JobRegistry(
        strategy_name=payload.strategy,
        status=JobStatus.QUEUED,
        progress=0.0,
        parameters=payload.model_dump()
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return {"job_id": job.job_id, "status": job.status}

@app.get("/api/v1/jobs")
def list_jobs(db: Session = Depends(get_db)):
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
