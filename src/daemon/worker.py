import time
import threading
import traceback
from sqlalchemy.orm import Session
from .jobs_db import SessionLocal
from .models import JobRegistry, JobStatus
import sys
import os

# Ensure the root dir is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.controller import ApplicationController

MAX_CONCURRENT_RAY_JOBS = 1

def process_job(job_id: str):
    db: Session = SessionLocal()
    job = db.query(JobRegistry).filter(JobRegistry.job_id == job_id).first()
    if not job:
        db.close()
        return

    try:
        job.status = JobStatus.RUNNING
        job.progress = 10.0
        db.commit()
        
        payload = job.parameters
        
        controller = ApplicationController()
        
        job.progress = 50.0
        db.commit()
        
        result = controller.execute_job(payload)
        
        job.status = JobStatus.COMPLETED
        job.progress = 100.0
        # Optional: Save artifact_path or error_log here based on result
        db.commit()
        
    except Exception as e:
        db.rollback()
        job.status = JobStatus.FAILED
        job.error_log = traceback.format_exc()
        db.commit()
    finally:
        db.close()

def worker_loop():
    while True:
        db: Session = SessionLocal()
        
        running_count = db.query(JobRegistry).filter(JobRegistry.status == JobStatus.RUNNING).count()
        
        if running_count < MAX_CONCURRENT_RAY_JOBS:
            next_job = db.query(JobRegistry).filter(JobRegistry.status == JobStatus.QUEUED).first()
            if next_job:
                next_job.status = JobStatus.RUNNING
                db.commit()
                job_id = next_job.job_id
                db.close()
                
                process_job(job_id)
            else:
                db.close()
        else:
            db.close()
            
        time.sleep(2)

def start_worker_thread():
    thread = threading.Thread(target=worker_loop, daemon=True)
    thread.start()
    return thread
