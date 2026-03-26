import os
import sys
import json
import traceback
import redis
from rq import get_current_job
from .models import JobStatus

# Ensure the root directory is in the path so we can import src.engine.controller
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.engine.controller import ApplicationController
from src.logger import daemon_logger

# Connect to Redis synchronously for the worker
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# 1MB limit in bytes
MAX_ARTIFACT_SIZE_BYTES = 1024 * 1024 
ARTIFACT_DIR = os.path.join(root_dir, "artifacts")

def process_job(job_id: str, payload: dict):
    """
    The background task executed by RQ. 
    Updates the Redis Hash directly to maintain state.
    """
    # Get the safe, per-process Redis connection RQ is already using
    job = get_current_job()
    redis_client = job.connection
    
    job_key = f"job:{job_id}"
    
    try:
        daemon_logger.info(f"Worker claimed Job: {job_id}")
        
        # Update state to RUNNING
        redis_client.hset(job_key, mapping={
            "status": JobStatus.RUNNING.value,
            "progress": "10.0"
        })
        
        controller = ApplicationController()
        
        # Simulate mid-job progress update
        redis_client.hset(job_key, "progress", "50.0")
        
        # Execute the actual work
        result = controller.execute_job(payload)
        
        # Handle the artifact and the 1MB size limit
        artifact_value = ""
        if result:
            result_str = json.dumps(result)
            result_bytes = result_str.encode('utf-8')
            
            if len(result_bytes) > MAX_ARTIFACT_SIZE_BYTES:
                # Save to disk
                os.makedirs(ARTIFACT_DIR, exist_ok=True)
                file_path = os.path.join(ARTIFACT_DIR, f"{job_id}.json")
                with open(file_path, "w") as f:
                    f.write(result_str)
                artifact_value = f"FILE_PATH:{file_path}"
                daemon_logger.info(f"Artifact > 1MB. Saved to disk: {file_path}")
            else:
                # Save directly in Redis
                artifact_value = result_str
        
        # Update to COMPLETED
        redis_client.hset(job_key, mapping={
            "status": JobStatus.COMPLETED.value,
            "progress": "100.0",
            "artifact_path": artifact_value
        })
        
        daemon_logger.info(f"Job completed successfully: {job_id}")
        
    except Exception as e:
        error_log = traceback.format_exc()
        daemon_logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        
        # Mark as FAILED and save traceback
        redis_client.hset(job_key, mapping={
            "status": JobStatus.FAILED.value,
            "error_log": error_log
        })