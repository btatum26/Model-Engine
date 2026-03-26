import json
import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import redis
from rq import Queue

from src.engine.controller import ExecutionMode
from .models import JobRegistry, JobStatus
from ..logger import logger, daemon_logger
from ..config import config

# Global variables for our connections
redis_pool = None
redis_client = None
task_queue = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the API server."""
    global redis_pool, redis_client, task_queue
    
    logger.info(f"API Online at {config.api_url}")
    daemon_logger.info("FastAPI Server starting up... Connecting to Redis.")
    
    # Initialize Redis Connection Pool
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_pool = redis.ConnectionPool.from_url(redis_url, decode_responses=True)
    redis_client = redis.Redis(connection_pool=redis_pool)
    task_queue = Queue('default', connection=redis_client)
    
    # Ensure connection is valid
    redis_client.ping()
    daemon_logger.info("Connected to Redis successfully.")
    
    yield
    
    redis_pool.disconnect()
    logger.info("API Offline.")
    daemon_logger.info("FastAPI Server shut down. Redis disconnected.")

app = FastAPI(title="Research Engine API", lifespan=lifespan)

class TimeframeRequest(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None

class JobPayloadRequest(BaseModel):
    strategy: str
    assets: List[str]
    interval: str
    mode: ExecutionMode
    timeframe: Optional[TimeframeRequest] = None
    multi_asset_mode: Optional[str] = "BATCH"

@app.get("/health")
def health_check():
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis client not initialized")
    return {"status": "ok", "redis": redis_client.ping()}

@app.post("/submit")
def submit_job(payload: JobPayloadRequest):
    """Save job state to Redis and enqueue for background execution."""
    try:
        if task_queue is None:
            raise HTTPException(status_code=503, detail="Task queue not initialized")
        if redis_client is None:
            raise HTTPException(status_code=503, detail="Redis client not initialized")
        
        job = JobRegistry(
            strategy_name=payload.strategy,
            parameters=payload.model_dump()
        )
        job_id = job.job_id
        
        # Create the Redis Hash for job state
        # We must serialize nested dicts (like parameters) to strings for Redis Hashes
        mapping = {
            "job_id": job_id,
            "strategy_name": job.strategy_name,
            "status": job.status.value,
            "progress": str(job.progress),
            "parameters": json.dumps(job.parameters),
            "created_at": str(time.time())
        }
        redis_client.hset(f"job:{job_id}", mapping=mapping)
        
        # Add to the Sorted Set (ZSET) for ordered pagination
        # Score is the current timestamp
        redis_client.zadd("jobs:all", {job_id: time.time()})
        
        # Enqueue the task to RQ
        # We pass the job_id so the worker knows which Redis Hash to update
        from .tasks import process_job  # Import here to avoid circular dependencies
        task_queue.enqueue(
            process_job, 
            args=(job_id, job.parameters), 
            job_id=job_id,  # Forces RQ to use your UUID
            job_timeout='1h'
        )
        
        daemon_logger.info(f"Job Queued: {job_id} for strategy {payload.strategy}")
        return {"job_id": job_id, "status": job.status}
        
    except Exception as e:
        daemon_logger.error(f"Failed to submit job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str):
    """Fetch a single job's status from Redis."""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis client not initialized")
    
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Deserialize parameters back to dict for the API response
    if "parameters" in job_data:
        job_data["parameters"] = json.loads(job_data["parameters"])
        
    return job_data

@app.get("/api/v1/jobs")
def list_jobs(limit: int = 50, offset: int = 0):
    """Return an ordered list of jobs using the Redis ZSET."""
    try:
        if redis_client is None:
            raise HTTPException(status_code=503, detail="Redis client not initialized")
        
        # ZREVRANGE returns the newest jobs first
        job_ids = redis_client.zrevrange("jobs:all", offset, offset + limit - 1)
        
        if not job_ids:
            return []
            
        # Use a Redis pipeline to fetch all Hashes efficiently in one round-trip
        pipe = redis_client.pipeline()
        for j_id in job_ids:
            pipe.hgetall(f"job:{j_id}")
        jobs_data = pipe.execute()
        
        # Format for output
        formatted_jobs = []
        for job in jobs_data:
            if job:
                if "parameters" in job:
                    job["parameters"] = json.loads(job["parameters"])
                formatted_jobs.append(job)
                
        return formatted_jobs
    except Exception as e:
        daemon_logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")