import pytest
import os
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.daemon.main import app
from src.daemon.models import JobStatus

client = TestClient(app)

def test_health_check(mock_redis_client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "redis": True}

@patch("src.daemon.main.task_queue")
def test_submit_job(mock_task_queue, mock_redis_client):
    payload = {
        "strategy": "momentum_surge",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "BACKTEST",
        "timeframe": {"start": "2023-01-01", "end": "2023-12-31"}
    }
    response = client.post("/submit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    # Check if status is JobStatus.QUEUED or "QUEUED"
    status = data["status"]
    assert status == "QUEUED" or status == JobStatus.QUEUED.value
    mock_task_queue.enqueue.assert_called_once()

@patch("src.daemon.main.task_queue")
def test_list_jobs(mock_task_queue, mock_redis_client):
    # Submit a job first
    payload = {
        "strategy": "momentum_surge",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "BACKTEST"
    }
    client.post("/submit", json=payload)
        
    response = client.get("/api/v1/jobs")
    assert response.status_code == 200
    jobs = response.json()
    assert len(jobs) >= 1
    assert jobs[0]["strategy_name"] == "momentum_surge"
    # Status can be returned as string or enum value
    status = jobs[0]["status"]
    assert status == "QUEUED" or status == JobStatus.QUEUED.value
