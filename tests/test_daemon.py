import pytest
import os
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.daemon.main import app
from src.daemon.models import JobStatus

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_submit_job():
    payload = {
        "strategy": "momentum_surge",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "BACKTEST",
        "timeframe": {"start": "2023-01-01", "end": "2023-12-31"}
    }
    # Mock job_executor.submit to avoid actual background process
    with patch("src.daemon.main.job_executor.submit") as mock_submit:
        response = client.post("/submit", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        # Check if status is JobStatus.QUEUED or "QUEUED"
        status = data["status"]
        assert status == "QUEUED" or status == JobStatus.QUEUED.value
        mock_submit.assert_called_once()

def test_list_jobs():
    # Submit a job first
    payload = {
        "strategy": "momentum_surge",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "BACKTEST"
    }
    with patch("src.daemon.main.job_executor.submit"):
        client.post("/submit", json=payload)
        
    response = client.get("/api/v1/jobs")
    assert response.status_code == 200
    jobs = response.json()
    assert len(jobs) >= 1
    assert jobs[0]["strategy_name"] == "momentum_surge"
    # Status can be returned as string or enum value
    status = jobs[0]["status"]
    assert status == "QUEUED" or status == JobStatus.QUEUED.value
