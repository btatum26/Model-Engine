import pytest
from fastapi.testclient import TestClient
import json
import time
from unittest.mock import patch
from src.daemon.main import app

# Create a test client
client = TestClient(app)

def test_health_endpoint(mock_redis_client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "redis": True}

def test_submit_invalid_enum(mock_redis_client):
    payload = {
        "strategy": "dummy",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "MAGIC"
    }
    response = client.post("/submit", json=payload)
    assert response.status_code == 422

@patch("src.daemon.main.task_queue")
def test_submit_success_redis_hash(mock_task_queue, mock_redis_client):
    payload = {
        "strategy": "dummy",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "BACKTEST",
        "timeframe": {"start": "2023-01-01", "end": "2023-12-31"}
    }
    response = client.post("/submit", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "job_id" in data
    job_id = data["job_id"]
    
    # Query Redis
    redis_data = mock_redis_client.hgetall(f"job:{job_id}")
    assert redis_data["status"] == "PENDING" or redis_data["status"] == "QUEUED"
    
    # Assert parameters is a valid serialized JSON string
    parameters_str = redis_data["parameters"]
    parameters_dict = json.loads(parameters_str)
    assert parameters_dict["strategy"] == "dummy"
    mock_task_queue.enqueue.assert_called_once()

@patch("src.daemon.main.task_queue")
def test_submit_success_redis_zset(mock_task_queue, mock_redis_client):
    payload = {
        "strategy": "dummy",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "BACKTEST"
    }
    response = client.post("/submit", json=payload)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    
    # Zscore returns the score (timestamp) if it exists
    score = mock_redis_client.zscore("jobs:all", job_id)
    assert score is not None
    assert isinstance(score, float)
    mock_task_queue.enqueue.assert_called_once()

def test_get_job_404(mock_redis_client):
    response = client.get("/api/v1/jobs/fake-uuid")
    assert response.status_code == 404

def test_get_job_deserialization(mock_redis_client):
    job_id = "test-job-123"
    params = {"key": "value"}
    mapping = {
        "job_id": job_id,
        "parameters": json.dumps(params)
    }
    mock_redis_client.hset(f"job:{job_id}", mapping=mapping)
    
    response = client.get(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["job_id"] == job_id
    # Assert it's a parsed dict, not a string
    assert isinstance(data["parameters"], dict)
    assert data["parameters"]["key"] == "value"

def test_list_jobs_empty_state(mock_redis_client):
    response = client.get("/api/v1/jobs")
    assert response.status_code == 200
    assert response.json() == []

def test_list_jobs_pagination_limit(mock_redis_client):
    # Inject 10 jobs
    for i in range(10):
        job_id = f"job-{i}"
        mock_redis_client.hset(f"job:{job_id}", "job_id", job_id)
        mock_redis_client.zadd("jobs:all", {job_id: float(i)})
        
    response = client.get("/api/v1/jobs?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5

def test_list_jobs_pagination_ordering(mock_redis_client):
    # Inject 3 jobs with staggered timestamps
    mock_redis_client.hset("job:job1", mapping={"job_id": "job1"})
    mock_redis_client.zadd("jobs:all", {"job1": 100.0})
    
    mock_redis_client.hset("job:job2", mapping={"job_id": "job2"})
    mock_redis_client.zadd("jobs:all", {"job2": 200.0})
    
    mock_redis_client.hset("job:job3", mapping={"job_id": "job3"})
    mock_redis_client.zadd("jobs:all", {"job3": 300.0})
    
    response = client.get("/api/v1/jobs")
    assert response.status_code == 200
    data = response.json()
    
    # Should be newest first (descending score)
    assert len(data) == 3
    assert data[0]["job_id"] == "job3"
    assert data[1]["job_id"] == "job2"
    assert data[2]["job_id"] == "job1"
