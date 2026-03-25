import os
import json
import pytest
from unittest.mock import patch, MagicMock
from src.daemon.tasks import process_job

@patch("src.daemon.tasks.get_current_job")
@patch("src.daemon.tasks.ApplicationController")
def test_worker_claim_state(mock_controller_class, mock_get_job, mock_redis_client):
    mock_job = MagicMock()
    mock_job.connection = mock_redis_client
    mock_get_job.return_value = mock_job

    state_during_execution = {}
    
    def side_effect_execute(*args, **kwargs):
        state_during_execution.update(mock_redis_client.hgetall("job:test-1"))
        return {}

    mock_instance = MagicMock()
    mock_instance.execute_job.side_effect = side_effect_execute
    mock_controller_class.return_value = mock_instance

    process_job("test-1", {"some": "payload"})

    assert state_during_execution.get("status") == "RUNNING"
    assert state_during_execution.get("progress") == "50.0"

@patch("src.daemon.tasks.get_current_job")
@patch("src.daemon.tasks.ApplicationController")
def test_artifact_under_1mb_redis_storage(mock_controller_class, mock_get_job, mock_redis_client):
    mock_job = MagicMock()
    mock_job.connection = mock_redis_client
    mock_get_job.return_value = mock_job
    
    mock_instance = MagicMock()
    mock_instance.execute_job.return_value = {"test": "data"}
    mock_controller_class.return_value = mock_instance
    
    process_job("test-2", {})
    
    job_data = mock_redis_client.hgetall("job:test-2")
    assert job_data["status"] == "COMPLETED"
    assert job_data["progress"] == "100.0"
    assert job_data["artifact_path"] == json.dumps({"test": "data"})

@patch("src.daemon.tasks.get_current_job")
@patch("src.daemon.tasks.ApplicationController")
def test_artifact_over_1mb_disk_spillover(mock_controller_class, mock_get_job, mock_redis_client, managed_artifact_dir):
    mock_job = MagicMock()
    mock_job.connection = mock_redis_client
    mock_get_job.return_value = mock_job
    
    # Generate ~2MB payload to force disk spillover
    large_data = [{"key": "A" * 1000} for _ in range(2000)]
    
    mock_instance = MagicMock()
    mock_instance.execute_job.return_value = large_data
    mock_controller_class.return_value = mock_instance
    
    process_job("test-3", {})
    
    job_data = mock_redis_client.hgetall("job:test-3")
    assert job_data["status"] == "COMPLETED"
    assert job_data["artifact_path"].startswith("FILE_PATH:")

@patch("src.daemon.tasks.get_current_job")
@patch("src.daemon.tasks.ApplicationController")
def test_artifact_over_1mb_file_creation(mock_controller_class, mock_get_job, mock_redis_client, managed_artifact_dir):
    mock_job = MagicMock()
    mock_job.connection = mock_redis_client
    mock_get_job.return_value = mock_job
    
    # Generate ~2MB payload to force disk spillover
    large_data = [{"key": "A" * 1000} for _ in range(2000)]
    
    mock_instance = MagicMock()
    mock_instance.execute_job.return_value = large_data
    mock_controller_class.return_value = mock_instance
    
    process_job("test-4", {})
    
    job_data = mock_redis_client.hgetall("job:test-4")
    file_path = job_data["artifact_path"].split("FILE_PATH:")[1]
    
    assert os.path.exists(file_path)
    with open(file_path, "r") as f:
        loaded = json.load(f)
    assert loaded == large_data

@patch("src.daemon.tasks.get_current_job")
@patch("src.daemon.tasks.ApplicationController")
def test_worker_exception_handling(mock_controller_class, mock_get_job, mock_redis_client):
    mock_job = MagicMock()
    mock_job.connection = mock_redis_client
    mock_get_job.return_value = mock_job
    
    mock_instance = MagicMock()
    mock_instance.execute_job.side_effect = RuntimeError("Data missing")
    mock_controller_class.return_value = mock_instance
    
    process_job("test-5", {})
    
    job_data = mock_redis_client.hgetall("job:test-5")
    assert job_data["status"] == "FAILED"

@patch("src.daemon.tasks.get_current_job")
@patch("src.daemon.tasks.ApplicationController")
def test_worker_traceback_logging(mock_controller_class, mock_get_job, mock_redis_client):
    mock_job = MagicMock()
    mock_job.connection = mock_redis_client
    mock_get_job.return_value = mock_job
    
    mock_instance = MagicMock()
    mock_instance.execute_job.side_effect = RuntimeError("Data missing")
    mock_controller_class.return_value = mock_instance
    
    process_job("test-6", {})
    
    job_data = mock_redis_client.hgetall("job:test-6")
    assert "error_log" in job_data
    assert "RuntimeError: Data missing" in job_data["error_log"]
    assert "Traceback (most recent call last)" in job_data["error_log"]
