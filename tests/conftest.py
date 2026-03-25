import os
import sys
import shutil
import pytest
import pandas as pd
import fakeredis
from rq import Queue
from unittest.mock import patch, MagicMock

# 1.1 Custom Pytest Hooks
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # Execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()
    # Set an attribute on the item to indicate failure
    if rep.when == "call" and rep.failed:
        item.test_failed = True

# 1.2 Global Mock Fixtures
@pytest.fixture
def mock_redis_client():
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    with patch("src.daemon.main.redis_client", fake_client), \
         patch("src.daemon.tasks.redis_client", fake_client):
        yield fake_client

@pytest.fixture
def mock_rq_queue(mock_redis_client):
    fake_queue = Queue('default', connection=mock_redis_client, is_async=False)
    with patch("src.daemon.main.task_queue", fake_queue):
        yield fake_queue

@pytest.fixture
def mock_data_broker():
    # A static Pandas DataFrame fixture to ensure zero network calls to yfinance
    df = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0],
        "High": [105.0, 106.0, 107.0],
        "Low": [95.0, 96.0, 97.0],
        "Close": [102.0, 103.0, 104.0],
        "Volume": [1000, 1100, 1200]
    }, index=pd.date_range("2023-01-01", periods=3))
    
    with patch("src.data_broker.data_broker.DataBroker.get_data", return_value=df):
        yield df

# 1.3 Conditional Artifact Management Fixture
@pytest.fixture
def managed_artifact_dir(request):
    test_name = request.node.name
    artifact_dir = os.path.abspath(f"./tests/failed_artifacts/{test_name}")
    os.makedirs(artifact_dir, exist_ok=True)
    
    with patch("src.daemon.tasks.ARTIFACT_DIR", artifact_dir):
        yield artifact_dir
        
    if not getattr(request.node, "test_failed", False):
        # Pass
        shutil.rmtree(artifact_dir, ignore_errors=True)
    else:
        # Fail
        print(f"\nArtifacts preserved at {artifact_dir}")

# 1.4 Strategy Fixtures
@pytest.fixture
def dummy_strategy(tmp_path):
    strat_dir = tmp_path / "dummy_strat"
    strat_dir.mkdir()
    
    # manifest.json
    manifest_content = """{
        "name": "Dummy Strategy",
        "description": "A dummy strategy for testing",
        "features": [{"id": "feature_1"}],
        "parameter_bounds": {
            "window": [10, 20],
            "threshold": [0.1, 0.2]
        }
    }"""
    (strat_dir / "manifest.json").write_text(manifest_content)
    
    # context.py
    context_content = """class Context:
    def __init__(self):
        self.mapping = {}
"""
    (strat_dir / "context.py").write_text(context_content)
    
    # model.py
    model_content = """import pandas as pd
from src.controller import SignalModel

class DummyModel(SignalModel):
    def train(self, df, context, params):
        return {"trained": True}
        
    def generate_signals(self, df, context, params, artifacts):
        # Deterministic alternating signals
        signals = [1.0, -1.0] * (len(df) // 2) + [1.0] * (len(df) % 2)
        return pd.Series(signals, index=df.index, name="signal")
"""
    (strat_dir / "model.py").write_text(model_content)
    
    return str(strat_dir)
