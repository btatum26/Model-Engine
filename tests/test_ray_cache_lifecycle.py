import pytest
import ray
import pandas as pd
import numpy as np
import sys
import os
import time
import shutil
from src.optimization.local_cache import LocalCache
from src.optimization.optimizer_core import evaluate_parameters_cpu

def test_ray_cache_lifecycle():
    if not ray.is_initialized():
        ray.init(num_cpus=1, ignore_reinit_error=True)
    
    cache = LocalCache()
    df = pd.DataFrame({"close": np.random.randn(100)})
    dataset_ref = "test_data_lifecycle"
    
    # Test A - Insertion & Retrieval
    ref = cache.load_to_ram(dataset_ref, df)
    assert isinstance(ref, ray.ObjectRef)
    
    retrieved_df = ray.get(ref)
    pd.testing.assert_frame_equal(df, retrieved_df)
    
    # Test B - Cache Clearance
    cache.clear_cache(dataset_ref)
    assert cache.get_ref(dataset_ref) is None
    
    ray.shutdown()

def test_worker_namespace_isolation():
    if not ray.is_initialized():
        ray.init(num_cpus=1, ignore_reinit_error=True)
    
    # Create a dummy strategy path
    strat_path = os.path.abspath("tests/dummy_strat")
    os.makedirs(strat_path, exist_ok=True)
    # Create model.py that fails
    with open(os.path.join(strat_path, "model.py"), "w") as f:
        f.write("class DummyModel:\n    def generate_signals(self, df, params):\n        raise ValueError('Forced Failure')")
    
    df = pd.DataFrame({"close": np.random.randn(100)})
    data_ref = ray.put(df)
    
    # Run the worker which should fail
    result = ray.get(evaluate_parameters_cpu.remote(data_ref, {}, [], strat_path))
    assert "Forced Failure" in result.get("error", "")
    
    # Check that sys.path does not contain strat_path in a subsequent task on the same worker
    @ray.remote(num_cpus=1)
    def check_sys_path(path_to_check):
        import sys
        return path_to_check in sys.path
    
    # Give some time for the worker to be available
    time.sleep(1)
    
    is_present = ray.get(check_sys_path.remote(strat_path))
    assert not is_present, f"Strategy path {strat_path} leaked into worker sys.path after failure"
    
    # Cleanup
    shutil.rmtree(strat_path)
    ray.shutdown()
