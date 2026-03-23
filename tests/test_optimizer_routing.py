import pytest
from src.optimization.ray_cluster import RayClusterManager
from src.optimization.local_cache import LocalCache
from src.optimization.optimizer_core import OptimizerCore
import ray
import pandas as pd
from unittest.mock import MagicMock

def test_ray_cluster_manager_init():
    manager = RayClusterManager(reserve_cpus=1)
    assert manager.reserve_cpus == 1

def test_local_cache_operations():
    # We need Ray for this
    if not ray.is_initialized():
        ray.init(num_cpus=1, ignore_reinit_error=True)
    
    cache = LocalCache()
    df = pd.DataFrame({"close": [100, 101, 102]})
    dataset_ref = "test_data"
    
    ref = cache.load_to_ram(dataset_ref, df)
    assert isinstance(ref, ray.ObjectRef)
    
    get_ref = cache.get_ref(dataset_ref)
    assert get_ref == ref
    
    cache.clear_cache(dataset_ref)
    assert cache.get_ref(dataset_ref) is None
    
    ray.shutdown()

def test_optimizer_core_instantiation():
    manifest = {
        "hyperparameters": {"rsi_lower": [20, 30], "rsi_upper": [70, 80]},
        "features": [{"id": "RSI_14", "type": "momentum", "params": {"period": 14}}]
    }
    optimizer = OptimizerCore(
        strategy_path="src/strategies/momentum_surge",
        dataset_ref="AAPL_1h",
        manifest=manifest,
        ticker="AAPL",
        interval="1h"
    )
    assert optimizer.ticker == "AAPL"
    assert optimizer.interval == "1h"
    assert optimizer.manifest == manifest

def test_optimizer_circuit_breaker_routing():
    # Mocking dependencies to test routing logic only
    manifest_small = {
        "hyperparameters": {"p1": list(range(20)), "p2": list(range(40))}, # 800 permutations
    }
    manifest_large = {
        "hyperparameters": {"p1": list(range(50)), "p2": list(range(25))}, # 1250 permutations
    }
    
    optimizer_small = OptimizerCore("path", "ref", manifest_small)
    optimizer_large = OptimizerCore("path", "ref", manifest_large)
    
    # Use mocks for the search methods
    optimizer_small._run_grid_search = MagicMock(return_value={"p1": 0, "p2": 0})
    optimizer_small._run_optuna_search = MagicMock()
    
    optimizer_large._run_grid_search = MagicMock()
    optimizer_large._run_optuna_search = MagicMock(return_value={"p1": 0, "p2": 0})
    
    # Dummy data_ref
    data_ref = MagicMock()
    
    optimizer_small._phase_a_discovery(data_ref)
    optimizer_small._run_grid_search.assert_called_once()
    optimizer_small._run_optuna_search.assert_not_called()
    
    optimizer_large._phase_a_discovery(data_ref)
    optimizer_large._run_optuna_search.assert_called_once()
    optimizer_large._run_grid_search.assert_not_called()
