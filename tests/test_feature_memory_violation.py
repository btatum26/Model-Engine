import pytest
import pandas as pd
import numpy as np
from src.features.base import Feature, FeatureResult, register_feature
from src.features.features import compute_all_features

@register_feature("MemoryViolator")
class MemoryViolator(Feature):
    @property
    def name(self): return "MemoryViolator"
    @property
    def description(self): return "Violates memory"
    @property
    def category(self): return "Test"

    def compute(self, df: pd.DataFrame, params: dict, cache) -> FeatureResult:
        # This should trigger the Blast Shield if df is read-only
        df["violator"] = df["close"] * 2
        return FeatureResult(data={"violator": df["violator"]}, visuals=[])

def test_feature_memory_violation_catch():
    # Create a read-only DataFrame
    df = pd.DataFrame({"close": np.random.randn(100)})
    
    # Make the underlying numpy array read-only (simulating Ray shared memory)
    # Note: df.values.setflags(write=False) makes the underlying array read-only
    df.values.setflags(write=False)
    
    feature_config = [
        {"id": "MemoryViolator", "params": {}}
    ]
    
    with pytest.raises(RuntimeError) as excinfo:
        compute_all_features(df, feature_config)
    
    assert "[MEMORY VIOLATION]" in str(excinfo.value)
    assert "MemoryViolator" in str(excinfo.value)
    assert "MUST NOT assign new columns directly to `df`" in str(excinfo.value)

def test_feature_dependency_memory_violation_catch():
    @register_feature("DependencyViolator")
    class DependencyViolator(Feature):
        @property
        def name(self): return "DependencyViolator"
        @property
        def description(self): return "Violates memory via dependency"
        @property
        def category(self): return "Test"

        def compute(self, df: pd.DataFrame, params: dict, cache) -> FeatureResult:
            # Triggering dependency that violates memory
            cache.get_series("MemoryViolator", {}, df)
            return FeatureResult(data={"dummy": df["close"]}, visuals=[])

    # Create a read-only DataFrame
    df = pd.DataFrame({"close": np.random.randn(100)})
    df.values.setflags(write=False)
    
    feature_config = [
        {"id": "DependencyViolator", "params": {}}
    ]
    
    with pytest.raises(RuntimeError) as excinfo:
        compute_all_features(df, feature_config)
    
    assert "[MEMORY VIOLATION]" in str(excinfo.value)
    assert "MemoryViolator" in str(excinfo.value)
    assert "feature dependency" in str(excinfo.value).lower()
