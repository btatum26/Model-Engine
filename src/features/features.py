import os
import importlib
import pkgutil
import pandas as pd
from typing import List, Dict, Any, Optional
from .base import FEATURE_REGISTRY, FeatureResult, Feature

def load_features():
    """
    Dynamically imports all modules in the features subdirectories 
    to trigger @register_feature decorators.
    Uses pkgutil for reliable submodule discovery.
    """
    base_dir = os.path.dirname(__file__)
    # Iterates through all submodules in the current package
    for loader, module_name, is_pkg in pkgutil.walk_packages([base_dir], prefix="src.features."):
        if not is_pkg:
            # Skip base and features itself to avoid circularity
            if module_name.endswith(".base") or module_name.endswith(".features"):
                continue
            try:
                importlib.import_module(module_name)
            except Exception as e:
                # Silently fail if it's not a real feature module or has other issues
                pass

# Initial load to populate FEATURE_REGISTRY
load_features()

class FeatureCache:
    """
    Dependency Resolver (Smart Caching).
    Instantiated freshly on every backtest run.
    """
    def __init__(self):
        # Internal dictionary to store computed pd.Series
        self._memory: Dict[str, pd.Series] = {}

    def _generate_key(self, feature_id: str, params: Dict[str, Any]) -> str:
        """Generates a deterministic key like 'EMA_period12'"""
        if not params:
            return feature_id
        # Sort parameters to ensure consistent key generation regardless of dictionary order
        param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
        return f"{feature_id}_{param_str}"

    def get_series(self, feature_id: str, params: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
        """
        Retrieves a series from memory, or computes and caches it if missing.
        """
        key = self._generate_key(feature_id, params)
        
        # Check cache
        if key in self._memory:
            return self._memory[key]
            
        # Validate existence
        if feature_id not in FEATURE_REGISTRY:
            raise ValueError(f"Dependency '{feature_id}' not found in registry.")
        feature_cls = FEATURE_REGISTRY[feature_id]
        feature_instance = feature_cls()
        
        # Pass 'self' so nested dependencies can also use the cache
        result: FeatureResult = feature_instance.compute(df, params, self)
        if not result.data:
            raise ValueError(f"Feature '{feature_id}' returned no data.")
            
        # Cache all returned outputs (e.g., MACD returns macd_line and signal_line)
        primary_series = None
        for col_name, series in result.data.items():
            self._memory[col_name] = series
            if primary_series is None:
                primary_series = series # Default to the first series added
                
        # Also map the specific deterministic key to the primary series
        self._memory[key] = primary_series
        return primary_series

    def set_series(self, key: str, series: pd.Series):
        """Allows the orchestrator to manually warm the cache with top-level features."""
        self._memory[key] = series


class FeatureOrchestrator:
    """
    Execution Batcher (Stateless).
    Handles validation, caching, and batch execution of features.
    """
    # Removed __init__ with self.shared_cache to enforce strict statelessness.

    def validate_config(self, feature_config: List[Dict[str, Any]]):
        for config in feature_config:
            feature_id = config.get("id")
            if not feature_id:
                raise ValueError("Feature config missing 'id' field.")
            if feature_id not in FEATURE_REGISTRY:
                raise ValueError(f"Feature '{feature_id}' not found in registry.")

    def compute_features(self, df: pd.DataFrame, feature_config: List[Dict[str, Any]]) -> tuple[pd.DataFrame, List[Any], int]:
        self.validate_config(feature_config)

        computed_features = {}
        visuals_master_list = []
        l_max = 0
        lookback_keys = ["window", "period", "slow", "fast", "lookback"]

        # Instantiate a fresh cache for this specific execution run
        cache = FeatureCache()

        for config in feature_config:
            feature_id = config.get("id")
            params = config.get("params", {})

            # Calculate l_max (absolute maximum lookback)
            for k, v in params.items():
                if k.lower() in lookback_keys and isinstance(v, (int, float)):
                    l_max = max(l_max, int(v))

            feature_cls = FEATURE_REGISTRY[feature_id]
            feature_instance = feature_cls()

            # Compute feature, passing in the localized cache
            result: FeatureResult = feature_instance.compute(df, params, cache)

            # Collect data and warm the cache
            if result.data:
                for col_name, series in result.data.items():
                    # Only append if we haven't already calculated it as a dependency
                    if col_name not in computed_features:
                        computed_features[col_name] = series
                        cache.set_series(col_name, series)

            # Collect visuals
            if result.visuals:
                visuals_master_list.extend(result.visuals)

        # Single-pass concatenation
        if computed_features:
            new_features_df = pd.DataFrame(computed_features)
            df = pd.concat([df, new_features_df], axis=1)

            # Drop duplicated columns in case user config requested a dependency explicitly
            df = df.loc[:, ~df.columns.duplicated()]

        return df, visuals_master_list, l_max

# Global instance for easy access
orchestrator = FeatureOrchestrator()

def compute_all_features(df: pd.DataFrame, feature_config: List[Dict[str, Any]]):
    return orchestrator.compute_features(df, feature_config)
