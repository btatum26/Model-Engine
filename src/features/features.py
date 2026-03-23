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
    """
    base_dir = os.path.dirname(__file__)
    for loader, module_name, is_pkg in pkgutil.walk_packages([base_dir], prefix="src.features."):
        if not is_pkg:
            if module_name.endswith(".base") or module_name.endswith(".features"):
                continue
            try:
                importlib.import_module(module_name)
            except Exception as e:
                pass

# Initial load to populate FEATURE_REGISTRY
load_features()

class FeatureCache:
    """Dependency Resolver (Smart Caching)."""
    def __init__(self):
        self._memory: Dict[str, pd.Series] = {}

    def _generate_key(self, feature_id: str, params: Dict[str, Any]) -> str:
        if not params:
            return feature_id
        param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
        return f"{feature_id}_{param_str}"

    def get_series(self, feature_id: str, params: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
        key = self._generate_key(feature_id, params)
        
        if key in self._memory:
            return self._memory[key]
            
        if feature_id not in FEATURE_REGISTRY:
            raise ValueError(f"Dependency '{feature_id}' not found in registry.")
            
        feature_cls = FEATURE_REGISTRY[feature_id]
        feature_instance = feature_cls()
        
        # --- THE IRONCLAD BLAST SHIELD (Dependency Level) ---
        initial_col_count = len(df.columns)
        try:
            result: FeatureResult = feature_instance.compute(df, params, self)
        except ValueError as e:
            if "read-only" in str(e).lower() or "not writable" in str(e).lower():
                self._raise_memory_violation(feature_id, is_dependency=True)
            raise e

        # Explicitly audit the DataFrame shape to catch silent column additions
        if len(df.columns) != initial_col_count:
            self._raise_memory_violation(feature_id, is_dependency=True)
        # ----------------------------------------------------

        if not result.data:
            raise ValueError(f"Feature '{feature_id}' returned no data.")
            
        primary_series = None
        for col_name, series in result.data.items():
            self._memory[col_name] = series
            if primary_series is None:
                primary_series = series 
                
        self._memory[key] = primary_series
        return primary_series

    def set_series(self, key: str, series: pd.Series):
        self._memory[key] = series

    def _raise_memory_violation(self, feature_id: str, is_dependency: bool = False):
        dep_str = "dependency " if is_dependency else ""
        raise RuntimeError(
            f"\n[MEMORY VIOLATION] The feature {dep_str}'{feature_id}' attempted to mutate "
            f"the read-only DataFrame in place.\n"
            f"Rule: Phase 2 Features MUST NOT assign new columns directly to `df` "
            f"(e.g., df['temp'] = ...).\n"
            f"Solution: Use temporary variables (e.g., temp_series = df['close'] * 2) "
            f"and return them via the FeatureResult dictionary."
        )

class FeatureOrchestrator:
    """Execution Batcher (Stateless)."""

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

        cache = FeatureCache()

        for config in feature_config:
            feature_id = config.get("id")
            params = config.get("params", {})

            for k, v in params.items():
                if k.lower() in lookback_keys and isinstance(v, (int, float)):
                    l_max = max(l_max, int(v))

            feature_cls = FEATURE_REGISTRY[feature_id]
            feature_instance = feature_cls()

            # --- THE IRONCLAD BLAST SHIELD (Orchestrator Level) ---
            initial_col_count = len(df.columns)
            try:
                result: FeatureResult = feature_instance.compute(df, params, cache)
            except ValueError as e:
                if "read-only" in str(e).lower() or "not writable" in str(e).lower():
                    self._raise_memory_violation(feature_id)
                raise e

            # Explicit audit
            if len(df.columns) != initial_col_count:
                self._raise_memory_violation(feature_id)
            # ------------------------------------------------------

            if result.data:
                for col_name, series in result.data.items():
                    if col_name not in computed_features:
                        computed_features[col_name] = series
                        cache.set_series(col_name, series)

            if result.visuals:
                visuals_master_list.extend(result.visuals)

        if computed_features:
            new_features_df = pd.DataFrame(computed_features)
            df = pd.concat([df, new_features_df], axis=1)
            df = df.loc[:, ~df.columns.duplicated()]

        return df, visuals_master_list, l_max

    def _raise_memory_violation(self, feature_id: str):
        raise RuntimeError(
            f"\n[MEMORY VIOLATION] The feature '{feature_id}' attempted to mutate "
            f"the read-only DataFrame in place.\n"
            f"Rule: Features MUST NOT assign new columns directly to `df` "
            f"(e.g., df['temp'] = ...).\n"
            f"Solution: Use temporary variables (e.g., temp_series = df['close'] * 2) "
            f"and return them via the FeatureResult dictionary."
        )

orchestrator = FeatureOrchestrator()

def compute_all_features(df: pd.DataFrame, feature_config: List[Dict[str, Any]]):
    return orchestrator.compute_features(df, feature_config)