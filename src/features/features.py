import os
import importlib
import pkgutil
import pandas as pd
from typing import List, Dict, Any, Optional
from .base import FEATURE_REGISTRY, FeatureResult, Feature
from ..logger import logger
from ..exceptions import FeatureError

def load_features():
    """
    Dynamically imports all modules in the features subdirectories 
    to trigger registration of feature classes.
    """
    base_dir = os.path.dirname(__file__)
    for loader, module_name, is_pkg in pkgutil.walk_packages([base_dir], prefix="src.features."):
        if not is_pkg:
            # Skip base modules to avoid circular imports
            if module_name.endswith(".base") or module_name.endswith(".features"):
                continue
            try:
                importlib.import_module(module_name)
            except Exception as e:
                logger.error(f"Failed to load feature module {module_name}: {e}")

# Initial load to populate the feature registry
load_features()

class FeatureCache:
    """Manages in-memory caching of computed feature series."""
    
    def __init__(self):
        self._memory: Dict[str, pd.Series] = {}

    def _generate_key(self, feature_id: str, params: Dict[str, Any]) -> str:
        """Generates a unique cache key based on feature ID and parameters."""
        if not params:
            return feature_id
        param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
        return f"{feature_id}_{param_str}"

    def get_series(self, feature_id: str, params: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
        """Retrieves a feature series from cache or computes it if missing."""
        key = self._generate_key(feature_id, params)
        
        if key in self._memory:
            return self._memory[key]
            
        if feature_id not in FEATURE_REGISTRY:
            raise FeatureError(f"Feature '{feature_id}' not found in registry.")
            
        feature_cls = FEATURE_REGISTRY[feature_id]
        feature_instance = feature_cls()
        
        # Verify that the feature does not modify the input DataFrame
        initial_col_count = len(df.columns)
        try:
            result: FeatureResult = feature_instance.compute(df, params, self)
        except ValueError as e:
            # Catch common errors related to read-only views
            if "read-only" in str(e).lower() or "not writable" in str(e).lower():
                self._raise_memory_violation(feature_id, is_dependency=True)
            raise e
        except Exception as e:
            logger.error(f"Error computing feature {feature_id}: {e}")
            raise FeatureError(f"Feature computation failed for {feature_id}")

        if len(df.columns) != initial_col_count:
            self._raise_memory_violation(feature_id, is_dependency=True)

        if not result.data:
            raise FeatureError(f"Feature '{feature_id}' returned no data.")
            
        primary_series = None
        for col_name, series in result.data.items():
            self._memory[col_name] = series
            if primary_series is None:
                primary_series = series 
                
        self._memory[key] = primary_series
        return primary_series

    def set_series(self, key: str, series: pd.Series):
        """Manually adds a series to the cache."""
        self._memory[key] = series

    def _raise_memory_violation(self, feature_id: str, is_dependency: bool = False):
        """Centralized error for in-place DataFrame mutations."""
        dep_str = "dependency " if is_dependency else ""
        error_msg = (
            f"Feature {dep_str}'{feature_id}' attempted to mutate the input DataFrame in place. "
            f"Features must return new Series objects instead of assigning new columns to the input df."
        )
        logger.error(error_msg)
        raise FeatureError(error_msg)

class FeatureOrchestrator:
    """Handles the batch computation of multiple features."""

    def validate_config(self, feature_config: List[Dict[str, Any]]):
        """Validates the feature configuration list."""
        for config in feature_config:
            feature_id = config.get("id")
            if not feature_id:
                raise ValidationError("Feature configuration missing 'id' field.")
            if feature_id not in FEATURE_REGISTRY:
                raise ValidationError(f"Feature '{feature_id}' not found in registry.")

    def compute_features(self, df: pd.DataFrame, feature_config: List[Dict[str, Any]]) -> tuple[pd.DataFrame, List[Any], int]:
        """Executes feature computation based on the provided configuration."""
        self.validate_config(feature_config)

        computed_features = {}
        visuals_master_list = []
        l_max = 0
        lookback_keys = ["window", "period", "slow", "fast", "lookback"]

        cache = FeatureCache()

        for config in feature_config:
            feature_id = config.get("id")
            params = config.get("params", {})

            # Track the maximum lookback window required
            for k, v in params.items():
                if k.lower() in lookback_keys and isinstance(v, (int, float)):
                    l_max = max(l_max, int(v))

            feature_cls = FEATURE_REGISTRY[feature_id]
            feature_instance = feature_cls()

            # Ensure data integrity by checking column counts
            initial_col_count = len(df.columns)
            try:
                result: FeatureResult = feature_instance.compute(df, params, cache)
            except ValueError as e:
                if "read-only" in str(e).lower() or "not writable" in str(e).lower():
                    self._raise_memory_violation(feature_id)
                raise e
            except Exception as e:
                logger.error(f"Orchestrator failed to compute feature {feature_id}: {e}")
                raise FeatureError(f"Feature computation failed for {feature_id}")

            if len(df.columns) != initial_col_count:
                self._raise_memory_violation(feature_id)

            if result.data:
                for col_name, series in result.data.items():
                    if col_name not in computed_features:
                        computed_features[col_name] = series
                        cache.set_series(col_name, series)

            if result.visuals:
                visuals_master_list.extend(result.visuals)

        # Merge new features into the main DataFrame
        if computed_features:
            new_features_df = pd.DataFrame(computed_features)
            df = pd.concat([df, new_features_df], axis=1)
            df = df.loc[:, ~df.columns.duplicated()]

        return df, visuals_master_list, l_max

    def _raise_memory_violation(self, feature_id: str):
        """Centralized error for in-place DataFrame mutations."""
        error_msg = (
            f"Feature '{feature_id}' attempted to mutate the input DataFrame in place. "
            f"Features must return new Series objects instead of assigning new columns to the input df."
        )
        logger.error(error_msg)
        raise FeatureError(error_msg)

orchestrator = FeatureOrchestrator()

def compute_all_features(df: pd.DataFrame, feature_config: List[Dict[str, Any]]):
    """Utility function for high-level feature computation."""
    return orchestrator.compute_features(df, feature_config)
