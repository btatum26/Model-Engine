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
    for loader, module_name, is_pkg in pkgutil.walk_packages([base_dir], prefix="."):
        if not is_pkg:
            try:
                importlib.import_module(module_name, package=__package__)
            except Exception as e:
                print(f"Error loading feature module {module_name}: {e}")

# Initial load to populate FEATURE_REGISTRY
load_features()

class FeatureOrchestrator:
    """
    Handles validation, caching, and batch execution of features.
    """
    def __init__(self):
        self.shared_cache: Dict[str, pd.Series] = {}

    def validate_config(self, feature_config: List[Dict[str, Any]]):
        """
        Validates that all requested features exist and have valid parameters.
        """
        for config in feature_config:
            feature_id = config.get("id")
            if not feature_id:
                raise ValueError("Feature config missing 'id' field.")
            
            if feature_id not in FEATURE_REGISTRY:
                raise ValueError(f"Feature '{feature_id}' not found in registry.")
            
            # Additional validation (e.g., parameter types) could be added here
            # by checking against the feature's parameter_options.

    def compute_features(self, df: pd.DataFrame, feature_config: List[Dict[str, Any]]) -> tuple[pd.DataFrame, List[Any]]:
        """
        Executes a batch of features and returns the updated DataFrame and visual outputs.
        """
        self.validate_config(feature_config)
        
        computed_features = {}
        visuals_master_list = []
        
        # Reset cache for a fresh run
        self.shared_cache = {}

        for config in feature_config:
            feature_id = config.get("id")
            params = config.get("params", {})
            
            feature_cls = FEATURE_REGISTRY[feature_id]
            feature_instance = feature_cls()
            
            # Compute feature
            result: FeatureResult = feature_instance.compute(df, params, self.shared_cache)
            
            # Collect data
            if result.data:
                for col_name, series in result.data.items():
                    computed_features[col_name] = series
            
            # Collect visuals
            if result.visuals:
                visuals_master_list.extend(result.visuals)
        
        # Single-pass concatenation
        if computed_features:
            new_features_df = pd.DataFrame(computed_features)
            df = pd.concat([df, new_features_df], axis=1)
            
        return df, visuals_master_list

# Global instance for easy access
orchestrator = FeatureOrchestrator()

def compute_all_features(df: pd.DataFrame, feature_config: List[Dict[str, Any]]):
    """Convenience wrapper for the orchestrator."""
    return orchestrator.compute_features(df, feature_config)
