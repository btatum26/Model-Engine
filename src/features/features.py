import os
import importlib
import pandas as pd
from typing import List, Dict, Any, Optional
from .base import FEATURE_REGISTRY, FeatureResult, Feature

def load_features():
    """
    Dynamically imports all modules in the features subdirectories 
    to trigger @register_feature decorators.
    """
    base_dir = os.path.dirname(__file__)
    for root, dirs, files in os.walk(base_dir):
        # Skip the base directory itself (base.py, features.py)
        if root == base_dir:
            continue
            
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Convert path to module format
                # e.g., src/features/momentum/rsi.py -> .momentum.rsi
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                module_name = "." + rel_path.replace(os.path.sep, ".").replace(".py", "")
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
        
        # Reset cache for a fresh run if needed, or maintain it across symbols?
        # For now, we'll keep it for the duration of this call.
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
