import json
import importlib.util
import sys
import os
import pandas as pd
import itertools
from typing import List, Dict, Any, Optional
from .features.features import compute_all_features
from .logger import logger
from .exceptions import StrategyError

class LocalBacktester:
    """Handles the local execution and testing of strategy models."""
    
    def __init__(self, strategy_dir: str):
        self.strategy_dir = os.path.normpath(strategy_dir)
        self.manifest_path = os.path.join(self.strategy_dir, "manifest.json")
        try:
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load manifest at {self.manifest_path}: {e}")
            raise StrategyError(f"Missing or invalid manifest in {self.strategy_dir}")

    def _load_user_model_and_context(self):
        """Dynamically imports the user-defined model and context classes."""
        model_path = os.path.join(self.strategy_dir, "model.py")
        context_module_path = os.path.join(self.strategy_dir, "context.py")
        
        if not os.path.exists(model_path):
            raise StrategyError(f"model.py not found in {self.strategy_dir}")

        try:
            # Import context first
            context_module_name = f"user_context_{os.path.basename(self.strategy_dir)}"
            spec_ctx = importlib.util.spec_from_file_location(context_module_name, context_module_path)
            context_module = importlib.util.module_from_spec(spec_ctx)
            spec_ctx.loader.exec_module(context_module)
            context_class = getattr(context_module, 'Context', None)

            # Import model
            model_module_name = f"user_strat_{os.path.basename(self.strategy_dir)}"
            spec = importlib.util.spec_from_file_location(model_module_name, model_path)
            module = importlib.util.module_from_spec(spec)
            
            # Allow internal imports within the strategy directory
            if self.strategy_dir not in sys.path:
                sys.path.insert(0, self.strategy_dir)
                
            try:
                sys.modules[model_module_name] = module
                spec.loader.exec_module(module)
            finally:
                if self.strategy_dir in sys.path:
                    sys.path.remove(self.strategy_dir)
            
            from .controller import SignalModel
            for obj_name in dir(module):
                obj = getattr(module, obj_name)
                if isinstance(obj, type) and issubclass(obj, SignalModel) and obj is not SignalModel:
                    return obj(), context_class
                    
            raise StrategyError(f"No valid SignalModel subclass found in {model_path}")
        except Exception as e:
            logger.error(f"Failed to load strategy components: {e}")
            raise StrategyError(f"Strategy initialization failed: {e}")

    def _audit_nans(self, df: pd.DataFrame, feature_ids: List[str]):
        """Scans for NaN values in computed features and logs warnings if found."""
        for fid in feature_ids:
            cols = [c for c in df.columns if c.startswith(fid)]
            for col in cols:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logger.warning(f"Feature '{col}' has {nan_count} NaN values.")

    def run(self, raw_data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        """Executes a single backtest pass with specified parameters."""
        try:
            features_config = self.manifest.get('features', [])
            df_full, _, _ = compute_all_features(raw_data, features_config)
            
            feature_ids = [f['id'] for f in features_config]
            self._audit_nans(df_full, feature_ids)
            
            model, context = self._load_user_model_and_context()
            hyperparams = params if params is not None else self.manifest.get('hyperparameters', {})
            
            artifacts = model.train(df_full, context, hyperparams)
            signals = model.generate_signals(df_full, context, hyperparams, artifacts)
            return signals
        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            raise StrategyError(f"Backtest run failed: {e}")

    def run_grid_search(self, raw_data: pd.DataFrame, param_bounds: Optional[Dict[str, List[Any]]] = None) -> List[pd.Series]:
        """Executes a parameter sweep across defined bounds."""
        try:
            if param_bounds is None:
                param_bounds = self.manifest.get('parameter_bounds', {})
                
            if not param_bounds:
                return [self.run(raw_data)]

            features_config = self.manifest.get('features', [])
            df_full, _, _ = compute_all_features(raw_data, features_config)
            
            feature_ids = [f['id'] for f in features_config]
            self._audit_nans(df_full, feature_ids)
            
            model, context = self._load_user_model_and_context()
            
            keys = list(param_bounds.keys())
            values = list(param_bounds.values())
            permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            
            logger.info(f"Starting parameter sweep across {len(permutations)} permutations.")
            results = []
            for i, p in enumerate(permutations):
                artifacts = model.train(df_full, context, p)
                signals = model.generate_signals(df_full, context, p, artifacts)
                
                param_str = ", ".join([f"{k}={v}" for k, v in p.items()])
                signals.name = f"{os.path.basename(self.strategy_dir)} ({param_str})"
                results.append(signals)
                
            return results
        except Exception as e:
            logger.error(f"Grid search failed: {e}")
            raise StrategyError(f"Grid search failed: {e}")
