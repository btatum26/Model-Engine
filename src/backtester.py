import json
import importlib.util
import sys
import os
import pandas as pd
import itertools
from typing import List, Dict, Any, Optional
from .features.features import compute_all_features

class LocalBacktester:
    """
    The Dynamic Backtester: Acts as the local test environment.
    Calculates features, injects parameters, and runs the strategy.
    """
    def __init__(self, strategy_dir: str):
        self.strategy_dir = os.path.normpath(strategy_dir)
        self.manifest_path = os.path.join(self.strategy_dir, "manifest.json")
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)

    def _load_user_model(self):
        """Dynamically imports model.py from the strategy directory."""
        model_path = os.path.join(self.strategy_dir, "model.py")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model.py not found in {self.strategy_dir}")

        module_name = f"user_strat_{os.path.basename(self.strategy_dir)}"
        print(f"      - Loading strategy model: {module_name}...")
        
        spec = importlib.util.spec_from_file_location(module_name, model_path)
        module = importlib.util.module_from_spec(spec)
        
        # Add strategy dir to sys.path so it can import context.py
        if self.strategy_dir not in sys.path:
            sys.path.insert(0, self.strategy_dir)
            
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        finally:
            if self.strategy_dir in sys.path:
                sys.path.remove(self.strategy_dir)
            if 'context' in sys.modules:
                del sys.modules['context']
        
        from .controller import SignalModel
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if isinstance(obj, type) and issubclass(obj, SignalModel) and obj is not SignalModel:
                return obj()
        
        class_name = os.path.basename(self.strategy_dir).title().replace('_', '')
        if hasattr(module, class_name):
            return getattr(module, class_name)()
            
        raise TypeError(f"No SignalModel subclass found in {model_path}")

    def run(self, raw_data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        """Runs the strategy once with given parameters."""
        features_config = self.manifest.get('features', [])
        print(f"      - Computing {len(features_config)} features...")
        df_with_features, _ = compute_all_features(raw_data, features_config)
        
        model = self._load_user_model()
        hyperparams = params if params is not None else self.manifest.get('hyperparameters', {})
        print(f"      - Running generate_signals with {len(hyperparams)} parameters.")
        signals = model.generate_signals(df_with_features, hyperparams)
        return signals

    def run_grid_search(self, raw_data: pd.DataFrame, param_bounds: Optional[Dict[str, List[Any]]] = None) -> List[pd.Series]:
        """Fast Grid Search (Hyperparameter Sweeps)."""
        if param_bounds is None:
            param_bounds = self.manifest.get('parameter_bounds', {})
            
        if not param_bounds:
            return [self.run(raw_data)]

        features_config = self.manifest.get('features', [])
        print(f"      - Grid Search: Pre-calculating {len(features_config)} features...")
        df_with_features, _ = compute_all_features(raw_data, features_config)
        
        model = self._load_user_model()
        
        keys = list(param_bounds.keys())
        values = list(param_bounds.values())
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"      - Starting sweep across {len(permutations)} permutations...")
        results = []
        for i, p in enumerate(permutations):
            if (i+1) % max(1, len(permutations)//5) == 0:
                print(f"      - Progress: {i+1}/{len(permutations)}...")
            signals = model.generate_signals(df_with_features, p)
            param_str = ", ".join([f"{k}={v}" for k, v in p.items()])
            signals.name = f"{os.path.basename(self.strategy_dir)} ({param_str})"
            results.append(signals)
            
        return results
