import json
import importlib.util
import sys
import os
import pandas as pd
import itertools
from typing import List, Dict, Any, Optional, Tuple

from .features.features import compute_all_features
from src.logger import logger
from src.exceptions import StrategyError

class LocalBacktester:
    """
    Handles the local execution and testing of user-defined strategy models.
    
    This class is responsible for loading the strategy manifest, dynamically 
    importing the user's Python scripts, computing Phase 2 features, purging 
    warmup NaNs, and executing the strategy's signal generation logic.
    """
    
    def __init__(self, strategy_dir: str):
        """
        Initializes the backtester for a specific strategy directory.

        Args:
            strategy_dir (str): The path to the strategy folder containing the 
                `manifest.json`, `model.py`, and `context.py` files.

        Raises:
            StrategyError: If the manifest file cannot be found or is invalid JSON.
        """
        self.strategy_dir = os.path.normpath(strategy_dir)
        self.manifest_path = os.path.join(self.strategy_dir, "manifest.json")
        try:
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load manifest at {self.manifest_path}: {e}")
            raise StrategyError(f"Missing or invalid manifest in {self.strategy_dir}")

    def _load_user_model_and_context(self) -> Tuple[type, Optional[type]]:
        """
        Dynamically imports the user-defined model and context classes.

        Returns:
            Tuple[type, Optional[type]]: 
                - model_class (type): The uninstantiated user SignalModel subclass.
                - context_class (Optional[type]): The uninstantiated user Context class, if it exists.

        Raises:
            StrategyError: If `model.py` is missing or fails to import safely.
        """
        model_path = os.path.join(self.strategy_dir, "model.py")
        context_module_path = os.path.join(self.strategy_dir, "context.py")
        
        if not os.path.exists(model_path):
            raise StrategyError(f"model.py not found in {self.strategy_dir}")

        try:
            # Import context first
            context_module_name = f"user_context_{os.path.basename(self.strategy_dir)}"
            spec_ctx = importlib.util.spec_from_file_location(context_module_name, context_module_path)
            context_module = importlib.util.module_from_spec(spec_ctx)
            if spec_ctx and spec_ctx.loader:
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
                if spec and spec.loader:
                    spec.loader.exec_module(module)
            finally:
                if self.strategy_dir in sys.path:
                    sys.path.remove(self.strategy_dir)
            
            from .controller import SignalModel
            for obj_name in dir(module):
                obj = getattr(module, obj_name)
                if isinstance(obj, type) and issubclass(obj, SignalModel) and obj is not SignalModel:
                    return obj, context_class
                    
            raise StrategyError(f"No valid SignalModel subclass found in {model_path}")
        except Exception as e:
            logger.error(f"Failed to load strategy components: {e}")
            raise StrategyError(f"Strategy initialization failed: {e}")
        
    def _audit_nans(self, df: pd.DataFrame, feature_ids: List[str]):
        """
        Scans for NaN values in computed features and logs warnings.

        Note: Since the l_max purge handles rolling indicator warmup periods, 
        any NaNs caught by this method indicate flawed data or broken indicators.

        Args:
            df (pd.DataFrame): The pre-processed dataset.
            feature_ids (List[str]): List of base feature names to audit.
        """
        for fid in feature_ids:
            cols = [c for c in df.columns if c.startswith(fid)]
            for col in cols:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logger.warning(f"Feature '{col}' has {nan_count} unexpected NaN values after l_max purge.")

    def run(self, raw_data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        """
        Executes a single vectorized backtest pass.

        Args:
            raw_data (pd.DataFrame): The raw OHLCV market data.
            params (Optional[Dict[str, Any]]): Strategy hyperparameters. Defaults to 
                the values in `manifest.json`.

        Returns:
            pd.Series: The generated conviction signals, bounded between [-1.0, 1.0].

        Raises:
            StrategyError: If execution fails at any point during feature computation or modeling.
        """
        try:
            features_config = self.manifest.get('features', [])
            
            # Universal Feature Calculation
            df_full, visuals, l_max = compute_all_features(raw_data, features_config)
            
            # Universal Warmup Purge (Protects both ML and Rule-Based from NaN lookbacks)
            df_clean = df_full.iloc[l_max:].copy()
            
            feature_ids = [f['id'] for f in features_config]
            self._audit_nans(df_clean, feature_ids)
            
            # Component Initialization
            model_class, context_class = self._load_user_model_and_context()
            model = model_class()
            context = context_class() if context_class else None
            hyperparams = params if params is not None else self.manifest.get('hyperparameters', {})
            
            # Routing (ML vs Rule-Based)
            is_ml = self.manifest.get("is_ml", False)
            
            if is_ml:
                # TODO: Inject MLBridge for MinMaxScaler transformations here
                pass
            
            # Execution
            artifacts = model.train(df_clean, context, hyperparams)
            
            # TODO: ArtifactManager should save/load here if splitting Train and Inference
            
            raw_signals = model.generate_signals(df_clean, context, hyperparams, artifacts)
            
            # TODO: Pass raw_signals through SignalValidator for [-1.0, 1.0] compression
            
            return raw_signals
            
        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            raise StrategyError(f"Backtest run failed: {e}")

    def run_grid_search(self, raw_data: pd.DataFrame, param_bounds: Optional[Dict[str, List[Any]]] = None) -> List[pd.Series]:
        """
        Executes a parameter sweep across defined hyperparameter bounds.

        Args:
            raw_data (pd.DataFrame): The raw OHLCV market data.
            param_bounds (Optional[Dict[str, List[Any]]]): Dictionary of lists containing 
                parameter options. Defaults to the bounds in `manifest.json`.

        Returns:
            List[pd.Series]: A list of signal series, one for each parameter permutation.
        """
        try:
            if param_bounds is None:
                param_bounds = self.manifest.get('parameter_bounds', {})
                
            if not param_bounds:
                return [self.run(raw_data)]

            features_config = self.manifest.get('features', [])
            df_full, _, l_max = compute_all_features(raw_data, features_config)
            
            # Apply Universal Warmup Purge
            df_clean = df_full.iloc[l_max:].copy()
            
            feature_ids = [f['id'] for f in features_config]
            self._audit_nans(df_clean, feature_ids)
            
            model_class, context_class = self._load_user_model_and_context()
            
            keys = list(param_bounds.keys())
            values = list(param_bounds.values())
            permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            
            logger.info(f"Starting parameter sweep across {len(permutations)} permutations.")
            results = []
            for i, p in enumerate(permutations):
                # Instantiate fresh objects to prevent state leakage
                model = model_class()
                context = context_class() if context_class else None
                
                artifacts = model.train(df_clean, context, p)
                signals = model.generate_signals(df_clean, context, p, artifacts)
                
                param_str = ", ".join([f"{k}={v}" for k, v in p.items()])
                signals.name = f"{os.path.basename(self.strategy_dir)} ({param_str})"
                results.append(signals)
                
            return results
        except Exception as e:
            logger.error(f"Grid search failed: {e}")
            raise StrategyError(f"Grid search failed: {e}")

    def run_batch(self, datasets: Dict[str, pd.DataFrame], params: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
        """
        Executes a batch of backtests across multiple assets efficiently.

        Args:
            datasets (Dict[str, pd.DataFrame]): A dictionary mapping ticker symbols to 
                their respective raw OHLCV DataFrames.
            params (Optional[Dict[str, Any]]): Strategy hyperparameters.

        Returns:
            Dict[str, pd.Series]: A dictionary mapping ticker symbols to their generated signals.
        """
        results = {}
        if not datasets:
            logger.warning("No datasets provided for batch backtest.")
            return results

        try:
            # Load the user's classes ONCE for the entire batch
            model_class, context_class = self._load_user_model_and_context()
            features_config = self.manifest.get('features', [])
            feature_ids = [f['id'] for f in features_config]
            hyperparams = params if params is not None else self.manifest.get('hyperparameters', {})

            for ticker, df_raw in datasets.items():
                logger.info(f"Processing batch execution for {ticker}")
                try:
                    df_full, _, l_max = compute_all_features(df_raw, features_config)
                    
                    # Apply Universal Warmup Purge
                    df_clean = df_full.iloc[l_max:].copy()
                    
                    self._audit_nans(df_clean, feature_ids)

                    # Instantiate fresh objects to prevent state leakage between assets
                    model = model_class()
                    context = context_class() if context_class else None

                    artifacts = model.train(df_clean, context, hyperparams)
                    signals = model.generate_signals(df_clean, context, hyperparams, artifacts)
                    results[ticker] = signals
                except Exception as e:
                    logger.error(f"Batch execution failed for {ticker}: {e}")
                    results[ticker] = pd.Series(dtype=float) 
                    
            return results
            
        except Exception as e:
            logger.error(f"Batch setup failed: {e}")
            raise StrategyError(f"Batch run failed: {e}")