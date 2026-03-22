import zipfile
import tempfile
import json
import importlib.util
import sys
import os
from .engine import SignalModel

class LiveTradingNode:
    """The Headless Consumer: Deploys .strat bundles in a secure, remote environment."""
    def __init__(self, strat_file_path: str):
        self.strat_file_path = strat_file_path
        self.model = None
        self.config = None

    def deploy(self):
        """Unzips the .strat file into a temporary directory and loads it."""
        # We use a temporary directory that persists for the lifetime of the deployment
        self._temp_dir = tempfile.TemporaryDirectory()
        temp_path = self._temp_dir.name
        
        # Unzip the bundle
        with zipfile.ZipFile(self.strat_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_path)
        with open(os.path.join(temp_path, "manifest.json"), 'r') as f:
            self.config = json.load(f)
            
        # Dynamically Load Logic
        module_name = f"live_strat_{os.path.basename(self.strat_file_path).replace('.', '_')}"
        model_path = os.path.join(temp_path, "model.py")
        spec = importlib.util.spec_from_file_location(module_name, model_path)
        module = importlib.util.module_from_spec(spec)
        
        if temp_path not in sys.path:       # context.py
            sys.path.insert(0, temp_path)
            
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        finally:
            # Clean up sys.path and context cache
            if temp_path in sys.path:
                sys.path.remove(temp_path)
            if 'context' in sys.modules:
                del sys.modules['context']
        
        # Find the class that inherits from SignalModel
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if isinstance(obj, type) and issubclass(obj, SignalModel) and obj is not SignalModel:
                self.model = obj()
                return self.model
                
        raise TypeError(f"No SignalModel found in {self.strat_file_path}")

    def on_market_tick(self, live_dataframe):
        """Triggered by market data feed."""
        if not self.model:
            raise RuntimeError("Strategy not deployed.")
            
        # Calculate signal (Phase 2 feature calculation assumed to be handled by caller)
        # In a real live node, the caller would ensure features are attached to live_dataframe
        signal = self.model.generate_signals(live_dataframe, self.config['hyperparameters'])
        
        # Extract the very last signal to execute
        current_signal = signal.iloc[-1]
        self._execute_trade(current_signal)

    def _execute_trade(self, signal_value: float):
        # Placeholder for broker API integration
        print(f"Executing trade with weight: {signal_value}")

    def cleanup(self):
        if hasattr(self, '_temp_dir'):
            self._temp_dir.cleanup()
