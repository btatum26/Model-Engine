import json
import os
import uuid
import joblib
import importlib.util
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
from .signals import SignalEvent
from .engine import BaseStrategy

class Strategy(BaseStrategy):
    """
    Modern Strategy class:
    1. Inherits from BaseStrategy.
    2. Stores configuration in JSON.
    3. Stores ML models in separate joblib files.
    4. Dynamically loads logic from a Python class.
    """
    def __init__(self, name: str, directory: str = "strategies"):
        super().__init__(name)
        self.directory = directory
        self.config_path = os.path.join(self.directory, f"{self.name}.json")
        self.models_dir = os.path.join(self.directory, "models", self.name)
        self.logic_path = os.path.join(self.directory, "logic", f"{self.name}.py")
        
        self.feature_config = {}
        self.signal_params = {}
        self.active_model_id = None
        self.model_metadata = {} # UUID -> {type, metrics, path, date}
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.logic_path), exist_ok=True)
        
        if os.path.exists(self.config_path):
            self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.feature_config = config.get("feature_config", {})
                self.signal_params = config.get("signal_params", {})
                self.active_model_id = config.get("active_model_id")
                self.model_metadata = config.get("model_metadata", {})
        except Exception as e:
            print(f"Error loading config for {self.name}: {e}")

    def save_config(self):
        config = {
            "name": self.name,
            "feature_config": self.feature_config,
            "signal_params": self.signal_params,
            "active_model_id": self.active_model_id,
            "model_metadata": self.model_metadata,
            "last_updated": datetime.now().isoformat()
        }
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def add_model(self, model, model_type: str, metrics: Dict[str, Any]):
        model_id = str(uuid.uuid4())
        model_filename = f"{model_id}.joblib"
        model_path = os.path.join(self.models_dir, model_filename)
        
        joblib.dump(model, model_path)
        
        self.model_metadata[model_id] = {
            "type": model_type,
            "metrics": metrics,
            "filename": model_filename,
            "date": datetime.now().isoformat()
        }
        self.active_model_id = model_id
        self.save_config()
        return model_id

    def get_active_model(self):
        if not self.active_model_id or self.active_model_id not in self.model_metadata:
            return None
        
        model_info = self.model_metadata[self.active_model_id]
        model_path = os.path.join(self.models_dir, model_info["filename"])
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Loads the strategy logic dynamically and executes it.
        """
        if not os.path.exists(self.logic_path):
            return pd.Series(0, index=df.index)

        try:
            module_name = f"strategy_logic_{self.name}"
            spec = importlib.util.spec_from_file_location(module_name, self.logic_path)
            module = importlib.util.module_from_spec(spec)
            
            # Remove from sys.modules to allow reloading
            import sys
            if module_name in sys.modules:
                del sys.modules[module_name]
                
            spec.loader.exec_module(module)
            
            # Expecting a class named 'StrategyLogic' in the file
            if hasattr(module, 'StrategyLogic'):
                logic_instance = module.StrategyLogic(self)
                return logic_instance.generate_signals(df)
        except Exception as e:
            print(f"Error executing strategy logic for {self.name}: {e}")
        
        return pd.Series(0, index=df.index)

    @staticmethod
    def list_available(directory: str = "strategies"):
        if not os.path.exists(directory):
            return []
        return [f.replace(".json", "") for f in os.listdir(directory) if f.endswith(".json")]
