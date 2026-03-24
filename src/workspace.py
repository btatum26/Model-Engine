import os
import json
from typing import List, Dict, Any
from .features.base import Feature, FEATURE_REGISTRY

class WorkspaceManager:
    """Manages the synchronization of strategy configuration with local workspace files."""
    
    def __init__(self, strategy_dir: str):
        self.strategy_dir = strategy_dir
        self.manifest_path = os.path.join(strategy_dir, "manifest.json")
        self.context_path = os.path.join(strategy_dir, "context.py")

    def sync(self, features: List[Dict[str, Any]], hparams: Dict[str, Any], bounds: Dict[str, Any]):
        """Update manifest and generate a synchronized context.py for strategy execution."""
        # 1. Update manifest.json
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {}

        manifest["features"] = features
        manifest["hyperparameters"] = hparams
        manifest["parameter_bounds"] = bounds

        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)

        # 2. Generate context.py mapping feature IDs to column names
        with open(self.context_path, 'w') as f:
            f.write("# Auto-generated context mapping feature IDs to DataFrame column names.\n\n")
            f.write("class Context:\n")
            f.write("    def __init__(self):\n")
            
            # Map each feature configuration to a standardized column name
            for feature_config in features:
                fid = feature_config.get("id")
                params = feature_config.get("params", {})
                
                if fid not in FEATURE_REGISTRY:
                    continue
                    
                feature_cls = FEATURE_REGISTRY[fid]
                outputs = feature_cls().outputs
                
                for output in outputs:
                    # Standardized column name generation
                    col_name = Feature.generate_column_name(fid, params, output)
                    
                    # Create a class attribute mapping for user access in model.py
                    # If it's a multi-output feature, use the output name as the attribute
                    if output:
                        attr_name = output.upper().replace(" ", "_").replace("-", "_")
                    else:
                        # Fallback for single-output features
                        attr_name = fid.upper().replace(" ", "_").replace("-", "_")
                        
                    f.write(f"        self.{attr_name} = '{col_name}'\n")
            
            if not features:
                f.write("        pass\n")
