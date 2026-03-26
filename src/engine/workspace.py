import os
import json
import keyword
from typing import List, Dict, Any
from jinja2 import Environment, FileSystemLoader

from .features.base import FEATURE_REGISTRY

class WorkspaceManager:
    """Manages the synchronization of strategy configuration with local workspace files."""
    
    def __init__(self, strategy_dir: str, template_dir: str = "src/engine/templates"):
        self.strategy_dir = strategy_dir
        self.manifest_path = os.path.join(strategy_dir, "manifest.json")
        self.context_path = os.path.join(strategy_dir, "context.py")
        self.model_path = os.path.join(strategy_dir, "model.py")
        
        # Initialize Jinja2 Environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _infer_type(self, value: Any) -> str:
        """Maps JSON/Python runtime types to type hint strings."""
        if isinstance(value, bool): return "bool"
        if isinstance(value, int): return "int"
        if isinstance(value, float): return "float"
        if isinstance(value, str): return "str"
        if isinstance(value, list): return "list"
        if isinstance(value, dict): return "dict"
        return "Any"

    def _build_features_payload(self, features_config: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generates the flat naming payload for features, handling multi-outputs."""
        payload = []
        for config in features_config:
            fid = config.get("id")
            params = config.get("params", {})
            
            if fid not in FEATURE_REGISTRY:
                continue
                
            feature_instance = FEATURE_REGISTRY[fid]()
            outputs = feature_instance.outputs
            
            # Filter out UI/Engine parameters
            core_params = {k: v for k, v in params.items() if k not in ["color", "normalize", "overbought", "oversold"] and not k.startswith("color_")}
            
            # Generate the Base Attribute Name
            if len(core_params) == 1 and ("period" in core_params or "window" in core_params):
                val = core_params.get("period") or core_params.get("window")
                base_attr = f"{fid.upper()}_{val}"
            elif not core_params:
                base_attr = fid.upper()
            else:
                param_str = "_".join([str(v) for k, v in sorted(core_params.items())])
                base_attr = f"{fid.upper()}_{param_str}"
                
            # Iterate over outputs to handle MACD, Bollinger Bands, etc.
            for output in outputs:
                col_name = feature_instance.generate_column_name(fid, params, output)
                
                if output:
                    suffix = output.upper().replace(" ", "_").replace("-", "_")
                    attr_name = f"{base_attr}_{suffix}"
                else:
                    attr_name = base_attr
                    
                payload.append({
                    "attr_name": attr_name,
                    "col_name": col_name,
                    "docstring": f"Feature: {feature_instance.name}\n    Outputs: {output or 'Primary'}\n    Params: {params}"
                })
        return payload

    def sync(self, features: List[Dict[str, Any]], hparams: Dict[str, Any], bounds: Dict[str, Any]):
        """Update manifest and generate synchronized context/model files using Jinja2."""
        
        # 1. Validate Hyperparameters (Block Python Keywords)
        for key in hparams.keys():
            if keyword.iskeyword(key):
                raise ValueError(f"Hyperparameter '{key}' is a reserved Python keyword and cannot be used.")
        
        # 2. Update manifest.json
        manifest = {}
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)

        manifest["features"] = features
        manifest["hyperparameters"] = hparams
        manifest["parameter_bounds"] = bounds

        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)

        # 3. Build Jinja Context Payload
        params_payload = [
            {"name": k, "type": self._infer_type(v), "value": repr(v)}
            for k, v in hparams.items()
        ]
        
        template_data = {
            "features": self._build_features_payload(features),
            "params": params_payload
        }

        # 4. Render and Write context.py (Always Overwrite)
        context_template = self.jinja_env.get_template("context.py.j2")
        rendered_context = context_template.render(**template_data)
        
        with open(self.context_path, 'w') as f:
            f.write(rendered_context)

        # 5. Render and Write model.py (Only if it doesn't exist)
        if not os.path.exists(self.model_path):
            model_template = self.jinja_env.get_template("model.py.j2")
            rendered_model = model_template.render()
            with open(self.model_path, 'w') as f:
                f.write(rendered_model)