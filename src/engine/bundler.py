import os
import zipfile
import py_compile

class Bundler:
    """The Exporter: Validates and zips the strategy directory into a .strat bundle."""
    @staticmethod
    def export(strategy_dir: str, output_dir: str) -> str:
        model_path = os.path.join(strategy_dir, "model.py")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model.py not found in {strategy_dir}")
            
        py_compile.compile(model_path, doraise=True)
        
        strat_name = os.path.basename(os.path.normpath(strategy_dir))
        os.makedirs(output_dir, exist_ok=True)
        bundle_path = os.path.join(output_dir, f"{strat_name}.strat")
        
        with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for file in ["manifest.json", "context.py", "model.py"]:
                filepath = os.path.join(strategy_dir, file)
                if os.path.exists(filepath):
                    bundle.write(filepath, file) 
                    
        return bundle_path
