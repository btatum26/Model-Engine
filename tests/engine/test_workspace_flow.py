import os
import sys
import pandas as pd
import numpy as np
import json

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.workspace import WorkspaceManager
from src.engine.backtester import LocalBacktester
from src.engine.bundler import Bundler

# 1. Setup Strategy Workspace
STRAT_DIR = "src/strategies/momentum_surge"
MODEL_PY_CONTENT = """
import numpy as np
import pandas as pd
from src.engine.controller import SignalModel

class MomentumSurge(SignalModel):
    def train(self, df, context, params):
        return {}

    def generate_signals(self, df, context, params, artifacts):
        # Use auto-generated context
        # RSI will map to "RSI_14" in df
        rsi_val = df[context.RSI]
        
        condition_long = (rsi_val < params['rsi_lower'])
        condition_short = (rsi_val > params['rsi_upper'])
        
        signals = np.select(
            [condition_long, condition_short], 
            [1.0, -1.0], 
            default=0.0
        )
        return pd.Series(signals, index=df.index)
"""

def test_full_flow():
    # A. Sync Workspace (GUI Action)
    print("--- 1. Syncing Workspace ---")
    wm = WorkspaceManager(STRAT_DIR)
    features = [{"id": "RSI", "params": {"period": 14}}]
    hyperparams = {"rsi_lower": 30, "rsi_upper": 70}
    parameter_bounds = {"rsi_lower": [20, 30], "rsi_upper": [70, 80]}
    
    wm.sync(features, hyperparams, parameter_bounds)
    
    # Write the model.py
    with open(os.path.join(STRAT_DIR, "model.py"), 'w') as f:
        f.write(MODEL_PY_CONTENT)
    
    print(f"Context generated at {wm.context_path}")
    with open(wm.context_path, 'r') as f:
        print(f"Context content:\n{f.read()}")

    # B. Run Backtest (Local Execution)
    print("\n--- 2. Running Local Backtest ---")
    # Mock Data
    df = pd.DataFrame({
        'Open': np.random.randn(100),
        'High': np.random.randn(100),
        'Low': np.random.randn(100),
        'Close': np.random.randn(100),
        'Volume': np.random.randn(100)
    }, index=pd.date_range('2023-01-01', periods=100, freq='D'))
    
    backtester = LocalBacktester(STRAT_DIR)
    signals = backtester.run(df)
    print(f"Backtest generated {len(signals)} signals.")
    print(f"Signal counts:\n{signals.value_counts()}")

    # C. Grid Search
    print("\n--- 3. Running Grid Search ---")
    grid_results = backtester.run_grid_search(df)
    print(f"Grid search generated {len(grid_results)} permutations.")
    for res in grid_results:
        print(f"Permutation: {res.name} | Mean Signal: {res.mean():.4f}")

    # D. Export to .strat (Deployment)
    print("\n--- 4. Exporting to .strat ---")
    bundle_path = Bundler.export(STRAT_DIR, "exports")
    print(f"Bundle created at: {bundle_path}")
    
    # Mock latest data (with features pre-computed as assumed by node)
    from src.engine.features.features import compute_all_features
    df_with_features, _, l_max = compute_all_features(df, features)
    
    # Needs a mock context for Live Node
    class MockContext:
        RSI_14 = "RSI_14"

if __name__ == "__main__":
    test_full_flow()
