import os
import pandas as pd
import numpy as np
import json
from src.workspace import WorkspaceManager
from src.backtester import LocalBacktester
from src.bundler import Bundler
from src.live_node import LiveTradingNode

# 1. Setup Strategy Workspace
STRAT_DIR = "src/strategies/momentum_surge"
MODEL_PY_CONTENT = """
import numpy as np
import pandas as pd
import context as ctx
from src.controller import SignalModel

class MomentumSurge(SignalModel):
    def generate_signals(self, df, params):
        # Use auto-generated context
        # RSI_14 will map to "RSI_14" in df
        rsi_val = df[ctx.RSI_14]
        
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

    # E. Live Node Deployment (Production)
    print("\n--- 5. Testing Live Node Deployment ---")
    node = LiveTradingNode(bundle_path)
    model = node.deploy()
    print(f"Live Node deployed strategy: {type(model).__name__}")
    
    # Mock latest data (with features pre-computed as assumed by node)
    from src.features.features import compute_all_features
    df_with_features, _ = compute_all_features(df, node.config['features'])
    
    node.on_market_tick(df_with_features)
    node.cleanup()
    print("Live Node cleanup complete.")

if __name__ == "__main__":
    test_full_flow()
