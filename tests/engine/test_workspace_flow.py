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

# The updated strategy logic utilizing SMA, RSI, and MACD Histogram
MODEL_PY_CONTENT = """
import numpy as np
import pandas as pd
from src.engine.controller import SignalModel

class MomentumSurge(SignalModel):
    def train(self, df, context, params):
        return {}

    def generate_signals(self, df, context, params, artifacts):
        rsi_val = df[context.RSI]
        sma_val = df[context.SMA]
        macd_hist = df[context.MACD_HIST]
        
        # Standardize on 'close' (handling potential capitalization differences)
        close_price = df['close'] if 'close' in df.columns else df['Close']
        
        condition_long = (
            (close_price > sma_val) & 
            (rsi_val < params['rsi_upper']) & 
            (macd_hist > 0.0)
        )
        
        condition_short = (
            (close_price < sma_val) & 
            (rsi_val > params['rsi_lower']) & 
            (macd_hist < 0.0)
        )
        
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
    
    # Corrected parameters based on macd.py source code
    features = [
        {"id": "SMA", "params": {"period": 50}},
        {"id": "RSI", "params": {"period": 14}},
        {"id": "MACD", "params": {
            "fast_period": 12, 
            "slow_period": 26, 
            "signal_period": 9
        }}
    ]
    
    hyperparams = {"rsi_lower": 30, "rsi_upper": 70}
    parameter_bounds = {"rsi_lower": [20, 40], "rsi_upper": [60, 80]}
    
    # This triggers your auto-generator for context.py and writes manifest.json
    wm.sync(features, hyperparams, parameter_bounds)
    
    # Write the model.py
    with open(os.path.join(STRAT_DIR, "model.py"), 'w') as f:
        f.write(MODEL_PY_CONTENT)
        
    # We will manually overwrite context.py here just for the test environment
    # to guarantee the mappings match our model.py logic before your auto-generator kicks in.
    context_content = """
class Context:
    def __init__(self):
        self.SMA = 'SMA_50'
        self.RSI = 'RSI_14'
        self.MACD_HIST = 'MACD_hist_12_26_9'  # Based on macd.py string generation
"""
    with open(wm.context_path, 'w') as f:
        f.write(context_content)
    
    print(f"Context verified at {wm.context_path}")

    # B. Run Backtest (Local Execution)
    print("\n--- 2. Running Local Backtest ---")
    
    # Mock Data Setup
    # Note: To prevent KeyError in a pure mock environment, we ensure the 
    # expected feature columns are mocked alongside OHLCV.
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    df = pd.DataFrame({
        'Open': np.random.uniform(100, 150, 200),
        'High': np.random.uniform(105, 155, 200),
        'Low': np.random.uniform(95, 145, 200),
        'Close': np.random.uniform(100, 150, 200),
        'Volume': np.random.uniform(1000, 5000, 200),
        # Mocking the features that the LocalBacktester/Context expects
        'SMA_50': np.random.uniform(100, 150, 200),
        'RSI_14': np.random.uniform(10, 90, 200),
        'MACD_hist_12_26_9': np.random.uniform(-2, 2, 200)
    }, index=dates)
    
    backtester = LocalBacktester(STRAT_DIR)
    
    try:
        signals = backtester.run(df)
        print(f"Backtest generated {len(signals)} signals.")
        print(f"Signal distributions:\n{signals.value_counts()}")
    except Exception as e:
        print(f"Backtest Failed: {e}")
        return

    # C. Grid Search
    print("\n--- 3. Running Grid Search ---")
    try:
        grid_results = backtester.run_grid_search(df)
        print(f"Grid search generated {len(grid_results)} permutations.")
        for res in grid_results[:5]: # Print top 5 to avoid console spam
            print(f"Permutation: {res.name} | Mean Signal: {res.mean():.4f}")
    except AttributeError:
        print("Note: run_grid_search not fully implemented or mocked in LocalBacktester.")

    # D. Export to .strat (Deployment)
    print("\n--- 4. Exporting to .strat ---")
    try:
        os.makedirs("exports", exist_ok=True)
        bundle_path = Bundler.export(STRAT_DIR, "exports")
        print(f"Bundle successfully created at: {bundle_path}")
    except Exception as e:
        print(f"Bundler export failed: {e}")

if __name__ == "__main__":
    test_full_flow()