import os
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

from .data_broker import DataBroker
from .workspace import WorkspaceManager
from .backtester import LocalBacktester
from .metrics import Tearsheet

class ApplicationController:
    """
    The Master Orchestrator.
    Wires together DataBroker, WorkspaceManager, Backtester, and Tearsheet.
    """
    def __init__(self, strategies_dir: str = "src/strategies"):
        self.strategies_dir = strategies_dir
        self.broker = DataBroker()

    def run_backtest(self, 
                     strat_name: str, 
                     ticker: str, 
                     interval: str, 
                     start: Optional[str] = None, 
                     end: Optional[str] = None):
        """
        Main execution workflow.
        """
        print(f"\n[1/4] Initializing Backtest: {strat_name} on {ticker} ({interval})")
        
        # Validate Strategy
        strat_path = os.path.join(self.strategies_dir, strat_name)
        if not os.path.exists(strat_path):
            raise FileNotFoundError(f"Strategy {strat_name} not found in {self.strategies_dir}")
        
        print(f"      - Strategy path validated: {strat_path}")
        wm = WorkspaceManager(strat_path)
        
        # Fetch Data
        print(f"[2/4] Retrieving data for {ticker}...")
        df_raw = self.broker.get_data(ticker, interval, start, end)
        if df_raw.empty:
            print(f"      - ERROR: No data returned for {ticker} ({interval})")
            return

        print(f"      - Data retrieval complete: {len(df_raw)} bars found.")

        # Execute Strategy
        print(f"[3/4] Executing Strategy: {strat_name}...")
        backtester = LocalBacktester(strat_path)
        signals = backtester.run(df_raw)
        print(f"      - Strategy execution complete. Generated {len(signals[signals != 0])} non-zero signals.")
        
        # Calculate Metrics
        print(f"[4/4] Calculating performance metrics...")
        metrics = Tearsheet.calculate_metrics(df_raw, signals)
        
        # Output Results
        Tearsheet.print_summary(metrics)
        
        return metrics

    def run_grid_search(self, 
                        strat_name: str, 
                        ticker: str, 
                        interval: str, 
                        start: Optional[str] = None, 
                        end: Optional[str] = None):
        """Runs a hyperparameter sweep and outputs a comparison table."""
        print(f"\n[1/3] Starting Grid Search: {strat_name} on {ticker} ({interval})")
        
        strat_path = os.path.join(self.strategies_dir, strat_name)
        print(f"[2/3] Retrieving data...")
        df_raw = self.broker.get_data(ticker, interval, start, end)
        
        print(f"[3/3] Running hyperparameter sweep...")
        backtester = LocalBacktester(strat_path)
        results = backtester.run_grid_search(df_raw)
        
        print(f"\n--- Grid Search Results ({len(results)} permutations) ---")
        summary_data = []
        for signals in results:
            metrics = Tearsheet.calculate_metrics(df_raw, signals)
            # Remove equity curve for summary table
            metrics.pop('equity_curve', None)
            metrics['Strategy'] = signals.name
            summary_data.append(metrics)
            
        summary_df = pd.DataFrame(summary_data)
        # Reorder columns to put Strategy first
        cols = ['Strategy'] + [c for c in summary_df.columns if c != 'Strategy']
        print(summary_df[cols].to_string(index=False))
        
        return summary_df
