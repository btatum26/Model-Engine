import time
from datetime import datetime
from .database import Database
from .fetcher import DataFetcher
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

class SignalEvaluation:
    """
    Evaluates vectorized signals for performance.
    """
    def __init__(self, forward_window=5, threshold=0.01):
        self.forward_window = forward_window
        self.threshold = threshold

    def evaluate_vectorized(self, df: pd.DataFrame, signals: pd.Series):
        """
        Calculates performance metrics for a series of signals in a vectorized way.
        signals: Series of -1, 0, 1
        """
        if signals.empty:
            return None
            
        # Shift close prices for forward window returns
        future_returns = df['Close'].shift(-self.forward_window) / df['Close'] - 1
        
        # Align signals with returns
        valid_signals = signals[signals != 0]
        if valid_signals.empty:
            return None
            
        perf = future_returns.loc[valid_signals.index] * valid_signals
        
        results = {
            "total_signals": len(valid_signals),
            "correct_calls": int((perf > self.threshold).sum()),
            "incorrect_calls": int((perf < -self.threshold).sum()),
            "avg_fwd_pnl": float(perf.mean()),
            "win_rate": float((perf > 0).mean())
        }
        return results

class BaseStrategy:
    """
    Standard Object-Oriented plugin architecture for strategies.
    """
    def __init__(self, name="Base Strategy"):
        self.name = name

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Takes a full DataFrame and returns a Series of signals (-1, 0, 1).
        Must be implemented by subclasses.
        """
        return pd.Series(0, index=df.index)

class TradingEngine:
    def __init__(self, db_path="data/stocks.db"):
        self.db = Database(db_path)
        self.fetcher = DataFetcher()

    def sync_data(self, ticker, interval, period=None, quiet=False):
        """Downloads data and saves to DB."""
        last_ts = self.db.get_latest_timestamp(ticker, interval)
        
        df = pd.DataFrame()
        if period:
            if not quiet: print(f"\nSyncing {ticker} ({interval}) for period {period}...")
            df = self.fetcher.fetch_historical(ticker, interval, period=period)
        elif last_ts:
            if not quiet: print(f"\nSyncing {ticker} ({interval}) from {last_ts} to now...")
            df = self.fetcher.fetch_historical(ticker, interval, start=last_ts)
        else:
            default_period = "1y"
            if not quiet: print(f"\nNo local data found. Syncing {ticker} ({interval}) for {default_period}...")
            df = self.fetcher.fetch_historical(ticker, interval, period=default_period)

        if not df.empty:
            self.db.save_data(df, ticker, interval)
            if not quiet: print(f"Saved {len(df)} bars.")
        elif not quiet:
            print("No new data found.")

    def run_backtest(self, ticker, interval, strategy: BaseStrategy, start=None, end=None, period="1y"):
        """Runs a strategy against historical data using full vectorization."""
        print(f"Starting Vectorized Backtest: {ticker} ({interval})")
        
        df = self.db.get_data(ticker, interval, start, end)
        if df.empty:
            print(f"No local data found for {ticker} ({interval}). Fetching...")
            self.sync_data(ticker, interval, period=period)
            df = self.db.get_data(ticker, interval, start, end)

        if df.empty:
            print("Still no data found. Aborting.")
            return pd.Series(0, index=pd.Index([])), pd.DataFrame()

        # Pure Vectorization: Call strategy with the entire DataFrame
        signals = strategy.generate_signals(df)
        
        # Phase 4: Portfolio Simulation (Vectorized)
        # T+1 Execution: Signals generated at Close execute on next Open/Close.
        # Here we assume execution on next Close for simplicity of returns calculation.
        daily_returns = df['Close'].pct_change().shift(-1) # Return from today close to tomorrow close
        
        # Matrix multiplication: Signals * Asset Returns
        strategy_returns = signals * daily_returns
        
        # Friction Modeling: Subtract spread/slippage (e.g., 0.1% per trade)
        trades = signals.diff().fillna(0) != 0
        friction = 0.001
        strategy_returns[trades] -= friction
        
        equity_curve = (1 + strategy_returns.fillna(0)).cumprod()
        
        return signals, equity_curve

    def run_live(self, ticker, interval, strategy: BaseStrategy, refresh_rate=60):
        """
        Simulates live testing.
        """
        print(f"Starting Live Simulation: {ticker} ({interval})")
        while True:
            try:
                df = self.fetcher.fetch_historical(ticker, interval, period="1d")
                if not df.empty:
                    signals = strategy.generate_signals(df)
                    if not signals.empty and signals.iloc[-1] != 0:
                        print(f"[{datetime.now()}] Signal detected: {signals.iloc[-1]}")
                
                time.sleep(refresh_rate)
            except KeyboardInterrupt:
                print("Stopping live simulation...")
                break
            except Exception as e:
                print(f"Error in live simulation: {e}")
                time.sleep(refresh_rate)
