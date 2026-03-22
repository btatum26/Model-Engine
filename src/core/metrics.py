import pandas as pd
import numpy as np
from typing import Dict, Any

class Tearsheet:
    """
    The Tearsheet Engine: Translates raw signals into trading reality.
    Grade the strategy's real-world viability mathematically.
    """
    @staticmethod
    def calculate_metrics(df: pd.DataFrame, signals: pd.Series, friction: float = 0.001) -> Dict[str, Any]:
        """
        Calculates performance metrics.
        - T+1 Routing: Signals generated at [T] close execute at [T+1] open.
        """
        print(f"      - Processing {len(signals)} signals with T+1 execution model...")
        # Vectorized returns calculation
        # Return from today's open to tomorrow's open (next bar execution)
        # Assuming df has 'Open' and signals are aligned with close of previous bar
        returns = df['Open'].pct_change().shift(-1) # pct change from T open to T+1 open, shifted to align with T signal
        
        # Apply signals
        strategy_returns = signals * returns
        
        # Apply friction (slippage/fees) on every trade (signal change)
        trades = signals.diff().fillna(0).abs() # Magnitude of signal change
        strategy_returns -= trades * friction
        
        # Equity Curve
        equity_curve = (1 + strategy_returns.fillna(0)).cumprod()
        
        # Basic Metrics
        total_return = (equity_curve.iloc[-1] - 1) * 100
        total_trades = int((trades > 0).sum())
        
        # Annualized Return (CAGR)
        days = (df.index.max() - df.index.min()).days
        if days > 0:
            cagr = ((equity_curve.iloc[-1]) ** (365.25 / days) - 1) * 100
        else:
            cagr = 0
            
        # Win Rate
        trade_returns = strategy_returns[trades > 0]
        win_rate = (trade_returns > 0).mean() * 100 if not trade_returns.empty else 0
        
        # Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Sharpe Ratio (Assuming 252 trading days for simplicity, can be adjusted)
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = (strategy_returns.mean() * 252) / volatility if volatility > 0 else 0

        return {
            "Total Return (%)": round(total_return, 2),
            "CAGR (%)": round(cagr, 2),
            "Max Drawdown (%)": round(max_drawdown, 2),
            "Win Rate (%)": round(win_rate, 2),
            "Total Trades": total_trades,
            "Sharpe Ratio": round(sharpe, 2),
            "equity_curve": equity_curve
        }

    @staticmethod
    def print_summary(metrics: Dict[str, Any]):
        """Outputs the final metrics to the console in a clean ASCII table."""
        print("\n" + "="*40)
        print(" " * 10 + "STRATEGY PERFORMANCE")
        print("="*40)
        for key, value in metrics.items():
            if key == "equity_curve": continue
            print(f"{key:<25}: {value}")
        print("="*40 + "\n")
