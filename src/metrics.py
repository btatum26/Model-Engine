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
        Calculates performance metrics using T+1 execution model.
        Signals generated at [T] close execute at [T+1] open.
        """
        print(f"      - Processing {len(signals)} signals with T+1 execution model...")
        
        # Pct change from T open to T+1 open, shifted to align with T signal
        returns = df['Open'].pct_change().shift(-1)
        strategy_returns = signals * returns
        
        # Apply friction on every trade (signal change)
        trades = signals.diff().fillna(0).abs()
        strategy_returns -= trades * friction
        
        equity_curve = (1 + strategy_returns.fillna(0)).cumprod()
        
        total_return = (equity_curve.iloc[-1] - 1) * 100
        total_trades = int((trades > 0).sum())
        
        days = (df.index.max() - df.index.min()).days
        if days > 0:
            cagr = ((equity_curve.iloc[-1]) ** (365.25 / days) - 1) * 100
        else:
            cagr = 0
            
        trade_returns = strategy_returns[trades > 0]
        win_rate = (trade_returns > 0).mean() * 100 if not trade_returns.empty else 0
        
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Sharpe Ratio assuming 252 trading days
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
