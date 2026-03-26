import numpy as np
import pandas as pd
from src.engine.controller import SignalModel

class Model(SignalModel):
    """
    RSI Fractal Divergence Strategy.
    Uses T-Zero confirmed fractals to detect bullish/bearish divergence.
    """

    def train(self, df: pd.DataFrame, context, params: dict) -> dict:
        """Optimization is handled by the core engine."""
        return {}

    def generate_signals(self, df: pd.DataFrame, context, params: dict, artifacts: dict) -> pd.Series:
        """
        Purely vectorized signal generation for RSI Fractal Divergence.
        Returns continuous float64 Series [-1.0 to 1.0].
        """
        lookback_limit = params.get('Macro_Lookback_Window', 40)
        exit_centerline_long = params.get('Exit_Centerline_Long', 55)
        exit_centerline_short = params.get('Exit_Centerline_Short', 45)
        
        is_fractal_low_confirmed = df[context.IS_FRACTAL_LOW_CONFIRMED]
        curr_low_price = df[context.CURRENT_CONFIRMED_LOW_PRICE]
        curr_low_rsi = df[context.CURRENT_CONFIRMED_LOW_RSI]
        prev_low_price = df[context.PREVIOUS_LOW_PRICE]
        prev_low_rsi = df[context.PREVIOUS_LOW_RSI]
        bars_since_prev_low = df[context.BARS_SINCE_PREVIOUS_LOW]
        
        is_fractal_high_confirmed = df[context.IS_FRACTAL_HIGH_CONFIRMED]
        curr_high_price = df[context.CURRENT_CONFIRMED_HIGH_PRICE]
        curr_high_rsi = df[context.CURRENT_CONFIRMED_HIGH_RSI]
        prev_high_price = df[context.PREVIOUS_HIGH_PRICE]
        prev_high_rsi = df[context.PREVIOUS_HIGH_RSI]
        bars_since_prev_high = df[context.BARS_SINCE_PREVIOUS_HIGH]
        
        rsi_base = df[context.RSI_BASE]
        close = df['Close'] if 'Close' in df.columns else df['close']

        # Condition: New confirmed low is lower than previous, but RSI is higher
        is_new_low = is_fractal_low_confirmed == True
        lower_low_price = curr_low_price < prev_low_price
        higher_low_rsi = curr_low_rsi > prev_low_rsi
        valid_timeframe_long = bars_since_prev_low <= lookback_limit
        # Breakout: Current price > previous fractal high to confirm momentum
        breakout_long = close > prev_high_price
        
        long_entry_mask = is_new_low & lower_low_price & higher_low_rsi & valid_timeframe_long & breakout_long
        
        # Condition: New confirmed high is higher than previous, but RSI is lower
        is_new_high = is_fractal_high_confirmed == True
        higher_high_price = curr_high_price > prev_high_price
        lower_high_rsi = curr_high_rsi < prev_high_rsi
        valid_timeframe_short = bars_since_prev_high <= lookback_limit
        # Breakout: Current price < previous fractal low
        breakout_short = close < prev_low_price
        
        short_entry_mask = is_new_high & higher_high_price & lower_high_rsi & valid_timeframe_short & breakout_short
        
        # Exit Masks
        exit_long_mask = rsi_base > exit_centerline_long
        exit_short_mask = rsi_base < exit_centerline_short
        
        # We use a state-persistent approach. 
        # Entry signals set the state to 1 or -1. 
        # Exit signals reset the state ONLY if we are in that specific position.
        
        # Initial signals: entries
        signals = pd.Series(np.nan, index=df.index)
        signals.loc[long_entry_mask] = 1.0
        signals.loc[short_entry_mask] = -1.0
        
        # We need to handle exits carefully to not wipe out a new entry on the same bar (rare but possible)
        # But for this strategy, we'll follow the plan's select/ffill logic
        
        # The plan's logic:
        # conditions = [long_entry_mask, short_entry_mask, exit_mask]
        # choices = [1.0, -1.0, 0.0]
        # This means an exit resets state regardless of current position.
        
        # Let's stick to the plan's logic for now as requested.
        exit_mask = exit_long_mask | exit_short_mask
        
        conditions = [long_entry_mask, short_entry_mask, exit_mask]
        choices = [1.0, -1.0, 0.0]
        
        raw_signals = np.select(conditions, choices, default=np.nan)
        
        # 7. Forward Fill to maintain state until an exit or reversal is triggered
        continuous_signals = pd.Series(raw_signals, index=df.index).ffill().fillna(0.0)
        
        return continuous_signals
