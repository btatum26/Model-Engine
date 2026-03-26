
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
