
import numpy as np
import pandas as pd
import context as ctx
from src.controller import SignalModel

class MomentumSurge(SignalModel):
    def generate_signals(self, df, params):
        # Use auto-generated context for robustness and IDE support
        # RSI_14 will map to "RSI_14" in df
        rsi_val = df[ctx.RSI_14]
        
        # Use ctx for hyperparameter keys as well
        condition_long = (rsi_val < params[ctx.RSI_LOWER])
        condition_short = (rsi_val > params[ctx.RSI_UPPER])
        
        signals = np.select(
            [condition_long, condition_short], 
            [1.0, -1.0], 
            default=0.0
        )
        return pd.Series(signals, index=df.index)
