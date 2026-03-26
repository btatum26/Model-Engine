
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
