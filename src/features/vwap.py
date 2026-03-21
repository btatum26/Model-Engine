from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Feature, FeatureOutput, LineOutput, FeatureResult

class VWAP(Feature):
    @property
    def name(self) -> str:
        return "VWAP"

    @property
    def description(self) -> str:
        return "Volume Weighted Average Price with daily resets."

    @property
    def category(self) -> str:
        return "Volume & Profile"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "color": "#00d8ff"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        # Create a temporary date column for grouping
        dates = df.index.date
        
        # Typical Price
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        # Volume * TP
        v_tp = tp * df['Volume']
        
        # Vectorized Daily VWAP using groupby and cumsum
        cum_v_tp = v_tp.groupby(dates).cumsum()
        cum_v = df['Volume'].groupby(dates).cumsum()
        
        vwap = cum_v_tp / cum_v
        
        visuals = [
            LineOutput(
                name="VWAP",
                data=vwap.where(pd.notnull(vwap), None).tolist(),
                color=params.get("color", "#00d8ff"),
                width=2
            )
        ]
        
        return FeatureResult(visuals=visuals, data={"VWAP": vwap})
