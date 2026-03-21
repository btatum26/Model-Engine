from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Feature, FeatureOutput, LineOutput, FeatureResult

class AverageTrueRange(Feature):
    @property
    def name(self) -> str:
        return "ATR"

    @property
    def description(self) -> str:
        return "Average True Range. Measures absolute volatility."

    @property
    def target_pane(self) -> str:
        return "new"

    @property
    def category(self) -> str:
        return "Volatility"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 14,
            "color": "#ff0000"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        period = int(params.get("period", 14))
        
        high = df['High']
        low = df['Low']
        close_prev = df['Close'].shift(1)
        
        # Highly optimized NumPy matrix math
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = tr.rolling(window=period).mean()
        
        visuals = [
            LineOutput(
                name=f"ATR_{period}",
                data=atr.where(pd.notnull(atr), None).tolist(),
                color=params.get("color", "#ff0000"),
                width=2
            )
        ]
        return FeatureResult(visuals=visuals, data={f"ATR_{period}": atr})