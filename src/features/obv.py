from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Feature, FeatureOutput, LineOutput, FeatureResult

class OBV(Feature):
    @property
    def name(self) -> str:
        return "On-Balance Volume (OBV)"

    @property
    def description(self) -> str:
        return "Cumulative volume indicator mapping buying/selling pressure."

    @property
    def category(self) -> str:
        return "Volume & Profile"

    @property
    def target_pane(self) -> str:
        return "new"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "color": "#00aaff"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        # np.sign returns 1 if positive, -1 if negative, 0 if flat
        price_diff = df['Close'].diff().fillna(0)
        direction = np.sign(price_diff)
        
        # Multiply volume by direction and calculate cumulative sum
        obv = (df['Volume'] * direction).cumsum()
        
        visuals = [
            LineOutput(
                name="OBV",
                data=obv.where(pd.notnull(obv), None).tolist(),
                color=params.get("color", "#00aaff"),
                width=2
            )
        ]
        
        # We also return a rolling SMA of OBV to the data dictionary.
        # This allows strategies to easily check if OBV is trending up or down.
        obv_sma = obv.rolling(window=20).mean()
        
        data_dict = {
            "OBV": obv,
            "OBV_SMA_20": obv_sma
        }
        
        return FeatureResult(visuals=visuals, data=data_dict)