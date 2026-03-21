from typing import Dict, Any, List
import pandas as pd
from .base import Feature, FeatureOutput, LineOutput, FeatureResult

class BollingerWidth(Feature):
    @property
    def name(self) -> str:
        return "Bollinger Band Width"

    @property
    def description(self) -> str:
        return "Measures standard deviation compression. Normalized for ML."

    @property
    def category(self) -> str:
        return "Volatility"

    @property
    def target_pane(self) -> str:
        return "new" # Plot the width oscillator in a new pane below the chart

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 20,
            "std_dev": 2.0,
            "color": "#00d4ff"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        period = int(params.get("period", 20))
        std_dev = float(params.get("std_dev", 2.0))
        
        # 1. Calculate the core components
        sma = df['Close'].rolling(window=period).mean()
        rolling_std = df['Close'].rolling(window=period).std()
        
        # 2. Calculate the bands
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        
        # 3. Calculate Normalized Width: (Upper - Lower) / SMA
        # We replace 0 with a tiny number to prevent divide-by-zero errors in flat markets
        safe_sma = sma.replace(0, 1e-9)
        bb_width = (upper_band - lower_band) / safe_sma
        
        visuals = [
            LineOutput(
                name=f"BB_Width_{period}_{std_dev}",
                data=bb_width.where(pd.notnull(bb_width), None).tolist(),
                color=params.get("color", "#00d4ff"),
                width=2
            )
        ]
        
        # We return the actual bands AND the width to the data engine just in case 
        # a strategy wants to calculate price crossing a band later.
        data_dict = {
            f"BB_Upper_{period}": upper_band,
            f"BB_Lower_{period}": lower_band,
            f"BB_Width_{period}": bb_width
        }
        
        return FeatureResult(visuals=visuals, data=data_dict)