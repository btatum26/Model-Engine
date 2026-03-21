from typing import Dict, Any, List
import pandas as pd
from .base import Feature, FeatureOutput, LineOutput, FeatureResult

class BollingerBands(Feature):
    @property
    def name(self) -> str:
        return "Bollinger Bands"

    @property
    def description(self) -> str:
        return "Volatility bands based on Standard Deviation. Supports Bollinger Width and systematic normalization."

    @property
    def category(self) -> str:
        return "Volatility"

    @property
    def target_pane(self) -> str:
        return "main"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 20,
            "std_dev": 2.0,
            "normalize": "none",
            "color_bands": "#00d4ff",
            "color_mid": "#ffffff"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        period = int(params.get("period", 20))
        std_dev = float(params.get("std_dev", 2.0))
        norm_method = params.get("normalize", "none")
        
        # Standardize OHLCV access
        close = df['Close'] if 'Close' in df.columns else df['close']
        
        # Calculate Bollinger Bands math
        mid_band = close.rolling(window=period).mean()
        rolling_std = close.rolling(window=period).std()
        
        upper_band = mid_band + (rolling_std * std_dev)
        lower_band = mid_band - (rolling_std * std_dev)
        
        # Bollinger Width: (Upper - Lower) / Mid
        # Useful for identifying "squeezes"
        width = (upper_band - lower_band) / mid_band.replace(0, 1e-9)
        
        # Prepare visuals for GUI (always raw prices)
        def clean(s): return s.where(pd.notnull(s), None).tolist()
        
        visuals = [
            LineOutput(
                name=f"BB_Upper_{period}", 
                data=clean(upper_band), 
                color=params.get("color_bands"), 
                width=1
            ),
            LineOutput(
                name=f"BB_Mid_{period}", 
                data=clean(mid_band), 
                color=params.get("color_mid"), 
                width=1
            ),
            LineOutput(
                name=f"BB_Lower_{period}", 
                data=clean(lower_band), 
                color=params.get("color_bands"), 
                width=1
            )
        ]
        
        # Normalize data based on strategy request
        norm_upper = self.normalize(df, upper_band, norm_method)
        norm_mid = self.normalize(df, mid_band, norm_method)
        norm_lower = self.normalize(df, lower_band, norm_method)
        
        # Format keys for Alpha Engine
        prefix = "Dist_" if norm_method == "pct_distance" else ""
        
        data_dict = {
            f"{prefix}BB_Upper_{period}": norm_upper,
            f"{prefix}BB_Mid_{period}": norm_mid,
            f"{prefix}BB_Lower_{period}": norm_lower,
            f"BB_Width_{period}": width
        }
        
        return FeatureResult(visuals=visuals, data=data_dict)
