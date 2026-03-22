from typing import Dict, Any, List
import pandas as pd
from ..base import Feature, FeatureOutput, LineOutput, FeatureResult, register_feature

@register_feature("BollingerBands")
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

    @property
    def outputs(self) -> List[str]:
        return ["upper", "mid", "lower", "width"]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        period = int(params.get("period", 20))
        std_dev = float(params.get("std_dev", 2.0))
        norm_method = params.get("normalize", "none")
        
        # Standardize OHLCV access
        close = df['Close'] if 'Close' in df.columns else df['close']
        
        # Calculate Bollinger Bands math using Cache for SMA
        if cache:
            mid_band = cache.get_series("SMA", {"period": period}, df)
        else:
            mid_band = close.rolling(window=period).mean()
            
        rolling_std = close.rolling(window=period).std()
        
        upper_band = mid_band + (rolling_std * std_dev)
        lower_band = mid_band - (rolling_std * std_dev)
        
        # Bollinger Width: (Upper - Lower) / Mid
        # Useful for identifying "squeezes"
        width = (upper_band - lower_band) / mid_band.replace(0, 1e-9)
        
        # Prepare visuals for GUI (always raw prices)
        def clean(s): return s.where(pd.notnull(s), None).tolist()
        
        col_upper = self.generate_column_name("BollingerBands", params, "upper")
        col_mid = self.generate_column_name("BollingerBands", params, "mid")
        col_lower = self.generate_column_name("BollingerBands", params, "lower")
        col_width = self.generate_column_name("BollingerBands", params, "width")
        
        visuals = [
            LineOutput(
                name=col_upper, 
                data=clean(upper_band), 
                color=params.get("color_bands"), 
                width=1
            ),
            LineOutput(
                name=col_mid, 
                data=clean(mid_band), 
                color=params.get("color_mid"), 
                width=1
            ),
            LineOutput(
                name=col_lower, 
                data=clean(lower_band), 
                color=params.get("color_bands"), 
                width=1
            )
        ]
        
        # Normalize data based on strategy request
        norm_upper = self.normalize(df, upper_band, norm_method)
        norm_mid = self.normalize(df, mid_band, norm_method)
        norm_lower = self.normalize(df, lower_band, norm_method)
        norm_width = self.normalize(df, width, norm_method)
        
        data_dict = {
            col_upper: norm_upper,
            col_mid: norm_mid,
            col_lower: norm_lower,
            col_width: norm_width
        }
        
        return FeatureResult(visuals=visuals, data=data_dict)
