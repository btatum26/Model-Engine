from typing import Dict, Any
import pandas as pd
import numpy as np
from ..base import Feature, LineOutput, FeatureResult

class KeltnerChannels(Feature):
    @property
    def name(self) -> str:
        return "Keltner Channels"

    @property
    def description(self) -> str:
        return "Volatility channels based on ATR and EMA."

    @property
    def category(self) -> str:
        return "Volatility"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "ema_period": 20,
            "atr_period": 10,
            "multiplier": 2.0,
            "normalize": "none",
            "color_center": "#ffffff",
            "color_bands": "#ffaa00"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        ema_period = int(params.get("ema_period", 20))
        atr_period = int(params.get("atr_period", 10))
        multiplier = float(params.get("multiplier", 2.0))
        norm_method = params.get("normalize", "none")
        
        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        close = df['Close'] if 'Close' in df.columns else df['close']
        
        # Center Line is an Exponential Moving Average of the closing price
        center_line = close.ewm(span=ema_period, adjust=False).mean()
        
        # Average True Range (ATR) calculation for channel width
        close_prev = close.shift(1)
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = tr.rolling(window=atr_period).mean()
        
        # Upper and Lower bands are offsets of the center line based on ATR
        upper_band = center_line + (multiplier * atr)
        lower_band = center_line - (multiplier * atr)
        
        visuals = [
            LineOutput(
                name="KC_Upper", 
                data=upper_band.where(pd.notnull(upper_band), None).tolist(), 
                color=params.get("color_bands"), 
                width=1
            ),
            LineOutput(
                name="KC_Center", 
                data=center_line.where(pd.notnull(center_line), None).tolist(), 
                color=params.get("color_center"), 
                width=1
            ),
            LineOutput(
                name="KC_Lower", 
                data=lower_band.where(pd.notnull(lower_band), None).tolist(), 
                color=params.get("color_bands"), 
                width=1
            )
        ]
        
        # Apply normalization to the center line and bands
        final_center = self.normalize(df, center_line, norm_method)
        final_upper = self.normalize(df, upper_band, norm_method)
        final_lower = self.normalize(df, lower_band, norm_method)
        
        col_prefix = "Dist_" if norm_method == "pct_distance" else ""
        
        data_dict = {
            f"{col_prefix}KC_Upper_{ema_period}": final_upper,
            f"{col_prefix}KC_Center_{ema_period}": final_center,
            f"{col_prefix}KC_Lower_{ema_period}": final_lower
        }
        
        return FeatureResult(visuals=visuals, data=data_dict)
