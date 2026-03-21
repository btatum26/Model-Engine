from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Feature, FeatureOutput, LineOutput, FeatureResult

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
    def target_pane(self) -> str:
        return "main" # Plotted directly over the price candles

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "ema_period": 20,
            "atr_period": 10,
            "multiplier": 2.0,
            "color_center": "#ffffff",
            "color_bands": "#ffaa00"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        ema_period = int(params.get("ema_period", 20))
        atr_period = int(params.get("atr_period", 10))
        multiplier = float(params.get("multiplier", 2.0))
        
        # 1. Calculate Center Line (EMA)
        center_line = df['Close'].ewm(span=ema_period, adjust=False).mean()
        
        # 2. Calculate ATR (Using our hyper-fast NumPy logic)
        high = df['High']
        low = df['Low']
        close_prev = df['Close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = tr.rolling(window=atr_period).mean()
        
        # 3. Calculate Bands
        upper_band = center_line + (multiplier * atr)
        lower_band = center_line - (multiplier * atr)
        
        def clean(s): return s.where(pd.notnull(s), None).tolist()
        
        visuals = [
            LineOutput(name="KC_Upper", data=clean(upper_band), color=params.get("color_bands"), width=1),
            LineOutput(name="KC_Center", data=clean(center_line), color=params.get("color_center"), width=1),
            LineOutput(name="KC_Lower", data=clean(lower_band), color=params.get("color_bands"), width=1)
        ]
        
        data_dict = {
            f"KC_Upper_{ema_period}_{multiplier}": upper_band,
            f"KC_Center_{ema_period}": center_line,
            f"KC_Lower_{ema_period}_{multiplier}": lower_band
        }
        
        return FeatureResult(visuals=visuals, data=data_dict)