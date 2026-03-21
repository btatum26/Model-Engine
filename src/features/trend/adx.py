from typing import Dict, Any, List
import pandas as pd
import numpy as np
from ..base import Feature, LineOutput, FeatureResult, register_feature

@register_feature("ADX")
class ADX(Feature):
    @property
    def name(self) -> str:
        return "ADX"

    @property
    def description(self) -> str:
        return "Average Directional Index (Trend Strength). Includes +DI and -DI."

    @property
    def category(self) -> str:
        return "Trend Indicators"

    @property
    def target_pane(self) -> str:
        return "new"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 14,
            "normalize": "none",
            "color_adx": "#ffffff",
            "color_plus_di": "#44ff44",
            "color_minus_di": "#ff4444"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        period = int(params.get("period", 14))
        norm_method = params.get("normalize", "none")
        
        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        close = df['Close'] if 'Close' in df.columns else df['close']
        
        # Calculate True Range
        close_prev = close.shift(1)
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high.diff()
        down_move = low.diff().abs()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values using Wilder's Smoothing (EMA with alpha=1/N)
        tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, 1e-9))
        minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, 1e-9))
        
        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        def clean(s): return s.where(pd.notnull(s), None).tolist()
        
        visuals = [
            LineOutput(name=f"ADX_{period}", data=clean(adx), color=params.get("color_adx"), width=2),
            LineOutput(name="+DI", data=clean(plus_di), color=params.get("color_plus_di"), width=1),
            LineOutput(name="-DI", data=clean(minus_di), color=params.get("color_minus_di"), width=1)
        ]
        
        # Apply systematic normalization
        final_adx = self.normalize(df, adx, norm_method)
        
        data_dict = {
            f"ADX_{period}": final_adx,
            f"Plus_DI_{period}": plus_di,
            f"Minus_DI_{period}": minus_di
        }
        
        return FeatureResult(visuals=visuals, data=data_dict)
