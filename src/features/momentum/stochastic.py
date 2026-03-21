from typing import Dict, Any, List
import pandas as pd
from ..base import Feature, LineOutput, FeatureResult, register_feature

@register_feature("Stochastic")
class Stochastic(Feature):
    @property
    def name(self) -> str:
        return "Stochastic"

    @property
    def description(self) -> str:
        return "Stochastic Oscillator (%K and %D)."

    @property
    def category(self) -> str:
        return "Oscillators (Momentum)"

    @property
    def target_pane(self) -> str:
        return "new"

    @property
    def y_range(self) -> List[float]:
        return [0, 100]

    @property
    def y_padding(self) -> float:
        return 0.05

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "k_period": 14,
            "d_period": 3,
            "normalize": "none",
            "color_k": "#00ffff",
            "color_d": "#ff00ff"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        k_period = int(params.get("k_period", 14))
        d_period = int(params.get("d_period", 3))
        norm_method = params.get("normalize", "none")
        
        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        close = df['Close'] if 'Close' in df.columns else df['close']
        
        # Calculate the lowest low and highest high over the lookback period
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()
        
        # Fast Stochastic (%K) calculation
        k_percent = 100 * ((close - low_min) / (high_max - low_min).replace(0, 1e-9))
        
        # Slow Stochastic (%D) calculation (Moving Average of %K)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        visuals = [
            LineOutput(
                name="%K", 
                data=k_percent.where(pd.notnull(k_percent), None).tolist(), 
                color=params.get("color_k"), 
                width=1
            ),
            LineOutput(
                name="%D", 
                data=d_percent.where(pd.notnull(d_percent), None).tolist(), 
                color=params.get("color_d"), 
                width=1
            )
        ]
        
        # Apply systematic normalization
        final_k = self.normalize(df, k_percent, norm_method)
        final_d = self.normalize(df, d_percent, norm_method)
        
        col_k = "Norm_%K" if norm_method != "none" else "%K"
        col_d = "Norm_%D" if norm_method != "none" else "%D"

        return FeatureResult(visuals=visuals, data={col_k: final_k, col_d: final_d})
