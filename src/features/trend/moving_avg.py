from typing import Dict, Any, List
import pandas as pd
from ..base import Feature, LineOutput, FeatureResult

class MovingAverage(Feature):
    @property
    def name(self) -> str:
        return "Moving Average"

    @property
    def description(self) -> str:
        return "Trend indicator (SMA, EMA)."

    @property
    def category(self) -> str:
        return "Trend Indicators"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 50,
            "type": "SMA",
            "normalize": "none",
            "color": "#ff9900"
        }
        
    @property
    def parameter_options(self) -> Dict[str, List[Any]]:
        return {
            "type": ["SMA", "EMA"]
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        period = int(params.get("period", 50))
        ma_type = params.get("type", "SMA")
        color = params.get("color", "#ff9900")
        norm_method = params.get("normalize", "none")
        
        close = df['Close'] if 'Close' in df.columns else df['close']
        
        # Calculate the requested type of Moving Average
        if ma_type == "EMA":
            ma = close.ewm(span=period, adjust=False).mean()
        else:
            ma = close.rolling(window=period).mean()
        
        visuals = [
            LineOutput(
                name=f"{ma_type}_{period}",
                data=ma.where(pd.notnull(ma), None).tolist(),
                color=color,
                width=2
            )
        ]
        
        # Apply systematic normalization
        final_data = self.normalize(df, ma, norm_method)
        
        col_name = f"Dist_{ma_type}_{period}" if norm_method == "pct_distance" else f"{ma_type}_{period}"
        return FeatureResult(visuals=visuals, data={col_name: final_data})
