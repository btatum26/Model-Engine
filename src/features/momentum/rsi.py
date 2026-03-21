from typing import Dict, Any, List
import pandas as pd
import numpy as np
from ..base import Feature, LineOutput, LevelOutput, FeatureResult, register_feature

@register_feature("RSI")
class RSI(Feature):
    @property
    def name(self) -> str:
        return "RSI"

    @property
    def description(self) -> str:
        return "Relative Strength Index (Wilder's Smoothing)."

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
            "period": 14,
            "overbought": 70,
            "oversold": 30,
            "normalize": "none",
            "color": "#aaff00"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], shared_cache: Dict[str, pd.Series] = None) -> FeatureResult:
        period = int(params.get("period", 14))
        ob = float(params.get("overbought", 70))
        os = float(params.get("oversold", 30))
        norm_method = params.get("normalize", "none")
        color = params.get("color", "#aaff00")
        
        close = df['Close'] if 'Close' in df.columns else df['close']
        
        # Calculate price changes
        delta = close.diff()
        
        # Calculate gains and losses using Wilder's Smoothing
        gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
        
        # Relative Strength (RS) and RSI calculation
        rs = gain / loss
        rsi = np.where(loss == 0, 100, 100 - (100 / (1 + rs)))
        rsi = pd.Series(rsi, index=df.index)
        
        visuals = [
            LineOutput(
                name=f"RSI_{period}",
                data=rsi.where(pd.notnull(rsi), None).tolist(),
                color=color,
                width=2
            ),
            LevelOutput(name="Overbought", min_price=ob, max_price=ob, color="#ff4444"),
            LevelOutput(name="Oversold", min_price=os, max_price=os, color="#44ff44")
        ]
        
        # Apply normalization to the bounded RSI series
        final_data = self.normalize(df, rsi, norm_method)
        
        col_name = f"Norm_RSI_{period}" if norm_method != "none" else f"RSI_{period}"
        return FeatureResult(visuals=visuals, data={col_name: final_data})
