from typing import Dict, Any, List
import pandas as pd
from ..base import Feature, LineOutput, FeatureResult

class ROC(Feature):
    @property
    def name(self) -> str:
        return "ROC"

    @property
    def description(self) -> str:
        return "Rate of Change (Percentage difference between current and n-period ago price)."

    @property
    def category(self) -> str:
        return "Oscillators (Momentum)"

    @property
    def target_pane(self) -> str:
        return "new"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 12,
            "normalize": "none",
            "color": "#00ffaa"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        period = int(params.get("period", 12))
        norm_method = params.get("normalize", "none")
        color = params.get("color", "#00ffaa")
        
        close = df['Close'] if 'Close' in df.columns else df['close']
        
        # ROC Calculation
        # ((Current Close - Close n periods ago) / Close n periods ago) * 100
        roc = close.pct_change(periods=period) * 100
        
        def clean(s): return s.where(pd.notnull(s), None).tolist()
        
        visuals = [
            LineOutput(
                name=f"ROC_{period}",
                data=clean(roc),
                color=color,
                width=2
            )
        ]
        
        # Apply systematic normalization
        final_data = self.normalize(df, roc, norm_method)
        
        col_name = f"ROC_{period}"
        return FeatureResult(visuals=visuals, data={col_name: final_data})
