from typing import Dict, Any, List
import pandas as pd
from ..base import Feature, LineOutput, FeatureResult

class VolumeZScore(Feature):
    @property
    def name(self) -> str:
        return "Volume Z-Score"

    @property
    def description(self) -> str:
        return "Identifies abnormal volume spikes using Z-Score normalization."

    @property
    def category(self) -> str:
        return "Volume Indicators"

    @property
    def target_pane(self) -> str:
        return "new"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 20,
            "color": "#ffaa00"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        period = int(params.get("period", 20))
        color = params.get("color", "#ffaa00")
        
        volume = df['Volume'] if 'Volume' in df.columns else df['volume']
        
        # Calculate Z-Score manually or via base class normalization
        # Here we do it manually to ensure we are using Volume
        z_score = self.normalize(df, volume, "z_score")
        
        def clean(s): return s.where(pd.notnull(s), None).tolist()
        
        visuals = [
            LineOutput(
                name=f"Vol_ZScore_{period}",
                data=clean(z_score),
                color=color,
                width=2
            )
        ]
        
        return FeatureResult(visuals=visuals, data={f"Vol_ZScore_{period}": z_score})
