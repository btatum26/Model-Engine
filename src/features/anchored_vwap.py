from typing import Dict, Any
import pandas as pd
from .base import Feature, LineOutput, FeatureResult

class AnchoredVWAP(Feature):
    @property
    def name(self) -> str:
        return "Anchored VWAP"

    @property
    def description(self) -> str:
        return "VWAP calculation starting from a specific number of bars ago."

    @property
    def category(self) -> str:
        return "Volume & Profile"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "anchor_bars_back": 100,
            "normalize": "none",
            "color": "#00d8ff"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        bars_back = int(params.get("anchor_bars_back", 100))
        color = params.get("color", "#00d8ff")
        norm_method = params.get("normalize", "none")
        
        # Determine the starting index for the anchored calculation
        if bars_back >= len(df):
            start_idx = 0
        else:
            start_idx = len(df) - bars_back
            
        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        close = df['Close'] if 'Close' in df.columns else df['close']
        volume = df['Volume'] if 'Volume' in df.columns else df['volume']
            
        # Typical Price calculation
        tp = (high + low + close) / 3
        v_tp = tp * volume
        
        # Calculate cumulative volume and price-volume product from the anchor point
        v_tp_slice = v_tp.iloc[start_idx:]
        vol_slice = volume.iloc[start_idx:]
        
        cum_v_tp = v_tp_slice.cumsum()
        cum_vol = vol_slice.cumsum()
        
        vwap_slice = cum_v_tp / cum_vol
        
        # Map the slice back to a full-length series aligned with the input DataFrame
        vwap_series = pd.Series(index=df.index, dtype=float)
        vwap_series.iloc[start_idx:] = vwap_slice
        
        visuals = [
            LineOutput(
                name=f"AVWAP ({bars_back})",
                data=vwap_series.where(pd.notnull(vwap_series), None).tolist(),
                color=color,
                width=2
            )
        ]
        
        # Apply normalization for machine learning features
        final_data = self.normalize(df, vwap_series, norm_method)
        
        col_name = f"Dist_AVWAP_{bars_back}" if norm_method == "pct_distance" else f"AVWAP_{bars_back}"
        return FeatureResult(visuals=visuals, data={col_name: final_data})
