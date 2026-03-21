from typing import Dict, Any, List
import pandas as pd
from ..base import Feature, LineOutput, FeatureResult, register_feature

@register_feature("MACD")
class MACD(Feature):
    @property
    def name(self) -> str:
        return "MACD"

    @property
    def description(self) -> str:
        return "Moving Average Convergence Divergence. Includes Signal Line and Histogram."

    @property
    def category(self) -> str:
        return "Oscillators (Momentum)"

    @property
    def target_pane(self) -> str:
        return "new"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "normalize": "none",
            "color_macd": "#00d4ff",
            "color_signal": "#ff9900"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        fast = int(params.get("fast_period", 12))
        slow = int(params.get("slow_period", 26))
        signal = int(params.get("signal_period", 9))
        norm_method = params.get("normalize", "none")
        
        # Calculate MACD Components using Cache
        if cache:
            ema_fast = cache.get_series("EMA", {"period": fast}, df)
            ema_slow = cache.get_series("EMA", {"period": slow}, df)
        else:
            close = df['Close'] if 'Close' in df.columns else df['close']
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        
        # Signal line is an EMA of the MACD line. 
        # Since cache.get_series expects a feature registered in the registry, 
        # and we don't have a feature that calculates EMA of an arbitrary series (only Close),
        # we compute the signal line here manually or we could potentially register a "SignalEMA" feature.
        # For now, let's keep it simple.
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        def clean(s): return s.where(pd.notnull(s), None).tolist()
        
        visuals = [
            LineOutput(
                name=f"MACD_{fast}_{slow}",
                data=clean(macd_line),
                color=params.get("color_macd"),
                width=2
            ),
            LineOutput(
                name=f"Signal_{signal}",
                data=clean(signal_line),
                color=params.get("color_signal"),
                width=1
            )
        ]
        
        # Normalize data
        final_macd = self.normalize(df, macd_line, norm_method)
        final_signal = self.normalize(df, signal_line, norm_method)
        final_hist = self.normalize(df, histogram, norm_method)
        
        data_dict = {
            f"MACD_{fast}_{slow}": final_macd,
            f"MACD_Signal_{signal}": final_signal,
            f"MACD_Hist_{fast}_{slow}": final_hist
        }
        
        return FeatureResult(visuals=visuals, data=data_dict)
