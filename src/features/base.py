from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Type
import pandas as pd

# Global Registry
FEATURE_REGISTRY: Dict[str, Type['Feature']] = {}

def register_feature(name: str):
    """
    Decorator to register a Feature class into the global registry.
    Usage:
    @register_feature("RSI")
    class RSI(Feature):
        ...
    """
    def decorator(cls: Type['Feature']):
        FEATURE_REGISTRY[name] = cls
        return cls
    return decorator

# --- Output Types ---
@dataclass
class FeatureOutput:
    name: str
    color: str = 'white'

@dataclass
class LineOutput(FeatureOutput):
    data: List[float] = None # Matches length of DF
    width: int = 1

@dataclass
class LevelOutput(FeatureOutput):
    price: float = 0.0
    min_price: float = 0.0
    max_price: float = 0.0
    strength: float = 0.0

@dataclass
class MarkerOutput(FeatureOutput):
    indices: List[int] = None
    values: List[float] = None
    shape: str = 'o' # 'o', 't', 's', 'd', '+', 'x'

@dataclass
class HeatmapOutput(FeatureOutput):
    """
    Represents a vertical density gradient (e.g. for KDE).
    price_grid: Array of price points (Y-axis)
    density: Array of density values (0.0 to 1.0) for each price point.
    """
    price_grid: List[float] = None
    density: List[float] = None
    color_map: str = 'plasma' # matplotlib colormap name or similar

@dataclass
class FeatureResult:
    visuals: List[FeatureOutput]
    data: Dict[str, pd.Series] = None # Raw numerical data for signal extraction

# --- Base Feature Class ---
class Feature(ABC):
    """
    Abstract Base Class for all Stock Bot Features.
    """
    @staticmethod
    def generate_column_name(feature_id: str, params: Dict[str, Any], output_name: Optional[str] = None) -> str:
        """
        Standardizes column naming across the system.
        Example: RSI + {period: 14} -> RSI_14
        Example: MACD + {fast: 12, slow: 26} + signal -> MACD_12_26_SIGNAL
        """
        # Prefix for normalized features
        norm = params.get("normalize", "none")
        prefix = "Norm_" if norm != "none" else ""
        
        # Identify core parameters (ignoring visual ones like color)
        # We assume parameters that affect calculation are integers or floats
        # and not strings like color or normalize
        core_params = {k: v for k, v in params.items() if k not in ["color", "normalize", "overbought", "oversold"]}
        
        # Simple Case: If there's only a 'period' or 'window', just use that number
        if len(core_params) == 1 and ("period" in core_params or "window" in core_params):
            val = core_params.get("period") or core_params.get("window")
            base = f"{feature_id}_{val}"
        elif not core_params:
            base = feature_id
        else:
            # Complex Case: Join all sorted core params
            param_str = "_".join([f"{v}" for k, v in sorted(core_params.items())])
            base = f"{feature_id}_{param_str}"
            
        suffix = f"_{output_name.upper()}" if output_name else ""
        return f"{prefix}{base}{suffix}"

    @property
    def target_pane(self) -> str:
        """
        'main': Overlay on price chart.
        'new': Create a new subplot below.
        """
        return "main"

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of the feature."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what the feature does."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Category of the feature (e.g., 'Price Levels', 'Trend', 'Volume')."""
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Dictionary of default parameters.
        Example: {'window': 14, 'threshold': 0.01}
        """
        return {}

    @property
    def parameter_options(self) -> Dict[str, Dict[str, Any]]:
        """
        Optional metadata for parameters (e.g., min, max, type).
        Example: {'window': {'min': 1, 'max': 100, 'type': 'int'}}
        """
        return {}

    @property
    def y_range(self) -> Optional[List[float]]:
        """
        Fixed Y-axis range [min, max].
        None if dynamic/data-driven.
        """
        return None

    @property
    def y_padding(self) -> float:
        """
        Default Y-axis padding (0.1 = 10% of data height).
        """
        return 0.1
    
    def normalize(self, df: pd.DataFrame, series: pd.Series, method: str, window: int = 20) -> pd.Series:
        """
        Systematically normalizes raw indicator data for Machine Learning.
        """
        if method == "none" or not method:
            return series
            
        # Ensure we have consistent column access
        close = df['Close'] if 'Close' in df.columns else df['close']
            
        # Percentage Distance from Price (For MAs, VWAP, Support/Resistance)
        if method == "pct_distance":
            # (Price - Indicator) / Indicator -> Output is a % (e.g., +0.02 means price is 2% above MA)
            return (close - series) / series.replace(0, 1e-9)
            
        # Ratio to Price (For ATR, Bollinger Width)
        elif method == "price_ratio":
            # Indicator / Price -> (e.g., ATR is 1.5% of the current stock price)
            return series / close.replace(0, 1e-9)
            
        # Z-Score (For Volume, or unbounded oscillators)
        elif method == "z_score":
            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std().replace(0, 1e-9)
            return (series - rolling_mean) / rolling_std
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @abstractmethod
    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        """
        Main logic. Receives OHLCV DataFrame, parameters, and the FeatureCache.
        Returns a FeatureResult containing visual outputs and raw data.
        """
        pass