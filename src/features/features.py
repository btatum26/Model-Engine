from .momentum.stochastic import Stochastic
from .momentum.rsi import RSI
from .momentum.roc import ROC
from .momentum.macd import MACD

from .volume.vwap import VWAP
from .volume.obv import OBV
from .volume.anchored_vwap import AnchoredVWAP
from .volume.volume_zscore import VolumeZScore

from .volatility.keltner_channels import KeltnerChannels
from .volatility.bollinger_bands import BollingerBands
from .volatility.atr import AverageTrueRange

from .trend.moving_avg import MovingAverage
from .trend.adx import ADX

from .levels.support_resistance import SupportResistance

AVAILABLE_FEATURES = {
    "Stochastic": Stochastic,
    "RSI": RSI,
    "ROC": ROC,
    "MACD": MACD,
    "VWAP": VWAP,
    "OBV": OBV,
    "AnchoredVWAP": AnchoredVWAP,
    "VolumeZScore": VolumeZScore,
    "KeltnerChannels": KeltnerChannels,
    "BollingerBands": BollingerBands,
    "ATR": AverageTrueRange,
    "MovingAverage": MovingAverage,
    "ADX": ADX,
    "SupportResistance": SupportResistance
}
