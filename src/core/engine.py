import pandas as pd
from abc import ABC, abstractmethod

class SignalModel(ABC):
    """The strict interface for all user strategies."""
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        Must return a vectorized Pandas Series (float64) 
        ranging from -1.0 (Short) to 1.0 (Long).
        """
        pass
