import pandas as pd
from typing import Optional, Dict, Any

class SignalModel:
    """
    Base class for all signal generation models.
    """
    def __init__(self):
        self.params = {}

    def generate_signals(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series]) -> pd.Series:
        """
        Takes a DataFrame and returns a Series of signals (-1, 0, 1).
        """
        raise NotImplementedError("Subclasses must implement generate_signals")
