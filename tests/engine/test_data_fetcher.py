import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.data_broker.fetcher import DataFetcher

@pytest.fixture
def fetcher():
    return DataFetcher()

def test_interval_mapping(fetcher):
    with patch('yfinance.Ticker') as mock_ticker:
        mock_stock = MagicMock()
        mock_ticker.return_value = mock_stock
        mock_stock.history.return_value = pd.DataFrame({'Close': [10, 20]}, index=pd.date_range('2023-01-01', periods=2))
        
        fetcher.fetch_historical("AAPL", "1w")
        mock_stock.history.assert_called_with(interval="1wk", period="max", start=None, end=None)
        
        fetcher.fetch_historical("AAPL", "1d")
        mock_stock.history.assert_called_with(interval="1d", period="max", start=None, end=None)
        
        fetcher.fetch_historical("AAPL", "4h")
        mock_stock.history.assert_called_with(interval="1h", period="max", start=None, end=None)

def test_resampling_4h(fetcher):
    # Create 8 hours of 1h data
    dates = pd.date_range('2023-01-01 00:00:00', periods=8, freq='h')
    data = {
        'Open': range(8),
        'High': range(10, 18),
        'Low': range(8),
        'Close': range(5, 13),
        'Volume': [100] * 8
    }
    df_1h = pd.DataFrame(data, index=dates)
    
    with patch('yfinance.Ticker') as mock_ticker:
        mock_stock = MagicMock()
        mock_ticker.return_value = mock_stock
        mock_stock.history.return_value = df_1h
        
        df_4h = fetcher.fetch_historical("AAPL", "4h")
        
        assert len(df_4h) == 2
        assert df_4h.index[0] == pd.Timestamp('2023-01-01 00:00:00')
        assert df_4h.index[1] == pd.Timestamp('2023-01-01 04:00:00')
        assert df_4h.iloc[0]['Open'] == 0
        assert df_4h.iloc[0]['High'] == 13 # max of 10, 11, 12, 13
        assert df_4h.iloc[1]['Close'] == 12 # last of 9, 10, 11, 12

def test_timezone_handling(fetcher):
    dates = pd.date_range('2023-01-01', periods=2, tz='America/New_York')
    df_tz = pd.DataFrame({'Close': [10, 20]}, index=dates)
    
    with patch('yfinance.Ticker') as mock_ticker:
        mock_stock = MagicMock()
        mock_ticker.return_value = mock_stock
        mock_stock.history.return_value = df_tz
        
        df = fetcher.fetch_historical("AAPL", "1d")
        
        assert df.index.tz is None
        # New York is UTC-5 in Jan
        assert df.index[0] == pd.Timestamp('2023-01-01 05:00:00')

def test_fetch_error_handling(fetcher):
    with patch('yfinance.Ticker') as mock_ticker:
        mock_stock = MagicMock()
        mock_ticker.return_value = mock_stock
        mock_stock.history.side_effect = Exception("API Down")
        
        df = fetcher.fetch_historical("AAPL", "1d")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

def test_fetch_live_quote(fetcher):
    with patch('yfinance.Ticker') as mock_ticker:
        mock_stock = MagicMock()
        mock_ticker.return_value = mock_stock
        mock_stock.fast_info = {'last_price': 150.5}
        
        price = fetcher.fetch_live_quote("AAPL")
        assert price == 150.5
