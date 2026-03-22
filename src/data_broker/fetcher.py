import yfinance as yf
import pandas as pd
import time
import random
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, proxies=None):
        self.proxies = proxies or []
        self.last_fetch_time = 0
        self.rate_limit_delay = 1.0

    def _get_proxy(self):
        return random.choice(self.proxies) if self.proxies else None

    def fetch_historical(self, ticker, interval, period="max", start=None, end=None):
        """
        Fetches historical data from yfinance with rate limiting.
        """
        elapsed = time.time() - self.last_fetch_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        
        mapping = {
            "1w": "1wk",
            "1d": "1d",
            "4h": "1h",
            "1h": "1h",
            "30m": "30m",
            "15m": "15m"
        }
        yf_interval = mapping.get(interval, "1d")
        
        try:
            proxy = self._get_proxy()
            stock = yf.Ticker(ticker)
            df = stock.history(interval=yf_interval, period=period, start=start, end=end)
            
            self.last_fetch_time = time.time()

            if df is None or df.empty:
                return pd.DataFrame()

            if interval == "4h":
                df = df.resample('4h', origin='start_day', closed='left', label='left').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
            return df
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_live_quote(self, ticker):
        """Fetches the latest available price."""
        try:
            stock = yf.Ticker(ticker)
            return stock.fast_info['last_price']
        except Exception as e:
            print(f"Error fetching live quote for {ticker}: {e}")
            return None
