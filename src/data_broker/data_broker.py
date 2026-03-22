import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union, Dict
from .database import Database
from .fetcher import DataFetcher

class DataBroker:
    """
    The Data Broker (Smart Hydration).
    Abstracts data gathering so the backtester never knows where the data came from.
    """
    def __init__(self, db_path: str = "data/stocks.db", padding: int = 300):
        self.db = Database(db_path)
        self.fetcher = DataFetcher()
        self.padding = padding # Standard warm-up pad for indicators

    def _get_padded_start(self, start: datetime, interval: str) -> datetime:
        """Subtracts the warm-up padding from the start date."""
        # Heuristic mapping for padding
        multiplier = {
            "15m": 0.25,
            "30m": 0.5,
            "1h": 1,
            "4h": 4,
            "1d": 24,
            "1w": 168
        }
        hours_to_subtract = self.padding * multiplier.get(interval, 24)
        return start - timedelta(hours=hours_to_subtract)

    def get_data(self, ticker: str, interval: str, 
                 start: Optional[Union[str, datetime]] = None, 
                 end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """Smart Hydration Logic with Indicator Warm-Up."""
        # Convert strings to datetime
        if isinstance(start, str):
            start = datetime.strptime(start, '%Y-%m-%d')
        if isinstance(end, str):
            end = datetime.strptime(end, '%Y-%m-%d')
        
        # Apply Padding for Indicator Warm-up
        fetch_start = start
        if start:
            fetch_start = self._get_padded_start(start, interval)
            print(f"      - Applying {self.padding} period warm-up pad. Adjusted start: {fetch_start}")

        # Query Database
        print(f"      - Querying local database for {ticker} ({interval})...")
        df_db = self.db.get_data(ticker, interval, fetch_start, end)
        
        # Gap Analysis
        needs_fetch = False
        if df_db.empty:
            print("      - Local database empty for this range.")
            needs_fetch = True
        else:
            db_start = df_db.index.min()
            db_end = df_db.index.max()
            print(f"      - Found local data from {db_start} to {db_end}.")

            if fetch_start and (db_start - fetch_start) > timedelta(hours=1):
                print(f"      - GAP DETECTED: Requested start {fetch_start} is before local start {db_start}.")
                needs_fetch = True
            
            now = datetime.now()
            target_end = end if end else now
            if (target_end - db_end) > timedelta(hours=24):
                print(f"      - GAP DETECTED: Local data ends at {db_end}, requested end is {target_end}.")
                needs_fetch = True

        # Fetch and fill missing data
        if needs_fetch:
            print(f"      - Syncing missing data from yfinance...")
            if df_db.empty:
                fetch_period = "max" if not fetch_start else None
                df_new = self.fetcher.fetch_historical(ticker, interval, period=fetch_period, start=fetch_start, end=end)
            else:
                last_ts = df_db.index.max()
                df_new = self.fetcher.fetch_historical(ticker, interval, start=last_ts, end=end)
            
            # Cache results
            if not df_new.empty:
                print(f"      - Sync complete. Saving {len(df_new)} new bars to database...")
                self.db.save_data(df_new, ticker, interval)
                df_db = self.db.get_data(ticker, interval, fetch_start, end)
            else:
                print("      - Warning: yfinance returned no new data.")

        return df_db
    
    def sync_historical(self, ticker: str, interval: str) -> bool:
        """
        Forces a fetch of all missing historical data up to the current moment.
        """
        pass
        
    def get_cached_inventory(self) -> Dict:
        """
        Returns a dictionary of what is in the DB.
        """
        pass
