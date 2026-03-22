import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union
from ..database import Database
from ..fetcher import DataFetcher

class DataBroker:
    """
    The Data Broker (Smart Hydration).
    Abstracts data gathering so the backtester never knows where the data came from.
    """
    def __init__(self, db_path: str = "data/stocks.db"):
        self.db = Database(db_path)
        self.fetcher = DataFetcher()

    def get_data(self, ticker: str, interval: str, 
                 start: Optional[Union[str, datetime]] = None, 
                 end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Smart Hydration Logic.
        """
        # Convert strings to datetime if needed
        if isinstance(start, str):
            start = datetime.strptime(start, '%Y-%m-%d')
        if isinstance(end, str):
            end = datetime.strptime(end, '%Y-%m-%d')
        
        # 1. Query Database
        print(f"      - Querying local database for {ticker} ({interval})...")
        df_db = self.db.get_data(ticker, interval, start, end)
        
        # 2. Gap Analysis
        needs_fetch = False
        
        if df_db.empty:
            print("      - Local database empty for this range.")
            needs_fetch = True
        else:
            db_start = df_db.index.min()
            db_end = df_db.index.max()
            print(f"      - Found local data from {db_start} to {db_end}.")
            
            # Simplified gap analysis: 
            if start and (db_start - start) > timedelta(days=1):
                print(f"      - GAP DETECTED: Requested start {start} is before local start {db_start}.")
                needs_fetch = True
            
            # Check for end gap (most common: data is old)
            now = datetime.now()
            target_end = end if end else now
            if (target_end - db_end) > timedelta(hours=24):
                print(f"      - GAP DETECTED: Local data ends at {db_end}, requested end is {target_end}.")
                needs_fetch = True

        # 3. Fetch & Fill
        if needs_fetch:
            print(f"      - Syncing missing data from yfinance...")
            # If we have NO data, fetch a default period or the requested range
            if df_db.empty:
                fetch_period = "max" if not start else None
                df_new = self.fetcher.fetch_historical(ticker, interval, period=fetch_period, start=start, end=end)
            else:
                # Just fetch from last DB entry to now/end
                last_ts = df_db.index.max()
                df_new = self.fetcher.fetch_historical(ticker, interval, start=last_ts, end=end)
            
            # 4. Cache
            if not df_new.empty:
                print(f"      - Sync complete. Saving {len(df_new)} new bars to database...")
                self.db.save_data(df_new, ticker, interval)
                # Re-query to get the full combined range
                df_db = self.db.get_data(ticker, interval, start, end)
            else:
                print("      - Warning: yfinance returned no new data.")

        return df_db
