import pandas as pd
import requests
from tqdm import tqdm
import time
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from .engine import TradingEngine

class DataSnapshot:
    def __init__(self, engine: TradingEngine):
        self.engine = engine

    def get_top_1000_tickers(self):
        """
        Fetches a list of tickers, prioritizing a local tickers.txt file.
        """
        local_file = "data/tickers.txt"
        if os.path.exists(local_file):
            print(f"Loading tickers from local file: {local_file}")
            with open(local_file, "r") as f:
                return [line.strip() for line in f if line.strip()]

        # could not find the local file
        tickers = set()
        sources = [
            # S&P 500
            "https://raw.githubusercontent.com/datasets/s-p-500-companies/master/data/constituents.csv",
            # S&P 400 (MidCap)
            "https://raw.githubusercontent.com/mfs-data/sp400-constituents/main/sp400.csv",
            # NASDAQ 100
            "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed.csv"
        ]

        print("Fetching ticker lists...")
        for url in sources:
            try:
                df = pd.read_csv(url)
                # Handle different column names across datasets
                symbol_col = next((c for c in df.columns if 'Symbol' in c or 'Ticker' in c), None)
                if symbol_col:
                    tickers.update(df[symbol_col].dropna().astype(str).tolist())
            except Exception as e:
                print(f"Warning: Could not fetch from {url}: {e}")

        # Clean tickers (remove duplicates, handle weird characters like '.' for '-' in yfinance)
        cleaned_tickers = []
        for t in tickers:
            t = str(t).strip().replace('.', '-')
            if t and len(t) <= 5 and t.isalpha():
                cleaned_tickers.append(t)

        final_list = sorted(list(set(cleaned_tickers)))
        
        # If we have too many, take the top 1000 (proxied by sorting, usually alphabetic here)
        if len(final_list) > 1000:
            return final_list[:1000]
        
        if not final_list:
            print("No tickers found online. Using manual fallback list.")
            return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "BRK-B", "JNJ", "V"]
            
        return final_list

    def _sync_ticker(self, ticker, intervals, period, six_months_ago):
        """
        Helper method to sync a single ticker across all intervals.
        """
        results = []
        for interval in intervals:
            try:
                # Check if data is already up to date
                latest_ts = self.engine.db.get_latest_timestamp(ticker, interval)
                if latest_ts and latest_ts > six_months_ago:
                    continue

                # Adjust period for yfinance limits
                current_period = period
                if interval in ["1h", "4h"]:
                    current_period = "2y"
                elif interval in ["30m", "15m"]:
                    current_period = "60d"
                
                retries = 3
                success = False
                while retries > 0 and not success:
                    try:
                        self.engine.sync_data(ticker, interval, period=current_period, quiet=True)
                        success = True
                    except Exception as e:
                        retries -= 1
                        if retries > 0:
                            time.sleep(1)
                
                results.append((interval, success))
            except Exception as e:
                results.append((interval, False))
        return ticker, results

    def run(self, period="10y", max_workers=15):
        tickers = self.get_top_1000_tickers()
        intervals = ["1w", "1d", "4h"]
        six_months_ago = datetime.now() - timedelta(days=180)
        
        print(f"Starting Multithreaded Snapshot for {len(tickers)} tickers (Workers: {max_workers})...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all ticker sync tasks
            future_to_ticker = {
                executor.submit(self._sync_ticker, ticker, intervals, period, six_months_ago): ticker 
                for ticker in tickers
            }
            
            # Use tqdm to track overall progress
            pbar = tqdm(total=len(tickers), desc="Syncing Tickers", unit="ticker")
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    ticker_done, results = future.result()
                    # Optional: Log failures if needed
                    # if any(not success for interval, success in results):
                    #     pbar.write(f"Partial failure for {ticker_done}")
                except Exception as e:
                    pbar.write(f"Error syncing {ticker}: {e}")
                finally:
                    pbar.update(1)
            
            pbar.close()

        print("\nSnapshot complete.")
