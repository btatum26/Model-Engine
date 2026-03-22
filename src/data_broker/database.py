from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, UniqueConstraint, Index, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os

Base = declarative_base()

class OHLCV(Base):
    __tablename__ = 'ohlcv'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    interval = Column(String)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    __table_args__ = (
        UniqueConstraint('ticker', 'timestamp', 'interval', name='_ticker_ts_interval_uc'),
        Index('idx_ticker_ts_interval', 'ticker', 'timestamp', 'interval'),
    )

class Database:
    def __init__(self, db_path="data/stocks.db"):
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.engine = create_engine(f"sqlite:///{db_path}")
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
        except Exception as e:
            print(f"Database initialization error: {e}")
            raise

    def save_data(self, df, ticker, interval):
        """Saves a pandas DataFrame to the database using an efficient bulk operation."""
        if df.empty:
            return

        save_df = df.copy()
        save_df['ticker'] = ticker
        save_df['interval'] = interval
        save_df = save_df.reset_index()
        
        ts_col = next((c for c in save_df.columns if c.lower() in ['date', 'datetime', 'timestamp']), None)
        if ts_col:
            save_df = save_df.rename(columns={ts_col: 'timestamp'})
        
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        save_df = save_df.rename(columns=column_mapping)
        
        required_cols = ['ticker', 'timestamp', 'interval', 'open', 'high', 'low', 'close', 'volume']
        save_df = save_df[required_cols]

        try:
            with self.engine.begin() as conn:
                save_df.to_sql('temp_ohlcv', conn, if_exists='replace', index=False)
                
                insert_stmt = text("""
                    INSERT OR IGNORE INTO ohlcv (ticker, timestamp, interval, open, high, low, close, volume)
                    SELECT ticker, timestamp, interval, open, high, low, close, volume FROM temp_ohlcv
                """)
                conn.execute(insert_stmt)
                
                conn.execute(text("DROP TABLE temp_ohlcv"))
        except Exception as e:
            print(f"Error bulk saving data: {e}")
            raise

    def get_latest_timestamp(self, ticker, interval):
        """Returns the most recent timestamp for a ticker/interval or None."""
        session = self.Session()
        try:
            result = session.query(OHLCV.timestamp).filter_by(
                ticker=ticker, 
                interval=interval
            ).order_by(OHLCV.timestamp.desc()).first()
            return result[0] if result else None
        finally:
            session.close()

    def get_all_tickers(self):
        """Returns a list of all distinct tickers in the database."""
        session = self.Session()
        try:
            results = session.query(OHLCV.ticker).distinct().all()
            return [r[0] for r in results]
        finally:
            session.close()

    def get_data(self, ticker, interval, start=None, end=None):
        """Retrieves data from the database as a pandas DataFrame."""
        try:
            session = self.Session()
            query = session.query(OHLCV).filter_by(ticker=ticker, interval=interval)
            if start:
                query = query.filter(OHLCV.timestamp >= start)
            if end:
                query = query.filter(OHLCV.timestamp <= end)
            
            results = query.order_by(OHLCV.timestamp).all()
            session.close()

            if not results:
                return pd.DataFrame()

            data = [{
                'Timestamp': r.timestamp,
                'Open': r.open,
                'High': r.high,
                'Low': r.low,
                'Close': r.close,
                'Volume': r.volume
            } for r in results]
            
            df = pd.DataFrame(data)
            df.set_index('Timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error retrieving data from database: {e}")
            return pd.DataFrame()
