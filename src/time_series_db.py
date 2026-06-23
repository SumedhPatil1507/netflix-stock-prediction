"""
PostgreSQL / TimescaleDB Time-Series Storage Layer.
Provides hardened persistence for market OHLCV data.
Falls back gracefully to SQLite if TimescaleDB connection params are missing or fail.
"""
from __future__ import annotations
import os
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

_is_sqlite = False
_sqlite_path = os.path.join("data", "timescale_cache.db")

def get_connection():
    """Get DB connection (psycopg2 for Postgres/TimescaleDB, sqlite3 for fallback)."""
    global _is_sqlite
    
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        _is_sqlite = True
        os.makedirs("data", exist_ok=True)
        return sqlite3.connect(_sqlite_path)
        
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=5
        )
        _is_sqlite = False
        return conn
    except Exception as e:
        logger.warning(f"Failed to connect to TimescaleDB ({e}). Falling back to local SQLite.")
        _is_sqlite = True
        os.makedirs("data", exist_ok=True)
        return sqlite3.connect(_sqlite_path)

def init_db() -> None:
    """Initialize the database tables and enable TimescaleDB hypertable if possible."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        if _is_sqlite:
            # SQLite schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    date TIMESTAMP,
                    ticker TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    stock_splits REAL,
                    PRIMARY KEY (date, ticker)
                )
            """)
            conn.commit()
            logger.info("SQLite storage initialized successfully.")
        else:
            # Postgres / TimescaleDB schema
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    date TIMESTAMP NOT NULL,
                    ticker VARCHAR(12) NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    stock_splits DOUBLE PRECISION
                );
            """)
            # Create composite unique index since TimescaleDB requires unique constraints to include partitioning column (date)
            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS uidx_date_ticker ON ohlcv_data (date, ticker);
            """)
            
            # Convert to hypertable
            try:
                cursor.execute("SELECT create_hypertable('ohlcv_data', 'date', if_not_exists => TRUE);")
            except Exception as he:
                # TimescaleDB extension might be loaded but create_hypertable might fail if not fully configured
                logger.warning(f"Could not convert to TimescaleDB hypertable ({he}). Using standard Postgres table.")
                
            conn.commit()
            logger.info("TimescaleDB storage initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        conn.rollback()
    finally:
        conn.close()

def save_ohlcv_data(df: pd.DataFrame, ticker: str) -> None:
    """
    Persist OHLCV DataFrame to storage.
    Expects DataFrame with a DatetimeIndex or a 'Date' column.
    """
    if df.empty:
        return
        
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'Date' column.")
            
    ticker = ticker.upper()
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        if _is_sqlite:
            # SQLite upsert
            for idx, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO ohlcv_data (date, ticker, open, high, low, close, volume, stock_splits)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(date, ticker) DO UPDATE SET
                        open=excluded.open,
                        high=excluded.high,
                        low=excluded.low,
                        close=excluded.close,
                        volume=excluded.volume,
                        stock_splits=excluded.stock_splits
                """, (
                    idx.strftime("%Y-%m-%d %H:%M:%S"),
                    ticker,
                    float(row.get("Open", 0)),
                    float(row.get("High", 0)),
                    float(row.get("Low", 0)),
                    float(row.get("Close", 0)),
                    float(row.get("Volume", 0)),
                    float(row.get("Stock Splits", 0))
                ))
            conn.commit()
            logger.info(f"Saved {len(df)} rows for {ticker} into SQLite storage.")
        else:
            # Postgres upsert
            for idx, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO ohlcv_data (date, ticker, open, high, low, close, volume, stock_splits)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, ticker) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        stock_splits = EXCLUDED.stock_splits
                """, (
                    idx.to_pydatetime(),
                    ticker,
                    float(row.get("Open", 0)),
                    float(row.get("High", 0)),
                    float(row.get("Low", 0)),
                    float(row.get("Close", 0)),
                    float(row.get("Volume", 0)),
                    float(row.get("Stock Splits", 0))
                ))
            conn.commit()
            logger.info(f"Saved {len(df)} rows for {ticker} into TimescaleDB storage.")
    except Exception as e:
        logger.error(f"Error saving OHLCV data for {ticker}: {e}")
        conn.rollback()
    finally:
        conn.close()

def load_ohlcv_data(ticker: str, days_back: int = 730) -> pd.DataFrame:
    """Load OHLCV data for a ticker from the database."""
    ticker = ticker.upper()
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    conn = get_connection()
    try:
        if _is_sqlite:
            query = """
                SELECT date, open, high, low, close, volume, stock_splits
                FROM ohlcv_data
                WHERE ticker = ? AND date >= ?
                ORDER BY date ASC
            """
            df = pd.read_sql_query(query, conn, params=(ticker, cutoff_date.strftime("%Y-%m-%d %H:%M:%S")), parse_dates=["date"])
        else:
            query = """
                SELECT date, open, high, low, close, volume, stock_splits
                FROM ohlcv_data
                WHERE ticker = %s AND date >= %s
                ORDER BY date ASC
            """
            df = pd.read_sql_query(query, conn, params=(ticker, cutoff_date), parse_dates=["date"])
            
        if df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Stock Splits"])
            
        df = df.rename(columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "stock_splits": "Stock Splits"
        })
        df = df.set_index("Date")
        return df
    except Exception as e:
        logger.error(f"Error loading OHLCV data for {ticker}: {e}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Stock Splits"])
    finally:
        conn.close()
