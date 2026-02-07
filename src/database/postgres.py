"""
PostgreSQL database operations for stock simulator.
Uses SQLAlchemy with connection pooling.
"""
import json
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Float, Date, Integer, JSON
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker, Session

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DatabaseConfig, LOGS_DIR

import loguru
logger = loguru.logger

# Create engine with connection pooling
engine = create_engine(
    DatabaseConfig.get_postgres_uri(),
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()


# Define tables
daily_prices_table = Table(
    'daily_prices', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('stock_code', String(20), nullable=False, index=True),
    Column('date', Date, nullable=False, index=True),
    Column('open', Float),
    Column('high', Float),
    Column('low', Float),
    Column('close', Float),
    Column('volume', Float),
    Column('amount', Float),
    Column('created_at', Date, default=datetime.utcnow),
)


alpha_results_table = Table(
    'alpha_results', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('stock_code', String(20), nullable=False, index=True),
    Column('date', Date, nullable=False, index=True),
    Column('alpha_name', String(50), nullable=False),
    Column('alpha_value', Float),
    Column('created_at', Date, default=datetime.utcnow),
)


backtest_results_table = Table(
    'backtest_results', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('backtest_id', String(50), nullable=False, unique=True),
    Column('factor_list', JSON),  # List of factor names used
    Column('start_date', Date),
    Column('end_date', Date),
    Column('initial_capital', Float),
    Column('final_capital', Float),
    Column('annual_return', Float),
    Column('sharpe_ratio', Float),
    Column('max_drawdown', Float),
    Column('win_rate', Float),
    Column('total_trades', Integer),
    Column('results_json', JSON),  # Full results as JSON
    Column('created_at', Date, default=datetime.utcnow),
)


simulator_portfolio_table = Table(
    'simulator_portfolio', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('date', Date, nullable=False, index=True),
    Column('stock_code', String(20), nullable=False),
    Column('position_value', Float),
    Column('shares', Integer),
    Column('avg_price', Float),
    Column('signal', Float),  # Alpha signal value
    Column('created_at', Date, default=datetime.utcnow),
)


def init_db():
    """Initialize database tables"""
    try:
        metadata.create_all(engine)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


@contextmanager
def get_db_session() -> Session:
    """Get database session context manager"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()


def save_daily_data(stock_code: str, df: pd.DataFrame) -> int:
    """
    Save daily price data to database.
    
    Args:
        stock_code: Stock symbol (e.g., '000001')
        df: DataFrame with columns [date, open, high, low, close, volume, amount]
    
    Returns:
        Number of rows inserted
    """
    if df.empty:
        return 0
    
    try:
        df = df.copy()
        df['stock_code'] = stock_code
        
        # Insert using raw SQL for performance
        with get_db_session() as session:
            # Use COPY or bulk insert
            for _, row in df.iterrows():
                stmt = text("""
                    INSERT INTO daily_prices 
                    (stock_code, date, open, high, low, close, volume, amount)
                    VALUES (:stock_code, :date, :open, :high, :low, :close, :volume, :amount)
                    ON CONFLICT (stock_code, date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        amount = EXCLUDED.amount
                """)
                session.execute(stmt, {
                    'stock_code': row['stock_code'],
                    'date': pd.to_datetime(row['date']).date() if isinstance(row['date'], str) else row['date'].date(),
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'amount': row['amount']
                })
        
        logger.info(f"Saved {len(df)} rows for {stock_code}")
        return len(df)
        
    except Exception as e:
        logger.error(f"Failed to save daily data for {stock_code}: {e}")
        raise


def load_daily_data(
    stock_code: str, 
    start_date: str, 
    end_date: str
) -> pd.DataFrame:
    """
    Load daily price data from database.
    
    Args:
        stock_code: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with daily price data
    """
    try:
        query = text("""
            SELECT stock_code, date, open, high, low, close, volume, amount
            FROM daily_prices
            WHERE stock_code = :stock_code
            AND date >= :start_date
            AND date <= :end_date
            ORDER BY date ASC
        """)
        
        with get_db_session() as session:
            df = pd.read_sql(query, session.bind, params={
                'stock_code': stock_code,
                'start_date': start_date,
                'end_date': end_date
            })
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        
        logger.debug(f"Loaded {len(df)} rows for {stock_code} from {start_date} to {end_date}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load daily data for {stock_code}: {e}")
        raise


def save_alpha_results(stock_code: str, date: str, alpha_values: Dict[str, float]) -> int:
    """
    Save alpha factor calculation results.
    
    Args:
        stock_code: Stock symbol
        date: Date (YYYY-MM-DD)
        alpha_values: Dict of {alpha_name: value}
    
    Returns:
        Number of records saved
    """
    try:
        with get_db_session() as session:
            count = 0
            for alpha_name, value in alpha_values.items():
                if value is not None and not pd.isna(value):
                    stmt = text("""
                        INSERT INTO alpha_results (stock_code, date, alpha_name, alpha_value)
                        VALUES (:stock_code, :date, :alpha_name, :alpha_value)
                        ON CONFLICT DO NOTHING
                    """)
                    session.execute(stmt, {
                        'stock_code': stock_code,
                        'date': date,
                        'alpha_name': alpha_name,
                        'alpha_value': float(value)
                    })
                    count += 1
        
        logger.debug(f"Saved {count} alpha values for {stock_code} on {date}")
        return count
        
    except Exception as e:
        logger.error(f"Failed to save alpha results: {e}")
        raise


def load_alpha_results(date: str, stock_codes: List[str] = None) -> pd.DataFrame:
    """
    Load alpha results for a specific date.
    
    Args:
        date: Date (YYYY-MM-DD)
        stock_codes: Optional list of stock codes to filter
    
    Returns:
        DataFrame with columns [stock_code, date, alpha_name, alpha_value]
    """
    try:
        query = text("""
            SELECT stock_code, date, alpha_name, alpha_value
            FROM alpha_results
            WHERE date = :date
        """)
        
        params = {'date': date}
        
        if stock_codes:
            placeholders = ', '.join([f':code{i}' for i in range(len(stock_codes))])
            query += f' AND stock_code IN ({placeholders})'
            for i, code in enumerate(stock_codes):
                params[f'code{i}'] = code
        
        with get_db_session() as session:
            df = pd.read_sql(query, session.bind, params=params)
        
        if not df.empty:
            df = df.pivot(index='stock_code', columns='alpha_name', values='alpha_value')
            df.index.name = None
            df.columns.name = None
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load alpha results for {date}: {e}")
        raise


def save_backtest_result(backtest_id: str, results: Dict[str, Any]) -> bool:
    """
    Save backtest result to database.
    
    Args:
        backtest_id: Unique identifier for this backtest
        results: Dict containing backtest metrics and details
    
    Returns:
        True if successful
    """
    try:
        with get_db_session() as session:
            stmt = text("""
                INSERT INTO backtest_results 
                (backtest_id, factor_list, start_date, end_date, initial_capital, 
                 final_capital, annual_return, sharpe_ratio, max_drawdown, 
                 win_rate, total_trades, results_json)
                VALUES (:backtest_id, :factor_list, :start_date, :end_date, :initial_capital,
                        :final_capital, :annual_return, :sharpe_ratio, :max_drawdown,
                        :win_rate, :total_trades, :results_json)
                ON CONFLICT (backtest_id) DO UPDATE SET
                    factor_list = EXCLUDED.factor_list,
                    start_date = EXCLUDED.start_date,
                    end_date = EXCLUDED.end_date,
                    initial_capital = EXCLUDED.initial_capital,
                    final_capital = EXCLUDED.final_capital,
                    annual_return = EXCLUDED.annual_return,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    max_drawdown = EXCLUDED.max_drawdown,
                    win_rate = EXCLUDED.win_rate,
                    total_trades = EXCLUDED.total_trades,
                    results_json = EXCLUDED.results_json
            """)
            
            session.execute(stmt, {
                'backtest_id': backtest_id,
                'factor_list': json.dumps(results.get('factor_list', [])),
                'start_date': results.get('start_date'),
                'end_date': results.get('end_date'),
                'initial_capital': results.get('initial_capital'),
                'final_capital': results.get('final_capital'),
                'annual_return': results.get('annual_return'),
                'sharpe_ratio': results.get('sharpe_ratio'),
                'max_drawdown': results.get('max_drawdown'),
                'win_rate': results.get('win_rate'),
                'total_trades': results.get('total_trades'),
                'results_json': json.dumps(results)
            })
        
        logger.info(f"Saved backtest result: {backtest_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save backtest result: {e}")
        raise


def load_backtest_results(limit: int = 100) -> pd.DataFrame:
    """
    Load all backtest results.
    
    Args:
        limit: Maximum number of results to return
    
    Returns:
        DataFrame with backtest results
    """
    try:
        query = text("""
            SELECT backtest_id, factor_list, start_date, end_date, 
                   initial_capital, final_capital, annual_return, sharpe_ratio,
                   max_drawdown, win_rate, total_trades, created_at
            FROM backtest_results
            ORDER BY created_at DESC
            LIMIT :limit
        """)
        
        with get_db_session() as session:
            df = pd.read_sql(query, session.bind, params={'limit': limit})
        
        # Parse factor list
        if 'factor_list' in df.columns:
            df['factor_list'] = df['factor_list'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load backtest results: {e}")
        raise


def save_simulator_portfolio(
    date: str, 
    portfolio: List[Dict[str, Any]]
) -> int:
    """
    Save simulator portfolio for a given date.
    
    Args:
        date: Date (YYYY-MM-DD)
        portfolio: List of position dicts
    
    Returns:
        Number of records saved
    """
    try:
        with get_db_session() as session:
            count = 0
            for position in portfolio:
                stmt = text("""
                    INSERT INTO simulator_portfolio 
                    (date, stock_code, position_value, shares, avg_price, signal)
                    VALUES (:date, :stock_code, :position_value, :shares, :avg_price, :signal)
                """)
                session.execute(stmt, {
                    'date': date,
                    'stock_code': position['stock_code'],
                    'position_value': position.get('position_value', 0),
                    'shares': position.get('shares', 0),
                    'avg_price': position.get('avg_price', 0),
                    'signal': position.get('signal', 0)
                })
                count += 1
        
        logger.debug(f"Saved {count} portfolio positions for {date}")
        return count
        
    except Exception as e:
        logger.error(f"Failed to save simulator portfolio: {e}")
        raise


def load_latest_portfolio(date: str = None) -> pd.DataFrame:
    """
    Load latest portfolio positions.
    
    Args:
        date: Optional specific date
    
    Returns:
        DataFrame with portfolio positions
    """
    try:
        if date:
            query = text("""
                SELECT date, stock_code, position_value, shares, avg_price, signal
                FROM simulator_portfolio
                WHERE date = :date
                ORDER BY stock_code
            """)
            params = {'date': date}
        else:
            # Get latest date
            query = text("""
                SELECT date, stock_code, position_value, shares, avg_price, signal
                FROM simulator_portfolio
                WHERE date = (SELECT MAX(date) FROM simulator_portfolio)
                ORDER BY stock_code
            """)
            params = {}
        
        with get_db_session() as session:
            df = pd.read_sql(query, session.bind, params=params)
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load portfolio: {e}")
        raise


if __name__ == "__main__":
    # Initialize database tables
    init_db()
    print("Database initialized successfully")
