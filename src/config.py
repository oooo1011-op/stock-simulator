"""
Configuration module for stock simulator.
Loads settings from environment variables.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Database Configuration
class DatabaseConfig:
    """Database settings"""
    
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "192.168.2.151")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "15432"))
    POSTGRES_USER = os.getenv("POSTGRES_USER", "oooo")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "543356")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "stock")
    
    @classmethod
    def get_postgres_uri(cls) -> str:
        """Get PostgreSQL connection URI"""
        return f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
    
    REDIS_HOST = os.getenv("REDIS_HOST", "192.168.2.151")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "16379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))


# Backtest Configuration
class BacktestConfig:
    """Backtest parameters"""
    
    INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))  # 10万
    TRANSACTION_FEE = float(os.getenv("TRANSACTION_FEE", "0.0005"))  # 万5
    SLIPPAGE = float(os.getenv("SLIPPAGE", "0.001"))  # 0.1%
    
    # Date range
    BACKTEST_START_DATE = os.getenv("BACKTEST_START_DATE", "20080101")
    BACKTEST_END_DATE = os.getenv("BACKTEST_END_DATE", "20251231")
    
    # Number of positions
    MIN_POSITIONS = int(os.getenv("MIN_POSITIONS", "5"))
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "10"))


# Simulator Configuration
class SimulatorConfig:
    """Live simulator parameters"""
    
    INITIAL_CAPITAL = float(os.getenv("SIM_INITIAL_CAPITAL", "100000"))
    TRANSACTION_FEE = float(os.getenv("SIM_TRANSACTION_FEE", "0.0005"))
    SLIPPAGE = float(os.getenv("SIM_SLIPPAGE", "0.001"))
    
    # Rebalancing
    REBALANCE_FREQUENCY = os.getenv("REBALANCE_FREQUENCY", "weekly")  # daily/weekly/monthly
    REBALANCE_DAY = int(os.getenv("REBALANCE_DAY", "0"))  # 0=Monday
    
    # Positions
    MIN_POSITIONS = int(os.getenv("SIM_MIN_POSITIONS", "5"))
    MAX_POSITIONS = int(os.getenv("SIM_MAX_POSITIONS", "10"))


# Factor Configuration
class FactorConfig:
    """Alpha factor settings"""
    
    # Number of top factors to select after backtest
    TOP_FACTORS_COUNT = int(os.getenv("TOP_FACTORS_COUNT", "10"))
    
    # Filter criteria
    MIN_ANNUAL_RETURN = float(os.getenv("MIN_ANNUAL_RETURN", "0.15"))  # 15%
    MIN_SHARPE_RATIO = float(os.getenv("MIN_SHARPE_RATIO", "1.0"))
    MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.20"))  # 20%
    MIN_WIN_RATE = float(os.getenv("MIN_WIN_RATE", "0.50"))  # 50%


# Data Configuration
class DataConfig:
    """Data source settings"""
    
    DEFAULT_START_DATE = "20080101"
    DATA_SOURCE = os.getenv("DATA_SOURCE", "tushare")  # tushare, akshare, baostock
    REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.34"))  # Tushare: 0.34s for 180 req/min
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
    # Data source priority
    SOURCE_PRIORITY = ['tushare', 'akshare', 'baostock']


class TushareConfig:
    """Tushare Pro settings"""
    
    TOKEN = os.getenv("TUSHARE_APIKEY") or os.getenv("TUSHARE_TOKEN")
    RATE_LIMIT_PER_MINUTE = 180  # Leave 10% margin from 200
    RATE_LIMIT_PER_DAY = 100000  # Per API


# Logging Configuration
class LoggingConfig:
    """Logging settings"""
    
    LEVEL = os.getenv("LOG_LEVEL", "INFO")
    FORMAT = os.getenv("LOG_FORMAT", "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    FILE = LOGS_DIR / "stock_simulator.log"


# Export commonly used config
db_config = DatabaseConfig()
backtest_config = BacktestConfig()
simulator_config = SimulatorConfig()
factor_config = FactorConfig()
data_config = DataConfig()
logging_config = LoggingConfig()
