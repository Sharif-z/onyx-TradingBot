# config.py
import os

class TalksyConfig:
    # Asset Settings
    TICKERS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "LINK/USDT"]
    TICKER = "BTC/USDT"  # Default focus ticker
    EXCHANGE_ID = "binance"
    TICKERS_CCXT = {
        "BTC/USDT": "BTC/USDT:USDT",
        "ETH/USDT": "ETH/USDT:USDT",
        "SOL/USDT": "SOL/USDT:USDT",
        "BNB/USDT": "BNB/USDT:USDT",
        "LINK/USDT": "LINK/USDT:USDT"
    }
    TICKER_CCXT = "BTC/USDT:USDT"  # Legacy single ticker CCXT fallback
    TIMEFRAME_PRIMARY = "15m"
    TICKING_INTERVAL_SECS = 10  # 10 seconds
    
    # Strategy Settings
    RISK_REWARD_RATIO = 1.5
    SWING_LOOKBACK = 10  # 15-minute candles
    SL_ATR_CUSHION = 1.5
    TREND_LEN = 200
    TREND_TYPE = "HMA"  # Options: "HMA", "EMA"
    USE_MACRO_TREND_FILTER = False  # Set to False to disable 1-Hour HMA filter checking during live execution
    PANIC_THRESHOLD = 75  # 75% bearish score triggers panic long exit, 75% bullish triggers short exit
    
    # Session Window Settings
    USE_SESSION_FILTER = False
    # Allowed trading windows in UTC (6:30 PM - 9:30 PM IST is 13:00 - 16:00 UTC)
    # (12:30 PM - 2:30 PM IST is 07:00 - 09:00 UTC)
    # (5:30 AM - 8:30 AM IST is 00:00 - 03:00 UTC)
    ALLOWED_SESSION_WINDOWS = [
        ("13:00", "16:00"),  # London-NY Overlap (High action)
        ("07:00", "09:00"),  # Asia-London Transition
        ("00:00", "03:00")   # Asian Session Opening (Calm moves)
    ]
    
    # Account Settings
    INITIAL_BALANCE = 10000.0
    TRADE_SIZE_BTC = 0.1     # Standard position size in BTC contracts
    MARGIN_PER_TRADE = 0.10  # Legacy config option (margin in USD)
    LEVERAGE = 50.0          # Leverage factor 50:1
    
    # Account & Risk Parameters
    TOTAL_CAPITAL = 10000.0       # Base capital allocation for risk calculation
    MAX_RISK_PCT = 1.5           # Max risk per trade (1.5% = $150 risk exposure)
    
    POSITION_LIMITS = {
        "BTC/USDT": {"min": 0.1, "max": 0.5, "round_digits": 1},
        "ETH/USDT": {"min": 1.0, "max": 5.0, "round_digits": 1},
        "SOL/USDT": {"min": 10.0, "max": 50.0, "round_digits": 0},
        "BNB/USDT": {"min": 2.0, "max": 10.0, "round_digits": 1},
        "LINK/USDT": {"min": 50.0, "max": 250.0, "round_digits": 0}
    }
    
    SENTIMENT_GATES = {
        "BTC/USDT": 52.0,
        "ETH/USDT": 55.0,
        "SOL/USDT": 54.0,
        "BNB/USDT": 60.0,
        "LINK/USDT": 60.0
    }
    
    # Web / Dashboard Settings
    WEB_HOST = "0.0.0.0"
    WEB_PORT = 8000
    LOOKBACK_DAYS = "7d"
    
    # Simulation / Live settings
    SIMULATION_MODE = False  # Set to True for fast-ticking simulation
    SIMULATION_TICK_SECS = 0.5  # Accelerated ticking for test runs
    
    # Machine Learning Gatekeeper Settings
    USE_ML_GATEKEEPER = True      # Enable ML probabilistic classifier filter
    ML_DRY_RUN = False            # False: Actively block trades that fail ML confidence score
    ML_CONFIDENCE_THRESHOLD = 0.65  # Minimum Safe Trade probability (P_Safe >= 65%)
    
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
    
    # Storage Settings
    LEDGER_FILE = os.path.join(PROJECT_ROOT, "data", "trade_ledger.csv")
