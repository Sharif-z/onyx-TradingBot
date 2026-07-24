# src/data_source.py
import asyncio
import os
import time
import random
import pandas as pd
import numpy as np
import ccxt

def retry_ccxt_operation(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator that catches CCXT Network Errors and Rate Limit Exceeded exceptions,
    retrying with exponential backoff and random jitter.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.RateLimitExceeded) as e:
                    if attempt == max_retries:
                        print(f"[CRITICAL] CCXT API Operation failed after {max_retries} retries: {e}")
                        raise e
                    
                    # Exponential backoff with jitter
                    sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0.1, 0.5)
                    print(f"[WARN] CCXT API Error ({e.__class__.__name__}): {e}. Retrying in {sleep_time:.2f}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(sleep_time)
                except Exception as e:
                    print(f"[ERROR] Non-retryable API Error: {e}")
                    raise e
        return wrapper
    return decorator

class DataSourceManager:
    def __init__(self, simulation_mode: bool = False, exchange_id: str = "binance", tickers: list = None, timeframe: str = "15m", lookback_days: str = "7d", ticker_ccxt: str = None, yf_ticker: str = None):
        self.simulation_mode = simulation_mode
        self.exchange_id = exchange_id
        self.tickers = tickers or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "LINK/USDT"]
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        
        # Mapping configurations
        self.tickers_ccxt = {
            "BTC/USDT": "BTC/USDT:USDT",
            "ETH/USDT": "ETH/USDT:USDT",
            "SOL/USDT": "SOL/USDT:USDT",
            "BNB/USDT": "BNB/USDT:USDT",
            "LINK/USDT": "LINK/USDT:USDT"
        }
        
        # 12 days of 15m candles is 1152. Capping at 1200 is safe and allows HMA 800.
        self.limit = 1200
        
        # Setup CCXT Exchange Connection (if not in simulation mode)
        self.exchange = None
        if not self.simulation_mode:
            try:
                exchange_class = getattr(ccxt, self.exchange_id)
                self.exchange = exchange_class({
                    'timeout': 5000,
                    'enableRateLimit': True
                })
                print(f"[DATA] CCXT Exchange connection initialized for {self.exchange_id}")
            except Exception as e:
                print(f"[WARN] Failed to initialize CCXT exchange {self.exchange_id}: {e}")
        
        # Setup Cache Directory relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_dir = os.path.join(project_root, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        suffix = "_sim" if self.simulation_mode else ""
        self.cache_files = {
            ticker: os.path.join(self.cache_dir, f"{ticker.replace('/', '_')}_15m_{self.exchange_id}_hist{suffix}.csv")
            for ticker in self.tickers
        }
        
        if self.simulation_mode:
            self._generate_initial_simulation_data()
        else:
            self._cold_boot_cache()

    @retry_ccxt_operation(max_retries=3, base_delay=1.0)
    def _fetch_ccxt_ohlcv_sync(self, symbol: str, limit: int = 700) -> pd.DataFrame:
        """Fetches OHLCV data from the CCXT exchange object synchronously."""
        if not self.exchange:
            raise ValueError("CCXT exchange not initialized")
            
        print(f"[DATA] Fetching last {limit} candles from CCXT {self.exchange_id} ({symbol})...")
        ohlcv = self.exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=self.timeframe,
            limit=limit
        )
        
        if not ohlcv or len(ohlcv) == 0:
            raise ValueError("CCXT returned empty list")
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = df['timestamp'].astype(int)
        
        # Check Binance rate limit weight if available in headers
        try:
            headers = self.exchange.last_response_headers
            if headers:
                weight = next((val for key, val in headers.items() if key.lower() == 'x-mbx-used-weight-1m'), None)
                if weight:
                    print(f"[DATA] Binance API weight used (1m): {weight} / 6000")
        except Exception:
            pass
            
        return df

    def _cold_boot_cache(self):
        """Perform initial one-time download and caching of 7 days of 15m historical candles for all tickers."""
        for ticker in self.tickers:
            cache_file = self.cache_files[ticker]
            if os.path.exists(cache_file):
                try:
                    df = pd.read_csv(cache_file)
                    if not df.empty and len(df) >= self.limit:
                        continue  # Cache file is valid
                except Exception:
                    pass
            
            print(f"[DATA] Cache empty for {ticker}. Performing historical cold-boot download...")
            try:
                ccxt_symbol = self.tickers_ccxt.get(ticker, ticker)
                df = self._fetch_ccxt_ohlcv_sync(ccxt_symbol, limit=self.limit)
                df.to_csv(cache_file, index=False)
                print(f"[DATA] Historical data cached successfully for {ticker}: {len(df)} bars saved.")
            except Exception as e:
                if not self.simulation_mode:
                    print(f"[ERROR] Cold boot download failed for {ticker}: {e}")
                    raise ConnectionError(f"Live exchange connection failed for {ticker}. Check network.")
                print(f"[WARN] Cold boot download failed for {ticker}: {e}. Seed will be simulated.")

    def _generate_initial_simulation_data(self):
        """Generate 1200 historical candles (12 days of 15m data) using a pure Gaussian random walk."""
        self.sim_candles = {ticker: [] for ticker in self.tickers}
        self.sim_tick_indices = {ticker: 1200 for ticker in self.tickers}
        now_ms = int(time.time() * 1000)
        start_time = now_ms - (1200 * 15 * 60 * 1000)
        
        seed_prices = {
            "BTC/USDT": 60000.0,
            "ETH/USDT": 3000.0,
            "SOL/USDT": 135.0,
            "BNB/USDT": 540.0,
            "LINK/USDT": 13.0
        }
        
        for ticker in self.tickers:
            current_price = seed_prices.get(ticker, 100.0)
            
            for i in range(1200):
                t = start_time + (i * 15 * 60 * 1000)
                
                pct_change = random.gauss(0, 0.0025)
                
                open_p = current_price
                close_p = current_price * (1.0 + pct_change)
                
                high_p = max(open_p, close_p) + random.uniform(0.0002, 0.0045) * current_price
                low_p = min(open_p, close_p) - random.uniform(0.0002, 0.0045) * current_price
                
                volume = 100.0 + abs(pct_change) * 200000.0 + random.uniform(50, 300)
                
                self.sim_candles[ticker].append([t, open_p, high_p, low_p, close_p, volume])
                current_price = close_p
 
    def _simulate_tick(self, ticker: str) -> pd.DataFrame:
        """Simulate a new candle close using a pure Gaussian random walk and append it to our list."""
        last_t, _, _, _, last_close, _ = self.sim_candles[ticker][-1]
        next_t = last_t + (15 * 60 * 1000)
        
        self.sim_tick_indices[ticker] += 1
        
        pct_change = random.gauss(0, 0.0025)
        
        open_p = last_close
        close_p = last_close * (1.0 + pct_change)
        
        high_p = max(open_p, close_p) + random.uniform(0.0002, 0.0045) * last_close
        low_p = min(open_p, close_p) - random.uniform(0.0002, 0.0045) * last_close
        
        volume = 100.0 + abs(pct_change) * 200000.0 + random.uniform(50, 300)
        
        self.sim_candles[ticker].append([next_t, open_p, high_p, low_p, close_p, volume])
        self.sim_candles[ticker] = self.sim_candles[ticker][-1200:]
        
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return pd.DataFrame(self.sim_candles[ticker], columns=columns)

    async def fetch_candles(self) -> dict[str, pd.DataFrame]:
        """Returns the updated 7-day dataset for all tickers in parallel."""
        if self.simulation_mode:
            return {ticker: self._simulate_tick(ticker) for ticker in self.tickers}
            
        tasks = [self.sync_ticker_candles(ticker) for ticker in self.tickers]
        results = await asyncio.gather(*tasks)
        return dict(zip(self.tickers, results))

    async def sync_ticker_candles(self, ticker: str) -> pd.DataFrame:
        """Downloads incremental slice and updates disk cache for a single ticker."""
        cache_file = self.cache_files[ticker]
        ccxt_symbol = self.tickers_ccxt.get(ticker, ticker)
        try:
            new_df = await asyncio.to_thread(self._fetch_ccxt_ohlcv_sync, ccxt_symbol, 5)
            if not os.path.exists(cache_file):
                self._cold_boot_cache()
                
            hist_df = pd.read_csv(cache_file)
            combined_df = pd.concat([hist_df, new_df])
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            combined_df = combined_df.tail(self.limit).reset_index(drop=True)
            
            combined_df.to_csv(cache_file, index=False)
            return combined_df
        except Exception as e:
            print(f"[WARN] Incremental sync failed for {ticker}: {e}. Returning cached data on disk...")
            if os.path.exists(cache_file):
                try:
                    df_cache = pd.read_csv(cache_file)
                    if not df_cache.empty:
                        return df_cache
                except Exception:
                    pass
            raise ConnectionError(f"Live data feed disconnected for {ticker} and no cache data is available.")

    @retry_ccxt_operation(max_retries=3, base_delay=1.0)
    def _fetch_order_book_sync(self, ticker: str, limit: int = 5) -> dict:
        """Fetches active bid/ask order book snapshot from CCXT or generates simulated book."""
        ccxt_symbol = self.tickers_ccxt.get(ticker, ticker)
        if self.simulation_mode or not self.exchange:
            cache_file = self.cache_files[ticker]
            last_price = 60000.0
            if os.path.exists(cache_file):
                try:
                    df = pd.read_csv(cache_file)
                    if not df.empty:
                        last_price = float(df['close'].iloc[-1])
                except Exception:
                    pass
                    
            bids = []
            asks = []
            for i in range(1, limit + 1):
                bids.append([last_price - (i * 0.5), random.uniform(0.1, 2.5)])
                asks.append([last_price + (i * 0.5), random.uniform(0.1, 2.5)])
            return {'bids': bids, 'asks': asks}
        
        try:
            return self.exchange.fetch_order_book(ccxt_symbol, limit=limit)
        except Exception as e:
            print(f"[WARN] Failed to fetch order book for {ticker} from CCXT: {e}. Falling back to simulated book.")
            # Fallback to simulated book around last cached price
            cache_file = self.cache_files[ticker]
            last_price = 60000.0
            if os.path.exists(cache_file):
                try:
                    df = pd.read_csv(cache_file)
                    if not df.empty:
                        last_price = float(df['close'].iloc[-1])
                except Exception:
                    pass
            bids = []
            asks = []
            for i in range(1, limit + 1):
                bids.append([last_price - (i * 0.5), random.uniform(0.1, 1.5)])
                asks.append([last_price + (i * 0.5), random.uniform(0.1, 1.5)])
            return {'bids': bids, 'asks': asks}

    async def fetch_order_book(self, ticker: str, limit: int = 5) -> dict:
        """Asynchronously fetches active order book snapshot for a single ticker."""
        return await asyncio.to_thread(self._fetch_order_book_sync, ticker, limit)

    async def fetch_order_books(self, limit: int = 5) -> dict[str, dict]:
        """Asynchronously fetches active order book snapshots for all tickers in parallel."""
        tasks = [self.fetch_order_book(ticker, limit) for ticker in self.tickers]
        results = await asyncio.gather(*tasks)
        return dict(zip(self.tickers, results))

    async def close(self):
        """Clean connection session resources."""
        pass
