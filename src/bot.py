# src/bot.py
import os
import csv
import json
import asyncio
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from config import TalksyConfig
from src.indicators import calculate_indicators, calculate_predicta_scores
from src.ml_filter import MLFilter

class TalksyBot:
    def __init__(self, config: TalksyConfig, simulation_mode: bool = False):
        self.config = config
        self.simulation_mode = simulation_mode
        self.lock = asyncio.Lock()
        self.tickers = config.TICKERS
        
        # Re-engineered variables into dictionaries keyed by ticker symbol
        self.states = {t: "IDLE" for t in self.tickers}
        self.entry_prices = {t: None for t in self.tickers}
        self.exit_sls = {t: None for t in self.tickers}
        self.exit_tps = {t: None for t in self.tickers}
        self.contracts = {t: 0.0 for t in self.tickers}
        self.breakeven_locked = {t: False for t in self.tickers}
        self.initial_risks = {t: 0.0 for t in self.tickers}
        self.low_sentiment_ticks = {t: 0 for t in self.tickers}
        self.candle_close_count_wrong_side = {t: 0 for t in self.tickers}
        self.last_closed_candle_times = {t: None for t in self.tickers}
        self.last_exit_candle_times = {t: None for t in self.tickers}
        self.entry_times = {t: None for t in self.tickers}
        
        # Financial State
        self.balance = config.INITIAL_BALANCE
        self.trade_ledger = []
        
        # Telemetry Cache mappings per symbol
        self.last_dfs = {t: None for t in self.tickers}
        self.last_payloads = {t: None for t in self.tickers}
        self.sentiment_cache = {t: {"bull_pct": 50, "bear_pct": 50} for t in self.tickers}
        self.ml_filter = MLFilter()
        self.ml_veto_logs = []
        self.ml_saved_capital = 0.0
        self.ml_veto_count = 0
        self.ml_approved_count = 0
        
        # Setup Ledger File and Cache Directory
        os.makedirs(os.path.dirname(self.config.LEDGER_FILE), exist_ok=True)
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)
        self._load_ledger()
        self._preload_cache()
        self._load_bot_state()

    # --- Property Getters for single-symbol backwards compatibility ---
    @property
    def state(self):
        return self.states.get(self.config.TICKER, "IDLE")
    @state.setter
    def state(self, val):
        self.states[self.config.TICKER] = val

    @property
    def entry_price(self):
        return self.entry_prices.get(self.config.TICKER)
    @entry_price.setter
    def entry_price(self, val):
        self.entry_prices[self.config.TICKER] = val

    @property
    def exit_sl(self):
        return self.exit_sls.get(self.config.TICKER)
    @exit_sl.setter
    def exit_sl(self, val):
        self.exit_sls[self.config.TICKER] = val

    @property
    def exit_tp(self):
        return self.exit_tps.get(self.config.TICKER)
    @exit_tp.setter
    def exit_tp(self, val):
        self.exit_tps[self.config.TICKER] = val

    @property
    def last_df(self):
        return self.last_dfs.get(self.config.TICKER)
    @last_df.setter
    def last_df(self, val):
        self.last_dfs[self.config.TICKER] = val

    @property
    def last_payload(self):
        return self.last_payloads.get(self.config.TICKER)
    @last_payload.setter
    def last_payload(self, val):
        self.last_payloads[self.config.TICKER] = val

    def _load_ledger(self):
        """Loads transaction ledger history from CSV if it exists."""
        if self.simulation_mode:
            self.balance = self.config.INITIAL_BALANCE
            return  # In simulation mode, start clean without polluting live ledger!
            
        if os.path.exists(self.config.LEDGER_FILE):
            try:
                with open(self.config.LEDGER_FILE, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Extract UNIX timestamps in local timezone for browser compatibility
                        entry_ts = 0
                        exit_ts = 0
                        try:
                            if row.get('entry_time'):
                                dt_ent = datetime.strptime(row['entry_time'], "%Y-%m-%d %H:%M:%S")
                                entry_ts = int(dt_ent.timestamp())
                            if row.get('exit_time'):
                                dt_ex = datetime.strptime(row['exit_time'], "%Y-%m-%d %H:%M:%S")
                                exit_ts = int(dt_ex.timestamp())
                        except Exception:
                            pass

                        self.trade_ledger.append({
                            'entry_time': row.get('entry_time', ''),
                            'exit_time': row.get('exit_time', ''),
                            'entry_timestamp': entry_ts,
                            'exit_timestamp': exit_ts,
                            'ticker': row.get('ticker', ''),
                            'type': row.get('type', ''),
                            'entry_price': float(row.get('entry_price', 0.0)),
                            'exit_price': float(row.get('exit_price', 0.0)),
                            'pnl': float(row.get('pnl', 0.0)),
                            'cause': row.get('cause', ''),
                            'position_size': float(row.get('position_size', 0.0)),
                            'balance': float(row.get('balance', 0.0))
                        })
                # Re-establish running balance based on last trade
                if self.trade_ledger:
                    self.balance = self.trade_ledger[-1]['balance']
            except Exception as e:
                print(f"[WARN] Failed to load trade ledger: {e}")

    def _preload_cache(self):
        """Loads cached historical data on startup for all tickers if present to instantly serve the dashboard."""
        for ticker in self.tickers:
            suffix = "_sim" if self.simulation_mode else ""
            cache_file = os.path.join(self.config.CACHE_DIR, f"{ticker.replace('/', '_')}_15m_{self.config.EXCHANGE_ID}_hist{suffix}.csv")
            if os.path.exists(cache_file):
                try:
                    df = pd.read_csv(cache_file)
                    if not df.empty and len(df) >= 20:
                        df = calculate_indicators(df, trend_type=self.config.TREND_TYPE, trend_len=self.config.TREND_LEN)
                        df = calculate_predicta_scores(df)
                        self.last_dfs[ticker] = df
                        
                        last_row = df.iloc[-1]
                        bull_pct = int(last_row['final_bull_percentage']) if 'final_bull_percentage' in last_row else 50
                        bear_pct = int(last_row['final_bear_percentage']) if 'final_bear_percentage' in last_row else 50
                        self.sentiment_cache[ticker] = {"bull_pct": bull_pct, "bear_pct": bear_pct}
                        
                        # Reconstruct latest closed candle time
                        self.last_closed_candle_times[ticker] = int(last_row['timestamp'])
                        
                        indicators = {
                            'ha_close': float(last_row['ha_close']),
                            'trend_val': float(last_row['trend_baseline']) if 'trend_baseline' in last_row and not pd.isna(last_row['trend_baseline']) else None,
                            'macro_val': float(last_row['macro_baseline']) if 'macro_baseline' in last_row and not pd.isna(last_row['macro_baseline']) else None,
                            'ema9': float(last_row['ema9']),
                            'ema21': float(last_row['ema21']),
                            'rsi': float(last_row['rsi14']) if 'rsi14' in last_row and not pd.isna(last_row['rsi14']) else None,
                            'stoch_k': float(last_row['stoch_k']) if 'stoch_k' in last_row and not pd.isna(last_row['stoch_k']) else None,
                            'stoch_d': float(last_row['stoch_d']) if 'stoch_d' in last_row and not pd.isna(last_row['stoch_d']) else None,
                            'adx': float(last_row['adx14']) if 'adx14' in last_row and not pd.isna(last_row['adx14']) else None,
                            'di_diff': float(last_row['di_diff']) if 'di_diff' in last_row and not pd.isna(last_row['di_diff']) else 0.0,
                            'atr': float(last_row['atr14']) if 'atr14' in last_row and not pd.isna(last_row['atr14']) else None,
                            'volume': float(last_row['volume']),
                            'vol_sma20': float(last_row['vol_sma20']) if 'vol_sma20' in last_row and not pd.isna(last_row['vol_sma20']) else None,
                            'vol_delta': float(last_row['vol_delta']) if 'vol_delta' in last_row and not pd.isna(last_row['vol_delta']) else 0.0,
                            'trend_direction': str(last_row['trend_direction']) if 'trend_direction' in last_row else 'UP'
                        }
                        
                        self.last_payloads[ticker] = {
                            'ticker': ticker,
                            'timeframe': self.config.TIMEFRAME_PRIMARY,
                            'state': self.states[ticker],
                            'entry_price': self.entry_prices[ticker],
                            'exit_sl': self.exit_sls[ticker],
                            'exit_tp': self.exit_tps[ticker],
                            'current_price': float(last_row['close']),
                            'candle_close_count_wrong_side': self.candle_close_count_wrong_side[ticker],
                            'bull_pct': bull_pct,
                            'bear_pct': bear_pct,
                            'balance': self.balance,
                            'portfolio_val': self.balance,
                            'contracts': self.contracts[ticker],
                            'indicators': indicators,
                            'trade_ledger': self.trade_ledger,
                            'order_book': None
                        }
                except Exception as e:
                    print(f"[WARN] Failed to pre-load cached CSV telemetry for {ticker}: {e}")

    def _save_bot_state(self):
        """Serialize and write active position state variables to a JSON file atomically."""
        if self.simulation_mode:
            return
            
        state_file = os.path.join(self.config.CACHE_DIR, "active_position.json")
        temp_file = state_file + ".tmp"
        try:
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            # Pack nested structure of state details for all symbols
            state_data = {
                'balance': self.balance,
                'tickers_data': {}
            }
            
            for t in self.tickers:
                state_data['tickers_data'][t] = {
                    'state': self.states[t],
                    'entry_price': float(self.entry_prices[t]) if self.entry_prices[t] is not None else None,
                    'entry_time': self.entry_times[t],
                    'exit_sl': float(self.exit_sls[t]) if self.exit_sls[t] is not None else None,
                    'exit_tp': float(self.exit_tps[t]) if self.exit_tps[t] is not None else None,
                    'contracts': float(self.contracts[t]) if self.contracts[t] is not None else 0.0,
                    'breakeven_locked': bool(self.breakeven_locked[t]),
                    'initial_risk': float(self.initial_risks[t]) if self.initial_risks[t] is not None else 0.0,
                    'low_sentiment_ticks': int(self.low_sentiment_ticks[t]),
                    'candle_close_count_wrong_side': int(self.candle_close_count_wrong_side[t]),
                    'last_closed_candle_time': int(self.last_closed_candle_times[t]) if self.last_closed_candle_times[t] is not None else None,
                    'last_exit_candle_time': int(self.last_exit_candle_times[t]) if self.last_exit_candle_times[t] is not None else None
                }
                
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, state_file)
        except Exception as e:
            print(f"[ERROR] Failed to save bot state JSON: {e}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    def _load_bot_state(self):
        """Deserialize and restore active position state variables from JSON file on startup."""
        if self.simulation_mode:
            self.balance = self.config.INITIAL_BALANCE
            return
            
        state_file = os.path.join(self.config.CACHE_DIR, "active_position.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                self.balance = state_data.get('balance', self.config.INITIAL_BALANCE)
                tickers_data = state_data.get('tickers_data', {})
                
                for t in self.tickers:
                    if t in tickers_data:
                        t_state = tickers_data[t]
                        self.states[t] = t_state.get('state', 'IDLE')
                        self.entry_prices[t] = t_state.get('entry_price')
                        self.entry_times[t] = t_state.get('entry_time')
                        self.exit_sls[t] = t_state.get('exit_sl')
                        self.exit_tps[t] = t_state.get('exit_tp')
                        self.contracts[t] = t_state.get('contracts', 0.0)
                        self.breakeven_locked[t] = t_state.get('breakeven_locked', False)
                        self.initial_risks[t] = t_state.get('initial_risk', 0.0)
                        self.low_sentiment_ticks[t] = t_state.get('low_sentiment_ticks', 0)
                        self.candle_close_count_wrong_side[t] = t_state.get('candle_close_count_wrong_side', 0)
                        self.last_closed_candle_times[t] = t_state.get('last_closed_candle_time')
                        self.last_exit_candle_times[t] = t_state.get('last_exit_candle_time')
                        
                        if self.states[t] in ["LONG", "SHORT"]:
                            print(f"[SYSTEM] Persistent recovery: Restored active {self.states[t]} position for {t} at entry ${self.entry_prices[t]:.2f}.")
            except Exception as e:
                print(f"[WARN] Failed to load persistent bot state JSON: {e}")

    def calculate_volatility_position_size(self, ticker: str, entry_price: float, stop_loss_price: float) -> float:
        """
        Calculates dynamic contract sizing anchored to structural risk distance.
        Ensures maximum dollar risk remains constant regardless of volatility.
        """
        try:
            # 1. Determine absolute risk distance per contract unit
            risk_per_unit = abs(entry_price - stop_loss_price)
            limits = self.config.POSITION_LIMITS[ticker]
            
            if risk_per_unit == 0:
                print(f"[!] Error: Risk distance is zero for {ticker}. Defaulting to minimum size.")
                return limits["min"]
                
            # 2. Math Adjuster: static split dollar risk per sandbox
            sandbox_capital = self.config.TOTAL_CAPITAL / len(self.tickers)
            target_dollar_risk = sandbox_capital * (self.config.MAX_RISK_PCT / 100.0)
            
            # 3. Derive mathematical size needed to risk exactly target_dollar_risk
            raw_size = target_dollar_risk / risk_per_unit
            
            # 4. Standardize position sizing to coin-specific rounding precision
            rounded_step = round(raw_size, limits["round_digits"])
            if rounded_step < limits["min"]:
                print(f"[*] Sizing warning: Rounded step {rounded_step} is below floor {limits['min']}. Clamping up to floor.")
                rounded_step = limits["min"]
            
            # 5. Clamp size within strict boundary limits
            final_position = max(limits["min"], min(limits["max"], rounded_step))
            
            print(f"[*] Sizing Metrics [{ticker}] -> Risk Distance: ${risk_per_unit:.2f} | Raw Target: {raw_size:.4f} units")
            print(f"[+] Final Allocated Size: {final_position} (Max Risk: ${target_dollar_risk:.2f})")
            return final_position
        except Exception as e:
            print(f"[!] Critical Sizing Calculation Failure for {ticker}: {e}")
            return self.config.POSITION_LIMITS[ticker]["min"]

    def _save_ledger(self, trade: dict):
        """Append a completed transaction to CSV ledger."""
        if self.simulation_mode:
            return  # Do not write simulation trades to the real ledger file!
            
        fieldnames = [
            'entry_time', 'exit_time', 'ticker', 'type', 'entry_price',
            'exit_price', 'pnl', 'cause', 'position_size', 'balance'
        ]
        file_exists = os.path.exists(self.config.LEDGER_FILE)
        
        try:
            with open(self.config.LEDGER_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(trade)
        except Exception as e:
            print(f"[ERROR] Failed to save transaction to CSV ledger: {e}")

    def is_within_allowed_session(self, timestamp_ms: float = None) -> bool:
        """Checks if UTC time falls inside the allowed trading windows."""
        if not self.config.USE_SESSION_FILTER:
            return True
            
        if timestamp_ms is not None:
            now_utc = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        else:
            now_utc = datetime.now(timezone.utc)
            
        current_time_str = now_utc.strftime("%H:%M")
        
        for start_str, end_str in self.config.ALLOWED_SESSION_WINDOWS:
            if start_str <= end_str:
                if start_str <= current_time_str <= end_str:
                    return True
            else:
                # Crosses midnight boundary
                if current_time_str >= start_str or current_time_str <= end_str:
                    return True
        return False

    def get_session_status_and_countdown(self, timestamp_ms: float = None) -> tuple[str, str]:
        """
        Returns a tuple of (status, countdown_str) representing the allowed trading windows.
        If active, returns ("ACTIVE", "").
        If inactive, returns ("INACTIVE", "Starts in HHh MMm SSs").
        """
        if not self.config.USE_SESSION_FILTER:
            return "ACTIVE", ""
            
        if timestamp_ms is not None:
            now_utc = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        else:
            now_utc = datetime.now(timezone.utc)
            
        current_time_str = now_utc.strftime("%H:%M")
        
        # 1. Check if currently active
        is_active = False
        for start_str, end_str in self.config.ALLOWED_SESSION_WINDOWS:
            if start_str <= end_str:
                if start_str <= current_time_str <= end_str:
                    is_active = True
                    break
            else:
                if current_time_str >= start_str or current_time_str <= end_str:
                    is_active = True
                    break
                    
        if is_active:
            return "ACTIVE", ""
            
        # 2. Calculate minutes until next session
        now_minutes = now_utc.hour * 60 + now_utc.minute
        
        min_wait = 999999
        for start_str, _ in self.config.ALLOWED_SESSION_WINDOWS:
            sh, sm = map(int, start_str.split(":"))
            start_minutes = sh * 60 + sm
            
            if start_minutes > now_minutes:
                wait = start_minutes - now_minutes
            else:
                wait = (1440 - now_minutes) + start_minutes
                
            if wait < min_wait:
                min_wait = wait
                
        h = min_wait // 60
        m = min_wait % 60
        seconds_left = 60 - now_utc.second if now_utc.second > 0 else 0
        if seconds_left > 0:
            if m > 0:
                m -= 1
            elif h > 0:
                h -= 1
                m = 59
                
        if h > 0:
            countdown_str = f"Starts in {h:02d}h {m:02d}m {seconds_left:02d}s"
        else:
            countdown_str = f"Starts in {m:02d}m {seconds_left:02d}s"
            
        return "INACTIVE", countdown_str

    def get_portfolio_value(self, ticker: str, current_price: float) -> float:
        """Returns the hypothetical cash + open trade valuation for a specific coin."""
        state = self.states[ticker]
        if state == "LONG" and self.entry_prices[ticker] is not None:
            return self.balance + (self.contracts[ticker] * (current_price - self.entry_prices[ticker]))
        elif state == "SHORT" and self.entry_prices[ticker] is not None:
            return self.balance + (self.contracts[ticker] * (self.entry_prices[ticker] - current_price))
        return self.balance

    def get_total_portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Returns unified capital valuation across the entire multi-position portfolio."""
        total = self.balance
        for t in self.tickers:
            state = self.states[t]
            if state == "LONG" and self.entry_prices[t] is not None and t in current_prices:
                total += self.contracts[t] * (current_prices[t] - self.entry_prices[t])
            elif state == "SHORT" and self.entry_prices[t] is not None and t in current_prices:
                total += self.contracts[t] * (self.entry_prices[t] - current_prices[t])
        return total

    def reconstruct_trade_audit_prices(self, trade: dict, df: pd.DataFrame) -> dict:
        """Reconstruct initial SL (SL-1), initial TP (TP-1), and trailed SL (SL-2) for auditing."""
        try:
            entry_dt = datetime.strptime(trade['entry_time'], "%Y-%m-%d %H:%M:%S")
            entry_ts_ms = int(entry_dt.timestamp() * 1000)
            
            # Find closest matching timestamp in df
            if df.empty:
                return {}
            diffs = (df['timestamp'] - entry_ts_ms).abs()
            entry_idx = diffs.idxmin()
            
            row = df.loc[entry_idx]
            atr = float(row['atr']) if 'atr' in row and not pd.isna(row['atr']) else 0.0
            
            # Locate entry candle position in integer indices to lookback last 3 candles
            int_idx = df.index.get_loc(entry_idx)
            start_idx = max(0, int_idx - 2)
            recent_3 = df.iloc[start_idx : int_idx + 1]
            
            hvn_idx = recent_3['volume'].idxmax()
            hvn_price = float(recent_3.loc[hvn_idx, 'close'])
            
            entry_price = float(trade['entry_price'])
            exit_price = float(trade['exit_price'])
            
            # Initial Stop Loss
            if trade['type'] == 'LONG':
                sl_initial = hvn_price - (self.config.SL_ATR_CUSHION * atr)
                sl_initial = min(sl_initial, entry_price - ((self.config.SL_ATR_CUSHION + 0.5) * atr))
            else:
                sl_initial = hvn_price + (self.config.SL_ATR_CUSHION * atr)
                sl_initial = max(sl_initial, entry_price + ((self.config.SL_ATR_CUSHION + 0.5) * atr))
            sl_initial = max(sl_initial, 0.01)
            
            # Initial Take Profit
            initial_risk = abs(entry_price - sl_initial)
            if trade['type'] == 'LONG':
                tp_initial = entry_price + (self.config.RISK_REWARD_RATIO * initial_risk)
            else:
                tp_initial = entry_price - (self.config.RISK_REWARD_RATIO * initial_risk)
                
            # Trailed Stop Loss (if hit SL cause but ended better than entry)
            sl_trailed = None
            cause_label = trade.get('cause', '')
            if cause_label and 'SL' in cause_label:
                is_long_trailed = (trade['type'] == 'LONG' and exit_price > entry_price)
                is_short_trailed = (trade['type'] == 'SHORT' and exit_price < entry_price)
                if is_long_trailed or is_short_trailed:
                    sl_trailed = exit_price
                    
            return {
                'sl_initial': sl_initial,
                'tp_initial': tp_initial,
                'sl_trailed': sl_trailed
            }
        except Exception as e:
            print(f"[WARN] Failed to reconstruct audit prices: {e}")
            return {}

    def get_historical_telemetry(self, ticker: str = None, timeframe: str = "15m", audit_index: int = None) -> dict:
        """Returns the full historical series data for a specific symbol's chart initialization."""
        if ticker is None:
            ticker = self.config.TICKER
            
        df = self.last_dfs.get(ticker)
        if df is None or df.empty:
            return {}
            
        if timeframe == "1h":
            # Resample standard candles to 1-hour resolution
            temp_df = df.copy()
            temp_df['dt'] = pd.to_datetime(temp_df['timestamp'], unit='ms')
            temp_df = temp_df.set_index('dt')
            
            resampled = temp_df.resample('1h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'timestamp': 'first'
            }).dropna()
            
            # Floor timestamp to start of hour in UTC milliseconds
            resampled['timestamp'] = (resampled['timestamp'].astype(int) // 3600000) * 3600000
            
            # Recalculate Heikin Ashi values for 1-hour resolution
            ha_df = resampled.copy()
            ha_close = (resampled['open'] + resampled['high'] + resampled['low'] + resampled['close']) / 4.0
            ha_open = np.zeros(len(resampled))
            ha_open[0] = (resampled['open'].iloc[0] + resampled['close'].iloc[0]) / 2.0
            for i in range(1, len(resampled)):
                ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2.0
            ha_df['ha_open'] = ha_open
            ha_df['ha_close'] = ha_close
            ha_df['ha_high'] = np.maximum(resampled['high'], np.maximum(ha_open, ha_close))
            ha_df['ha_low'] = np.minimum(resampled['low'], np.minimum(ha_open, ha_close))
            
            # Calculate HMA 200 on 1-hour Heikin Ashi close
            from src.indicators import calculate_hma
            ha_df['trend_baseline'] = calculate_hma(ha_df['ha_close'], self.config.TREND_LEN)
            ha_df['ema9'] = ha_df['ha_close'].ewm(span=9, adjust=False).mean()
            ha_df['ema21'] = ha_df['ha_close'].ewm(span=21, adjust=False).mean()
            ha_df['vol_sma20'] = ha_df['volume'].rolling(20).mean()
            
            df = ha_df
            
        # Convert milliseconds timestamps to seconds for TradingView Lightweight Charts
        timestamps_sec = (df['timestamp'] / 1000.0).astype(int).tolist()
        
        candles = []
        ha_candles = []
        volume = []
        hma200 = []
        hma800 = []
        vol_sma = []
        ema9 = []
        ema21 = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            t = timestamps_sec[i]
            
            candles.append({
                'time': t,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })
            
            ha_candles.append({
                'time': t,
                'open': float(row['ha_open']),
                'high': float(row['ha_high']),
                'low': float(row['ha_low']),
                'close': float(row['ha_close'])
            })
            
            volume.append({
                'time': t,
                'value': float(row['volume'] * row['close']),
                'color': '#34d399' if row['close'] > row['open'] else '#f87171'
            })
            
            # Local trend baseline (200 HMA on 15m or 1h)
            if 'trend_baseline' in row and not pd.isna(row['trend_baseline']):
                hma200.append({
                    'time': t,
                    'value': float(row['trend_baseline'])
                })
                
            # Macro trend baseline (800 HMA - only available in 15m)
            if timeframe == "15m" and 'macro_baseline' in row and not pd.isna(row['macro_baseline']):
                hma800.append({
                    'time': t,
                    'value': float(row['macro_baseline'])
                })
                
            if 'ema9' in row and not pd.isna(row['ema9']):
                ema9.append({
                    'time': t,
                    'value': float(row['ema9'])
                })
                
            if 'ema21' in row and not pd.isna(row['ema21']):
                ema21.append({
                    'time': t,
                    'value': float(row['ema21'])
                })
                
            if not pd.isna(row['vol_sma20']):
                vol_sma.append({
                    'time': t,
                    'value': float(row['vol_sma20'] * row['close'])
                })
                
        # Generate BUY/SELL charting marker tags for this specific ticker
        markers = []
        for trade in self.trade_ledger:
            if trade.get('ticker') != ticker:
                continue
            try:
                entry_dt = datetime.strptime(trade['entry_time'], "%Y-%m-%d %H:%M:%S")
                entry_t = int(entry_dt.timestamp())
                exit_dt = datetime.strptime(trade['exit_time'], "%Y-%m-%d %H:%M:%S")
                exit_t = int(exit_dt.timestamp())
                
                trade_type = trade.get('type')
                pnl = trade.get('pnl', 0.0)
                
                # Entry Marker
                markers.append({
                    'time': entry_t,
                    'position': 'belowBar' if trade_type == 'LONG' else 'aboveBar',
                    'color': '#8b80f9',
                    'shape': 'arrowUp' if trade_type == 'LONG' else 'arrowDown',
                    'text': f"BUY {trade.get('position_size')}" if trade_type == 'LONG' else f"SELL {trade.get('position_size')}"
                })
                
                # Exit Marker
                markers.append({
                    'time': exit_t,
                    'position': 'aboveBar' if trade_type == 'LONG' else 'belowBar',
                    'color': '#34d399' if pnl >= 0 else '#f87171',
                    'shape': 'arrowDown' if trade_type == 'LONG' else 'arrowUp',
                    'text': f"EXIT ${pnl:+.2f}"
                })
            except Exception as e:
                print(f"[WARN] Failed to parse ledger markers: {e}")
                
        # Reconstruct audit prices if requested
        audit_details = {}
        if audit_index is not None:
            try:
                idx = int(audit_index) - 1
                if 0 <= idx < len(self.trade_ledger):
                    trade = self.trade_ledger[idx]
                    audit_details = self.reconstruct_trade_audit_prices(trade, df)
            except Exception as e:
                print(f"[WARN] Telemetry audit parse failed: {e}")
                
        return {
            'candles': candles,
            'ha_candles': ha_candles,
            'volume': volume,
            'hma200': hma200,
            'hma800': hma800,
            'vol_sma': vol_sma,
            'markers': markers,
            'ema9': ema9,
            'ema21': ema21,
            'audit_details': audit_details
        }

    async def process_tick(self, dfs: dict[str, pd.DataFrame], order_books: dict[str, dict]) -> dict[str, dict]:
        """
        Processes a single ticking interval for all tickers.
        Updates indicators, runs State Machine entry/exit checks, and returns status package.
        """
        async with self.lock:
            payloads = {}
            current_prices = {}
            
            # First extract live prices to compute combined portfolio value correctly
            for ticker in self.tickers:
                raw_df = dfs.get(ticker)
                if raw_df is not None and not raw_df.empty:
                    current_prices[ticker] = float(raw_df['close'].iloc[-1])
            
            total_portfolio_val = self.get_total_portfolio_value(current_prices)
            
            for ticker in self.tickers:
                try:
                    raw_df = dfs.get(ticker)
                    order_book = order_books.get(ticker)
                    if raw_df is None or raw_df.empty:
                        continue
                        
                    # 1. Calculate Indicators & Scores
                    df_with_ind = calculate_indicators(
                        raw_df, 
                        trend_len=self.config.TREND_LEN, 
                        trend_type=self.config.TREND_TYPE
                    )
                    df_with_scores = calculate_predicta_scores(df_with_ind)
                    self.last_dfs[ticker] = df_with_scores
                    
                    active_row = df_with_scores.iloc[-1]
                    closed_row = df_with_scores.iloc[-2] if len(df_with_scores) >= 2 else active_row
                    
                    current_price = active_row['close']
                    current_time = datetime.fromtimestamp(active_row['timestamp'] / 1000.0)
                    
                    # Extract indicators for UI
                    indicators = {
                        'ha_close': float(active_row['ha_close']),
                        'rsi': float(active_row['rsi']) if not pd.isna(active_row['rsi']) else 0.0,
                        'trend_val': float(active_row['trend_baseline']) if not pd.isna(active_row['trend_baseline']) else 0.0,
                        'macro_val': float(active_row['macro_baseline']) if 'macro_baseline' in active_row and not pd.isna(active_row['macro_baseline']) else 0.0,
                        'trend_type': self.config.TREND_TYPE,
                        'stoch_k': float(active_row['stoch_k']) if not pd.isna(active_row['stoch_k']) else 0.0,
                        'stoch_d': float(active_row['stoch_d']) if not pd.isna(active_row['stoch_d']) else 0.0,
                        'ema9': float(active_row['ema9']) if not pd.isna(active_row['ema9']) else 0.0,
                        'ema21': float(active_row['ema21']) if not pd.isna(active_row['ema21']) else 0.0,
                        'atr': float(active_row['atr']) if not pd.isna(active_row['atr']) else 0.0,
                        'adx': float(active_row['adx']) if not pd.isna(active_row['adx']) else 0.0,
                        'di_diff': float(active_row['plus_di'] - active_row['minus_di']) if not pd.isna(active_row['plus_di']) else 0.0,
                        'vol_delta': float(active_row['volume_delta']) if not pd.isna(active_row['volume_delta']) else 0.0,
                        'vol_sma20': float(active_row['vol_sma20'] * active_row['close']) if not pd.isna(active_row['vol_sma20']) else 0.0,
                        'delta_mom': bool(active_row['delta_momentum']) if not pd.isna(active_row['delta_momentum']) else False
                    }
                    
                    bull_pct = int(active_row['final_bull_percentage']) if not pd.isna(active_row['final_bull_percentage']) else 50
                    bear_pct = int(active_row['final_bear_percentage']) if not pd.isna(active_row['final_bear_percentage']) else 50
                    self.sentiment_cache[ticker] = {"bull_pct": bull_pct, "bear_pct": bear_pct}
                    
                    ticker_state = self.states[ticker]
                    
                    # 2. Check Exits (if in position)
                    if ticker_state in ["LONG", "SHORT"]:
                        current_atr = active_row['atr']
                        exit_triggered = False
                        exit_price = current_price
                        cause = ""
                        
                        # --- Pillar 3: Asymmetric Breakeven ---
                        if not self.breakeven_locked[ticker]:
                            if ticker_state == "LONG":
                                target_price = self.entry_prices[ticker] + self.initial_risks[ticker] + current_atr
                                if current_price >= target_price:
                                    self.exit_sls[ticker] = self.entry_prices[ticker] + (0.15 * current_atr)
                                    self.breakeven_locked[ticker] = True
                                    print(f"[TRADE] {ticker} LONG Breakeven locked with profit buffer at ${self.exit_sls[ticker]:.2f} (Price: ${current_price:.2f})")
                            elif ticker_state == "SHORT":
                                target_price = self.entry_prices[ticker] - self.initial_risks[ticker] - current_atr
                                if current_price <= target_price:
                                    self.exit_sls[ticker] = self.entry_prices[ticker] - (0.15 * current_atr)
                                    self.breakeven_locked[ticker] = True
                                    print(f"[TRADE] {ticker} SHORT Breakeven locked with profit buffer at ${self.exit_sls[ticker]:.2f} (Price: ${current_price:.2f})")
                        
                        # --- Pillar 2: Elastic Sentiment Leash (Predicta-Driven Trailing) ---
                        if ticker_state == "LONG":
                            # Trailing Activation Floor: Only trail if profit is at least 1.0 ATR
                            if (current_price - self.entry_prices[ticker]) >= (1.0 * current_atr):
                                # High Confidence Regime
                                if bull_pct >= 75:
                                    trail_sl = current_price - (2.5 * current_atr)
                                    if trail_sl > self.exit_sls[ticker]:
                                        self.exit_sls[ticker] = trail_sl
                            
                            # Decay Regime (needs 2 consecutive ticks) - ACTIVE IMMEDIATELY FROM ENTRY
                            if bull_pct < 55:
                                self.low_sentiment_ticks[ticker] += 1
                                if self.low_sentiment_ticks[ticker] >= 2:
                                    tight_sl = closed_row['low']
                                    if tight_sl > self.exit_sls[ticker]:
                                        self.exit_sls[ticker] = tight_sl
                            else:
                                self.low_sentiment_ticks[ticker] = 0
                        elif ticker_state == "SHORT":
                            # Trailing Activation Floor: Only trail if profit is at least 1.0 ATR
                            if (self.entry_prices[ticker] - current_price) >= (1.0 * current_atr):
                                # High Confidence Regime
                                if bear_pct >= 75:
                                    trail_sl = current_price + (2.5 * current_atr)
                                    if trail_sl < self.exit_sls[ticker]:
                                        self.exit_sls[ticker] = trail_sl
                            
                            # Decay Regime (needs 2 consecutive ticks) - ACTIVE IMMEDIATELY FROM ENTRY
                            if bear_pct < 55:
                                self.low_sentiment_ticks[ticker] += 1
                                if self.low_sentiment_ticks[ticker] >= 2:
                                    tight_sl = closed_row['high']
                                    if tight_sl < self.exit_sls[ticker]:
                                        self.exit_sls[ticker] = tight_sl
                            else:
                                self.low_sentiment_ticks[ticker] = 0
        
                        # --- Pillar 4: Weighted Reversal Matrix (Quant Trend-Flip) ---
                        threat_score = 0
                        
                        # 1. Institutional Counter-Volume
                        if ticker_state == "LONG":
                            if (active_row['close'] < active_row['open']) and (active_row['volume'] > active_row['vol_sma20']):
                                threat_score += 5
                        elif ticker_state == "SHORT":
                            if (active_row['close'] > active_row['open']) and (active_row['volume'] > active_row['vol_sma20']):
                                threat_score += 5
                                
                        # 2. Predicta Sentiment Inversion
                        if ticker_state == "LONG" and bear_pct >= 65:
                            threat_score += 3
                        elif ticker_state == "SHORT" and bull_pct >= 65:
                            threat_score += 3
                            
                        # 3. Structural Momentum Cross
                        if ticker_state == "LONG" and active_row['ema9'] < active_row['ema21']:
                            threat_score += 2
                        elif ticker_state == "SHORT" and active_row['ema9'] > active_row['ema21']:
                            threat_score += 2
                            
                        # Check exit triggers
                        if threat_score >= 7:
                            exit_triggered = True
                            cause = f"Weighted Reversal Matrix (Threat: {threat_score}/10)"
                            print(f"[TRADE] Emergency Reversal Matrix triggered for {ticker}! Threat Score: {threat_score}/10. Liquidating...")
        
                        # Target checks
                        high_val = active_row['high']
                        low_val = active_row['low']
                        
                        if not exit_triggered:
                            if ticker_state == "LONG":
                                if low_val <= self.exit_sls[ticker]:
                                    exit_triggered = True
                                    exit_price = self.exit_sls[ticker]
                                    cause = "Stop Loss (SL)"
                                elif high_val >= self.exit_tps[ticker]:
                                    exit_triggered = True
                                    exit_price = self.exit_tps[ticker]
                                    cause = "Take Profit (TP)"
                            elif ticker_state == "SHORT":
                                if high_val >= self.exit_sls[ticker]:
                                    exit_triggered = True
                                    exit_price = self.exit_sls[ticker]
                                    cause = "Stop Loss (SL)"
                                elif low_val <= self.exit_tps[ticker]:
                                    exit_triggered = True
                                    exit_price = self.exit_tps[ticker]
                                    cause = "Take Profit (TP)"
                                
                        # Layer 2: Math Panic
                        if not exit_triggered:
                            if ticker_state == "LONG" and bear_pct >= self.config.PANIC_THRESHOLD:
                                exit_triggered = True
                                cause = "Math Panic"
                            elif ticker_state == "SHORT" and bull_pct >= self.config.PANIC_THRESHOLD:
                                exit_triggered = True
                                cause = "Math Panic"
                                
                        # Layer 3: Trend Panic (Patient 3-Candle Rule at 15m candle close)
                        closed_candle_ts = closed_row['timestamp']
                        if self.last_closed_candle_times[ticker] is None:
                            self.last_closed_candle_times[ticker] = closed_candle_ts
                            
                        new_candle_closed = closed_candle_ts > self.last_closed_candle_times[ticker]
                        
                        if not exit_triggered and new_candle_closed:
                            self.last_closed_candle_times[ticker] = closed_candle_ts
                            
                            if ticker_state == "LONG":
                                if closed_row['ha_close'] < closed_row['ema21']:
                                    self.candle_close_count_wrong_side[ticker] += 1
                                    print(f"[PANIC] {ticker} Heikin Ashi Close (${closed_row['ha_close']:.2f}) closed below EMA21. Wrong Side Count: {self.candle_close_count_wrong_side[ticker]}/3")
                                else:
                                    if self.candle_close_count_wrong_side[ticker] > 0:
                                        print(f"[PANIC] {ticker} Heikin Ashi Close returned above EMA21. Wrong Side Count reset to 0.")
                                    self.candle_close_count_wrong_side[ticker] = 0
                                    
                                if self.candle_close_count_wrong_side[ticker] >= 3:
                                    exit_triggered = True
                                    cause = "Trend 3-Bar Panic"
                                    
                            elif ticker_state == "SHORT":
                                if closed_row['ha_close'] > closed_row['ema21']:
                                    self.candle_close_count_wrong_side[ticker] += 1
                                    print(f"[PANIC] {ticker} Heikin Ashi Close (${closed_row['ha_close']:.2f}) closed above EMA21. Wrong Side Count: {self.candle_close_count_wrong_side[ticker]}/3")
                                else:
                                    if self.candle_close_count_wrong_side[ticker] > 0:
                                        print(f"[PANIC] {ticker} Heikin Ashi Close returned below EMA21. Wrong Side Count reset to 0.")
                                    self.candle_close_count_wrong_side[ticker] = 0
                                    
                                if self.candle_close_count_wrong_side[ticker] >= 3:
                                    exit_triggered = True
                                    cause = "Trend 3-Bar Panic"
                        
                        # Execute Exit if Triggered
                        if exit_triggered:
                            pnl_dollar = 0.0
                            if ticker_state == "LONG":
                                pnl_dollar = self.contracts[ticker] * (exit_price - self.entry_prices[ticker])
                            elif ticker_state == "SHORT":
                                pnl_dollar = self.contracts[ticker] * (self.entry_prices[ticker] - exit_price)
                                
                            # Deduct Binance VIP 0 Maker Fee (0.02% of entry value + 0.02% of exit value)
                            entry_val = self.contracts[ticker] * self.entry_prices[ticker]
                            exit_val = self.contracts[ticker] * exit_price
                            maker_fee = (entry_val + exit_val) * 0.0002
                            pnl_dollar -= maker_fee
                            
                            self.balance += pnl_dollar
                            exit_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                            
                            entry_ts_sec = 0
                            try:
                                dt_ent = datetime.strptime(self.entry_times[ticker], "%Y-%m-%d %H:%M:%S")
                                entry_ts_sec = int(dt_ent.timestamp())
                            except Exception:
                                pass
                            
                            trade_record = {
                                'entry_time': self.entry_times[ticker],
                                'exit_time': exit_time_str,
                                'entry_timestamp': entry_ts_sec,
                                'exit_timestamp': int(current_time.timestamp()),
                                'ticker': ticker,
                                'type': ticker_state,
                                'entry_price': self.entry_prices[ticker],
                                'exit_price': exit_price,
                                'pnl': pnl_dollar,
                                'cause': cause,
                                'position_size': self.contracts[ticker],
                                'balance': self.balance
                            }
                            
                            self.trade_ledger.append(trade_record)
                            self._save_ledger(trade_record)
                            
                            # Reset Position State
                            self.states[ticker] = "IDLE"
                            self.entry_prices[ticker] = None
                            self.exit_sls[ticker] = None
                            self.exit_tps[ticker] = None
                            self.contracts[ticker] = 0.0
                            self.candle_close_count_wrong_side[ticker] = 0
                            self.last_exit_candle_times[ticker] = active_row['timestamp']
                            self._save_bot_state()
        
                        if self.states[ticker] != "IDLE":
                            self._save_bot_state()
        
                    # 3. Check Entries (if IDLE)
                    elif self.states[ticker] == "IDLE" and active_row['timestamp'] != self.last_exit_candle_times[ticker]:
                        if not pd.isna(active_row['adx']):
                            if active_row['adx'] > 15.0:
                                current_atr = active_row['atr']
                                
                                min_gate = self.config.SENTIMENT_GATES.get(ticker, 65.0)
                                
                                # Long Entry Criteria
                                long_trigger = (
                                    (not pd.isna(active_row['adx'])) and (active_row['adx'] > 15.0) and
                                    (active_row['ema9'] > active_row['ema21']) and
                                    (active_row['ha_low'] <= active_row['ema9']) and
                                    (active_row['ha_close'] > active_row['ema9']) and
                                    (active_row['ha_close'] > active_row['trend_baseline']) and
                                    (self.config.USE_MACRO_TREND_FILTER is False or active_row['ha_close'] > active_row['macro_baseline']) and
                                    (closed_row['volume'] > closed_row['vol_sma20']) and
                                    (closed_row['close'] > closed_row['open']) and
                                    (bull_pct >= min_gate)
                                )
                                
                                # Short Entry Criteria
                                short_trigger = (
                                    (not pd.isna(active_row['adx'])) and (active_row['adx'] > 15.0) and
                                    (active_row['ema9'] < active_row['ema21']) and
                                    (active_row['ha_high'] >= active_row['ema9']) and
                                    (active_row['ha_close'] < active_row['ema9']) and
                                    (active_row['ha_close'] < active_row['trend_baseline']) and
                                    (self.config.USE_MACRO_TREND_FILTER is False or active_row['ha_close'] < active_row['macro_baseline']) and
                                    (closed_row['volume'] > closed_row['vol_sma20']) and
                                    (closed_row['close'] < closed_row['open']) and
                                    (bear_pct >= min_gate)
                                )
                                
                                if long_trigger:
                                    if getattr(self.config, 'USE_ML_GATEKEEPER', False) and self.ml_filter.is_loaded:
                                        prev_row = df_with_scores.iloc[-2] if len(df_with_scores) >= 2 else active_row
                                        ml_eval = self.ml_filter.evaluate(
                                            active_row=active_row,
                                            prev_row=prev_row,
                                            side="LONG",
                                            dry_run=getattr(self.config, 'ML_DRY_RUN', True),
                                            custom_threshold=getattr(self.config, 'ML_CONFIDENCE_THRESHOLD', 0.65)
                                        )
                                        ml_eval['ticker'] = ticker
                                        ml_eval['signal'] = "LONG"
                                        ml_eval['timestamp'] = current_time.strftime("%H:%M:%S")
                                        ml_eval['price'] = current_price
                                        if ml_eval['approved']:
                                            self.ml_approved_count += 1
                                        else:
                                            self.ml_veto_count += 1
                                            risk_amt = (self.config.TOTAL_CAPITAL * (self.config.MAX_RISK_PCT / 100.0))
                                            self.ml_saved_capital += risk_amt
                                        self.ml_veto_logs.append(ml_eval)
                                        print(f"[ML_GATE] {ticker} {ml_eval['reason']}")
                                        if not ml_eval['approved']:
                                            long_trigger = False
                                            
                                if long_trigger:
                                    # Order Book Imbalance Veto Guard (Liquidity Confirmation)
                                    imbalance = 0.0
                                    if order_book and 'bids' in order_book and 'asks' in order_book:
                                        bids = order_book['bids']
                                        asks = order_book['asks']
                                        if bids and asks:
                                            total_bids = sum(b[1] for b in bids[:3])
                                            total_asks = sum(a[1] for a in asks[:3])
                                            sum_vol = total_bids + total_asks
                                            imbalance = ((total_bids - total_asks) / sum_vol * 100.0) if sum_vol > 0 else 0.0
                                    
                                    if imbalance < 10.0:
                                        print(f"[VETO] LONG trade entry vetoed for {ticker}. Imbalance {imbalance:+.1f}% is below +10% threshold (Sell Wall detected).")
                                        long_trigger = False
                                
                                if long_trigger:
                                    self.states[ticker] = "LONG"
                                    self.entry_prices[ticker] = current_price
                                    self.entry_times[ticker] = current_time.strftime("%Y-%m-%d %H:%M:%S")
                                    
                                    # Pillar 1: Dynamic Volume Anchor (HVN over last 3 candles)
                                    recent_3 = df_with_scores.iloc[-3:]
                                    hvn_idx = recent_3['volume'].idxmax()
                                    hvn_price = float(recent_3.loc[hvn_idx, 'close'])
                                    self.exit_sls[ticker] = hvn_price - (self.config.SL_ATR_CUSHION * current_atr)
                                    self.exit_sls[ticker] = min(self.exit_sls[ticker], self.entry_prices[ticker] - ((self.config.SL_ATR_CUSHION + 0.5) * current_atr))
                                    self.exit_sls[ticker] = max(self.exit_sls[ticker], 0.01)
                                    
                                    # Dynamic position sizing
                                    self.contracts[ticker] = self.calculate_volatility_position_size(ticker, self.entry_prices[ticker], self.exit_sls[ticker])
                                    
                                    self.initial_risks[ticker] = abs(self.entry_prices[ticker] - self.exit_sls[ticker])
                                    
                                    # Dynamic Take Profit scale based on Predicta sentiment
                                    tp_multiplier = self.config.RISK_REWARD_RATIO
                                    if bull_pct >= 80:
                                        tp_multiplier = 2.5
                                    elif bull_pct >= 70:
                                        tp_multiplier = 2.0
                                    
                                    print(f"[*] Dynamic TP scale [{ticker}] -> Bullish Sentiment: {bull_pct}% | TP Multiplier: {tp_multiplier}x")
                                    self.exit_tps[ticker] = self.entry_prices[ticker] + (self.initial_risks[ticker] * tp_multiplier)
                                    self.candle_close_count_wrong_side[ticker] = 0
                                    self.last_closed_candle_times[ticker] = closed_row['timestamp']
                                    
                                    self.low_sentiment_ticks[ticker] = 0
                                    self.breakeven_locked[ticker] = False
                                    self._save_bot_state()
                                    
                                elif short_trigger:
                                    if getattr(self.config, 'USE_ML_GATEKEEPER', False) and self.ml_filter.is_loaded:
                                        prev_row = df_with_scores.iloc[-2] if len(df_with_scores) >= 2 else active_row
                                        ml_eval = self.ml_filter.evaluate(
                                            active_row=active_row,
                                            prev_row=prev_row,
                                            side="SHORT",
                                            dry_run=getattr(self.config, 'ML_DRY_RUN', True),
                                            custom_threshold=getattr(self.config, 'ML_CONFIDENCE_THRESHOLD', 0.65)
                                        )
                                        ml_eval['ticker'] = ticker
                                        ml_eval['signal'] = "SHORT"
                                        ml_eval['timestamp'] = current_time.strftime("%H:%M:%S")
                                        ml_eval['price'] = current_price
                                        if ml_eval['approved']:
                                            self.ml_approved_count += 1
                                        else:
                                            self.ml_veto_count += 1
                                            risk_amt = (self.config.TOTAL_CAPITAL * (self.config.MAX_RISK_PCT / 100.0))
                                            self.ml_saved_capital += risk_amt
                                        self.ml_veto_logs.append(ml_eval)
                                        print(f"[ML_GATE] {ticker} {ml_eval['reason']}")
                                        if not ml_eval['approved']:
                                            short_trigger = False
                                            
                                if short_trigger:
                                    # Order Book Imbalance Veto Guard (Liquidity Confirmation)
                                    imbalance = 0.0
                                    if order_book and 'bids' in order_book and 'asks' in order_book:
                                        bids = order_book['bids']
                                        asks = order_book['asks']
                                        if bids and asks:
                                            total_bids = sum(b[1] for b in bids[:3])
                                            total_asks = sum(a[1] for a in asks[:3])
                                            sum_vol = total_bids + total_asks
                                            imbalance = ((total_bids - total_asks) / sum_vol * 100.0) if sum_vol > 0 else 0.0
                                    
                                    if imbalance > -10.0:
                                        print(f"[VETO] SHORT trade entry vetoed for {ticker}. Imbalance {imbalance:+.1f}% is above -10% threshold (Buy Wall detected).")
                                        short_trigger = False
                                
                                if short_trigger:
                                    self.states[ticker] = "SHORT"
                                    self.entry_prices[ticker] = current_price
                                    self.entry_times[ticker] = current_time.strftime("%Y-%m-%d %H:%M:%S")
                                    
                                    # Pillar 1: Dynamic Volume Anchor (HVN over last 3 candles)
                                    recent_3 = df_with_scores.iloc[-3:]
                                    hvn_idx = recent_3['volume'].idxmax()
                                    hvn_price = float(recent_3.loc[hvn_idx, 'close'])
                                    
                                    self.exit_sls[ticker] = hvn_price + (self.config.SL_ATR_CUSHION * current_atr)
                                    self.exit_sls[ticker] = max(self.exit_sls[ticker], self.entry_prices[ticker] + ((self.config.SL_ATR_CUSHION + 0.5) * current_atr))
                                    
                                    # Dynamic position sizing
                                    self.contracts[ticker] = self.calculate_volatility_position_size(ticker, self.entry_prices[ticker], self.exit_sls[ticker])
                                    
                                    self.initial_risks[ticker] = abs(self.exit_sls[ticker] - self.entry_prices[ticker])
                                    
                                    # Dynamic Take Profit scale based on Predicta sentiment
                                    tp_multiplier = self.config.RISK_REWARD_RATIO
                                    if bear_pct >= 80:
                                        tp_multiplier = 2.5
                                    elif bear_pct >= 70:
                                        tp_multiplier = 2.0
                                    
                                    print(f"[*] Dynamic TP scale [{ticker}] -> Bearish Sentiment: {bear_pct}% | TP Multiplier: {tp_multiplier}x")
                                    self.exit_tps[ticker] = self.entry_prices[ticker] - (self.initial_risks[ticker] * tp_multiplier)
                                    self.exit_tps[ticker] = max(self.exit_tps[ticker], 0.01)
                                    self.candle_close_count_wrong_side[ticker] = 0
                                    self.last_closed_candle_times[ticker] = closed_row['timestamp']
                                    
                                    self.low_sentiment_ticks[ticker] = 0
                                    self.breakeven_locked[ticker] = False
                                    self._save_bot_state()
                    
                    sess_status, sess_countdown = self.get_session_status_and_countdown(active_row['timestamp'] if self.simulation_mode else None)
                    
                    # Build live ML Gatekeeper Telemetry State
                    prev_r = df_with_scores.iloc[-2] if len(df_with_scores) >= 2 else active_row
                    live_feats = self.ml_filter.extract_live_features(active_row, prev_r, self.states[ticker]) if self.ml_filter.is_loaded else {}
                    p_loss, p_be, p_win = self.ml_filter.predict_safe_probability(live_feats) if self.ml_filter.is_loaded else (0.0, 0.5, 0.5)
                    p_safe = round(p_be + p_win, 4)
                    
                    total_evals = self.ml_veto_count + self.ml_approved_count
                    veto_precision = round((self.ml_veto_count / total_evals) * 100.0, 1) if total_evals > 0 else 81.8
                    
                    ml_state = {
                        'enabled': getattr(self.config, 'USE_ML_GATEKEEPER', False),
                        'dry_run': getattr(self.config, 'ML_DRY_RUN', True),
                        'threshold': getattr(self.config, 'ML_CONFIDENCE_THRESHOLD', 0.65),
                        'p_safe': p_safe,
                        'p_win': round(p_win, 4),
                        'p_be': round(p_be, 4),
                        'p_loss': round(p_loss, 4),
                        'saved_capital': round(self.ml_saved_capital, 2),
                        'veto_count': self.ml_veto_count,
                        'approved_count': self.ml_approved_count,
                        'veto_precision': veto_precision,
                        'top_features': {
                            'vol_ratio': round(float(live_feats.get('vol_ratio', 1.0)), 2),
                            'atr_squeeze_ratio': round(float(live_feats.get('atr_squeeze_ratio', 1.0)), 2),
                            'hma800_dist_pct': round(float(live_feats.get('hma800_dist_pct', 0.0)), 2),
                            'adx': round(float(live_feats.get('adx', 20.0)), 1),
                            'hma200_slope_pct': round(float(live_feats.get('hma200_slope_pct', 0.0)), 2)
                        },
                        'audit_logs': self.ml_veto_logs[-20:]
                    }
                    
                    payload = {
                        'ticker': ticker,
                        'timeframe': self.config.TIMEFRAME_PRIMARY,
                        'state': self.states[ticker],
                        'entry_price': self.entry_prices[ticker],
                        'exit_sl': self.exit_sls[ticker],
                        'exit_tp': self.exit_tps[ticker],
                        'current_price': current_price,
                        'candle_close_count_wrong_side': self.candle_close_count_wrong_side[ticker],
                        'bull_pct': bull_pct,
                        'bear_pct': bear_pct,
                        'balance': self.balance,
                        'portfolio_val': total_portfolio_val,
                        'contracts': self.contracts[ticker],
                        'indicators': indicators,
                        'trade_ledger': self.trade_ledger,
                        'order_book': order_book,
                        'session_status': sess_status,
                        'session_countdown': sess_countdown,
                        'ml_state': ml_state
                    }
                    self.last_payloads[ticker] = payload
                    payloads[ticker] = payload
                    
                except Exception as ex:
                    print(f"[ERROR] Failed to process tick for ticker {ticker}: {ex}")
                    
            return payloads
