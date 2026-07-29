#!/usr/bin/env python3
"""
scripts/harvest_ml_dataset.py
--------------------------------
Institutional-Grade Historical Data Harvester & Lifecycle Simulator for Onyx ML Gatekeeper.

Simulates the full dynamic lifecycle of Onyx trades:
1. Break-even SL trailing: Moves SL to entry price when trade hits +1.0R profit.
2. Multi-class outcome labeling: CLEAN_WIN (2), BREAKEVEN (1), HARD_LOSS (0).
3. Continuous Net R-multiple returns (-1.0R to +2.5R).
4. Advanced Volatility Squeeze (ATR ratio), Multi-timeframe alignment, and Momentum Slopes.
"""

import os
import sys
import time
import math
import argparse
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ccxt

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
OUTPUT_CSV = os.path.join(DATA_DIR, "ml_trading_dataset.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

TICKERS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "LINK/USDT"]
CCXT_MAP = {
    "BTC/USDT": "BTC/USDT:USDT",
    "ETH/USDT": "ETH/USDT:USDT",
    "SOL/USDT": "SOL/USDT:USDT",
    "BNB/USDT": "BNB/USDT:USDT",
    "LINK/USDT": "LINK/USDT:USDT"
}

def wma(series: pd.Series, length: int) -> pd.Series:
    """Calculates Weighted Moving Average."""
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda s: np.dot(s, weights) / weights.sum(), raw=True)

def hma(series: pd.Series, length: int) -> pd.Series:
    """Calculates Hull Moving Average."""
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = wma(series, half_length)
    wma_full = wma(series, length)
    raw_hma = 2 * wma_half - wma_full
    return wma(raw_hma, sqrt_length)

def fetch_multi_year_ohlcv(exchange: ccxt.binance, symbol: str, timeframe: str = '15m', years: int = 3) -> pd.DataFrame:
    """
    Fetches up to N years of 15m OHLCV data from Binance, caching results to disk.
    """
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    cache_file = os.path.join(CACHE_DIR, f"{safe_symbol}_{timeframe}_{years}y.csv")
    
    if os.path.exists(cache_file):
        print(f"[DATA] Loading cached dataset for {symbol} ({cache_file})...")
        df = pd.read_csv(cache_file)
        if len(df) > 5000:
            return df
            
    print(f"[DATA] Fetching {years} years of {timeframe} data for {symbol} from Binance...")
    now_ms = int(time.time() * 1000)
    target_start_ms = now_ms - (years * 365 * 24 * 60 * 60 * 1000)
    
    all_ohlcv = []
    current_since = target_start_ms
    
    while current_since < now_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1
            print(f"  --> Downloaded {len(all_ohlcv)} bars... ({datetime.fromtimestamp(current_since/1000, timezone.utc).strftime('%Y-%m-%d')})")
            time.sleep(0.12)
        except Exception as e:
            print(f"[WARN] Fetch error for {symbol} at {current_since}: {e}. Retrying...")
            time.sleep(1.0)
            
    if not all_ohlcv:
        print(f"[ERROR] Could not fetch data for {symbol}.")
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df.to_csv(cache_file, index=False)
    print(f"[DATA] Saved {len(df)} bars to cache: {cache_file}")
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes all technical indicators required by Onyx and advanced ML features."""
    df = df.copy()
    
    # Standard EMAs
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_max20'] = df['volume'].rolling(20).max()
    
    # HMA Baselines
    df['trend_baseline'] = hma(df['close'], 200)
    df['macro_baseline'] = hma(df['close'], 800)
    
    # RSI (14)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_slope3'] = df['rsi'] - df['rsi'].shift(3)
    
    # ATR (14) & Long-Term Volatility Context
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
    df['atr_sma100'] = df['atr'].rolling(100).mean()
    df['atr_squeeze_ratio'] = df['atr'] / (df['atr_sma100'] + 1e-8)
    
    # ADX (14)
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / (df['atr'] + 1e-8))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / (df['atr'] + 1e-8))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()
    df['adx_slope3'] = df['adx'] - df['adx'].shift(3)
    
    # Heikin-Ashi Candles
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    ha_open = np.zeros(len(df))
    ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2.0
    ha_high = np.maximum.reduce([df['high'].values, ha_open, ha_close.values])
    ha_low = np.minimum.reduce([df['low'].values, ha_open, ha_close.values])
    
    df['ha_open'] = ha_open
    df['ha_high'] = ha_high
    df['ha_low'] = ha_low
    df['ha_close'] = ha_close
    
    return df

def extract_features_and_outcomes(df: pd.DataFrame, ticker: str) -> list[dict]:
    """
    Scans historical dataframe for Onyx strategy entry setups, extracts 30+ features,
    and forward-simulates trade outcome WITH DYNAMIC BREAK-EVEN STOP-LOSS TRAILING.
    """
    records = []
    n = len(df)
    
    # Minimum warmup required for HMA 800
    start_idx = 850
    
    for i in range(start_idx, n - 100):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        
        # Check required fields non-NaN
        if pd.isna(row['trend_baseline']) or pd.isna(row['macro_baseline']) or pd.isna(row['adx']) or pd.isna(row['atr']):
            continue
            
        # Onyx Strategy Rule Conditions
        adx_valid = row['adx'] > 15.0
        vol_valid = row['volume'] > row['vol_sma20']
        
        long_trigger = (
            adx_valid and
            (row['ema9'] > row['ema21']) and
            (row['ha_low'] <= row['ema9']) and
            (row['ha_close'] > row['ema9']) and
            (row['ha_close'] > row['trend_baseline']) and
            (row['ha_close'] > row['macro_baseline']) and
            vol_valid and
            (row['close'] > row['open'])
        )
        
        short_trigger = (
            adx_valid and
            (row['ema9'] < row['ema21']) and
            (row['ha_high'] >= row['ema9']) and
            (row['ha_close'] < row['ema9']) and
            (row['ha_close'] < row['trend_baseline']) and
            (row['ha_close'] < row['macro_baseline']) and
            vol_valid and
            (row['close'] < row['open'])
        )
        
        if not (long_trigger or short_trigger):
            continue
            
        side = "LONG" if long_trigger else "SHORT"
        entry_price = float(row['close'])
        atr = float(row['atr'])
        
        # HVN Structural Stop-Loss calculation (last 3 candles HVN)
        last_3 = df.iloc[i-2:i+1]
        hvn_idx = last_3['volume'].idxmax()
        hvn_price = float(last_3.loc[hvn_idx, 'close'])
        
        if side == "LONG":
            sl_price = hvn_price - (1.5 * atr)
            sl_price = min(sl_price, entry_price - (2.0 * atr))
            sl_price = max(sl_price, 0.01)
            risk = entry_price - sl_price
            tp_price = entry_price + (1.5 * risk)
            be_threshold = entry_price + (1.0 * risk)  # Arm break-even at +1.0R
        else:
            sl_price = hvn_price + (1.5 * atr)
            sl_price = max(sl_price, entry_price + (2.0 * atr))
            risk = sl_price - entry_price
            tp_price = entry_price - (1.5 * risk)
            be_threshold = entry_price - (1.0 * risk)  # Arm break-even at +1.0R
            
        if risk <= 0:
            continue
            
        # Realistic Forward Simulation Engine with +1.0R Break-even Trail
        outcome_class = 0  # 0: HARD_LOSS, 1: BREAKEVEN, 2: CLEAN_WIN
        net_r = -1.0
        max_rr = 0.0
        sl_moved_to_be = False
        current_sl = sl_price
        trade_resolved = False
        
        for f_idx in range(i + 1, min(i + 200, n)):
            f_row = df.iloc[f_idx]
            
            if side == "LONG":
                curr_rr = (f_row['high'] - entry_price) / risk
                if curr_rr > max_rr:
                    max_rr = curr_rr
                    
                # 1. Check if price hit +1.0R profit -> Arm Break-even Stop Loss
                if f_row['high'] >= be_threshold and not sl_moved_to_be:
                    sl_moved_to_be = True
                    current_sl = entry_price  # Move SL to Entry Price!
                    
                # 2. Check TP / SL hits
                if f_row['high'] >= tp_price and f_row['low'] <= current_sl:
                    # Same bar conflict -> if SL moved to BE, resolve as BE; else HARD_LOSS
                    outcome_class = 1 if sl_moved_to_be else 0
                    net_r = 0.0 if sl_moved_to_be else -1.0
                    trade_resolved = True
                    break
                elif f_row['high'] >= tp_price:
                    outcome_class = 2  # CLEAN_WIN
                    net_r = 1.5
                    trade_resolved = True
                    break
                elif f_row['low'] <= current_sl:
                    if sl_moved_to_be:
                        outcome_class = 1  # BREAKEVEN (0R loss!)
                        net_r = 0.0
                    else:
                        outcome_class = 0  # HARD_LOSS (-1R loss)
                        net_r = -1.0
                    trade_resolved = True
                    break
            else:
                curr_rr = (entry_price - f_row['low']) / risk
                if curr_rr > max_rr:
                    max_rr = curr_rr
                    
                if f_row['low'] <= be_threshold and not sl_moved_to_be:
                    sl_moved_to_be = True
                    current_sl = entry_price
                    
                if f_row['low'] <= tp_price and f_row['high'] >= current_sl:
                    outcome_class = 1 if sl_moved_to_be else 0
                    net_r = 0.0 if sl_moved_to_be else -1.0
                    trade_resolved = True
                    break
                elif f_row['low'] <= tp_price:
                    outcome_class = 2  # CLEAN_WIN
                    net_r = 1.5
                    trade_resolved = True
                    break
                elif f_row['high'] >= current_sl:
                    if sl_moved_to_be:
                        outcome_class = 1  # BREAKEVEN
                        net_r = 0.0
                    else:
                        outcome_class = 0  # HARD_LOSS
                        net_r = -1.0
                    trade_resolved = True
                    break
                    
        if not trade_resolved:
            continue
            
        # Feature Engineering (30+ features normalized for pure ML learning)
        dt = datetime.fromtimestamp(row['timestamp'] / 1000, timezone.utc)
        
        candle_range = (row['high'] - row['low']) + 1e-8
        body_size = abs(row['close'] - row['open'])
        ha_range = (row['ha_high'] - row['ha_low']) + 1e-8
        ha_body_size = abs(row['ha_close'] - row['ha_open'])
        
        features = {
            # Identifier Metadata
            'ticker': ticker,
            'timestamp': row['timestamp'],
            'datetime_utc': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'side': side,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'risk_dist_pct': (risk / entry_price) * 100.0,
            
            # Trend & Moving Average Features
            'ema9_ema21_diff_pct': ((row['ema9'] - row['ema21']) / row['ema21']) * 100.0,
            'close_ema9_diff_pct': ((row['close'] - row['ema9']) / row['ema9']) * 100.0,
            'hma200_dist_pct': ((row['close'] - row['trend_baseline']) / row['trend_baseline']) * 100.0,
            'hma800_dist_pct': ((row['close'] - row['macro_baseline']) / row['macro_baseline']) * 100.0,
            'hma200_slope_pct': ((row['trend_baseline'] - prev_row['trend_baseline']) / prev_row['trend_baseline']) * 100.0,
            
            # Momentum & Volatility Squeeze Features
            'rsi': float(row['rsi']),
            'rsi_slope3': float(row['rsi_slope3']),
            'adx': float(row['adx']),
            'adx_slope3': float(row['adx_slope3']),
            'atr_pct': (atr / entry_price) * 100.0,
            'atr_squeeze_ratio': float(row['atr_squeeze_ratio']),
            'vol_ratio': float(row['volume'] / (row['vol_sma20'] + 1e-8)),
            'vol_surge_ratio': float(row['volume'] / (row['vol_max20'] + 1e-8)),
            
            # Candle Anatomy & Exhaustion Metrics
            'body_to_range_ratio': float(body_size / candle_range),
            'upper_wick_ratio': float((row['high'] - max(row['open'], row['close'])) / candle_range),
            'lower_wick_ratio': float((min(row['open'], row['close']) - row['low']) / candle_range),
            'ha_body_to_range_ratio': float(ha_body_size / ha_range),
            'ha_upper_wick_ratio': float((row['ha_high'] - max(row['ha_open'], row['ha_close'])) / ha_range),
            'ha_lower_wick_ratio': float((min(row['ha_open'], row['ha_close']) - row['ha_low']) / ha_range),
            
            # Time & Cyclical Features
            'hour_of_day': dt.hour,
            'day_of_week': dt.weekday(),
            'hour_sin': math.sin(2 * math.pi * dt.hour / 24.0),
            'hour_cos': math.cos(2 * math.pi * dt.hour / 24.0),
            
            # Multi-Class & Continuous Targets
            'label_class': outcome_class,   # 2: CLEAN_WIN, 1: BREAKEVEN, 0: HARD_LOSS
            'label_net_r': round(net_r, 2), # -1.0, 0.0, +1.5
            'label_win': 1 if outcome_class > 0 else 0, # 1 if positive return (Win/Breakeven), 0 if Hard Loss
            'label_max_rr': round(max_rr, 2)
        }
        
        records.append(features)
        
    return records

def main():
    parser = argparse.ArgumentParser(description="Harvest institutional-grade historical trade features for Onyx ML Gatekeeper.")
    parser.add_argument("--years", type=int, default=3, help="Years of 15m historical data to harvest (default: 3)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ONYX QUANTITATIVE DESK — INSTITUTIONAL ML HARVESTER & LIFECYCLE ENGINE")
    print(f"  Target Horizon: {args.years} Years | Timeframe: 15m | Assets: {len(TICKERS)}")
    print("  Simulating Dynamic +1.0R Break-even Trailing Stop Loss & Multi-Class Targets")
    print("=" * 70)
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    all_dataset_records = []
    
    for ticker in TICKERS:
        ccxt_symbol = CCXT_MAP.get(ticker, ticker)
        df_raw = fetch_multi_year_ohlcv(exchange, ccxt_symbol, timeframe='15m', years=args.years)
        
        if df_raw.empty:
            continue
            
        print(f"[HARVEST] Computing technical & squeeze indicators for {ticker}...")
        df_ind = compute_indicators(df_raw)
        
        print(f"[HARVEST] Extracting strategy triggers & simulating breakeven lifecycles for {ticker}...")
        records = extract_features_and_outcomes(df_ind, ticker)
        print(f"  --> Extracted {len(records)} setup instances for {ticker}.")
        all_dataset_records.extend(records)
        
    dataset_df = pd.DataFrame(all_dataset_records)
    
    if dataset_df.empty:
        print("[ERROR] No trade setups were extracted. Check date ranges or strategy rules.")
        return
        
    dataset_df.to_csv(OUTPUT_CSV, index=False)
    
    clean_wins = len(dataset_df[dataset_df['label_class'] == 2])
    be_scratches = len(dataset_df[dataset_df['label_class'] == 1])
    hard_losses = len(dataset_df[dataset_df['label_class'] == 0])
    total = len(dataset_df)
    
    print("\n" + "=" * 70)
    print("  HARVESTING COMPLETE — UPGRADED DATASET METRICS SUMMARY")
    print("=" * 70)
    print(f"  Total Trade Setups Extracted : {total}")
    print(f"  Clean Wins (+1.5R Hit)      : {clean_wins} ({clean_wins/total*100:.2f}%)")
    print(f"  Breakeven Scratches (0.0R)  : {be_scratches} ({be_scratches/total*100:.2f}%)")
    print(f"  Hard Loss Setups (-1.0R SL) : {hard_losses} ({hard_losses/total*100:.2f}%)")
    print(f"  Positive Return Rate        : {(clean_wins + be_scratches)/total*100:.2f}% (Win + Breakeven)")
    print(f"  Dataset Saved To            : {OUTPUT_CSV}")
    print("=" * 70)

if __name__ == "__main__":
    main()
