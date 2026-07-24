# src/indicators.py
import pandas as pd
import numpy as np

def calculate_wma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Weighted Moving Average."""
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

def calculate_hma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Hull Moving Average."""
    wma_half = calculate_wma(series, period // 2)
    wma_full = calculate_wma(series, period)
    diff = 2 * wma_half - wma_full
    return calculate_wma(diff, int(np.sqrt(period)))

def convert_to_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw standard OHLCV DataFrame into a smoothed Heikin-Ashi DataFrame.
    Formula:
    HA_Close = (Open + High + Low + Close) / 4
    HA_Open = (HA_Open_prev + HA_Close_prev) / 2
    HA_High = max(High, HA_Open, HA_Close)
    HA_Low = min(Low, HA_Open, HA_Close)
    """
    ha_df = df.copy()
    
    # Calculate HA_Close first (fully vectorized)
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    
    # Calculate HA_Open recursively
    ha_open = np.zeros(len(df))
    # Initialize HA_Open_0 as average of Open_0 and Close_0
    ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2.0
    
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0
        
    ha_df['open'] = ha_open
    ha_df['close'] = ha_close
    ha_df['high'] = np.maximum(df['high'], np.maximum(ha_open, ha_close))
    ha_df['low'] = np.minimum(df['low'], np.minimum(ha_open, ha_close))
    
    return ha_df

def calculate_indicators(df: pd.DataFrame, trend_len: int = 200, trend_type: str = "HMA") -> pd.DataFrame:
    """
    Append mathematical indicators to the DataFrame.
    Takes standard raw OHLCV DataFrame, converts to Heikin-Ashi, and appends indicators.
    """
    # 1. Convert to Heikin-Ashi
    ha_df = convert_to_heikin_ashi(df)
    
    # Outputs DataFrame containing both raw and HA indicator columns
    out_df = df.copy()
    
    # Store Heikin-Ashi Columns with prefix 'ha_'
    out_df['ha_open'] = ha_df['open']
    out_df['ha_high'] = ha_df['high']
    out_df['ha_low'] = ha_df['low']
    out_df['ha_close'] = ha_df['close']
    
    # Fast & Slow Core EMAs over HA_Close
    out_df['ema8'] = out_df['ha_close'].ewm(span=8, adjust=False).mean()
    out_df['ema9'] = out_df['ha_close'].ewm(span=9, adjust=False).mean()
    out_df['ema21'] = out_df['ha_close'].ewm(span=21, adjust=False).mean()
    out_df['ema50'] = out_df['ha_close'].ewm(span=50, adjust=False).mean()
    
    # Trend Baseline over HA_Close (200 HMA or 200 EMA)
    if trend_type.upper() == "HMA":
        out_df['trend_baseline'] = calculate_hma(out_df['ha_close'], trend_len)
    else:
        out_df['trend_baseline'] = out_df['ha_close'].ewm(span=trend_len, adjust=False).mean()
        
    # Macro Trend Baseline (1-Hour equivalent filter = 800 HMA on 15m)
    out_df['macro_baseline'] = calculate_hma(out_df['ha_close'], 800)
        
    # ATR: 14-period Average True Range calculated on raw standard candle data.
    prev_close = out_df['close'].shift(1)
    tr1 = out_df['high'] - out_df['low']
    tr2 = (out_df['high'] - prev_close).abs()
    tr3 = (out_df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = pd.Series(np.nan, index=out_df.index)
    if len(out_df) >= 14:
        atr.iloc[13] = tr.iloc[0:14].mean()
        for i in range(14, len(out_df)):
            atr.iloc[i] = (atr.iloc[i - 1] * 13 + tr.iloc[i]) / 14.0
    out_df['atr'] = atr
    
    # RSI: 14-period Relative Strength Index on raw standard Close.
    delta = out_df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = pd.Series(np.nan, index=out_df.index)
    avg_loss = pd.Series(np.nan, index=out_df.index)
    
    if len(out_df) >= 15:
        avg_gain.iloc[14] = gain.iloc[1:15].mean()
        avg_loss.iloc[14] = loss.iloc[1:15].mean()
        for i in range(15, len(out_df)):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * 13 + gain.iloc[i]) / 14.0
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * 13 + loss.iloc[i]) / 14.0
            
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    out_df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
    
    # MACD: Fast=12, Slow=26, Signal=9 over raw standard close.
    ema12 = out_df['close'].ewm(span=12, adjust=False).mean()
    ema26 = out_df['close'].ewm(span=26, adjust=False).mean()
    out_df['macd_line'] = ema12 - ema26
    out_df['signal_line'] = out_df['macd_line'].ewm(span=9, adjust=False).mean()
    out_df['macd_hist'] = out_df['macd_line'] - out_df['signal_line']
    
    # Stochastics: 14-period Fast %K, smoothed with a 3-period SMA to get %D.
    lowest_low = out_df['low'].rolling(14).min()
    highest_high = out_df['high'].rolling(14).max()
    out_df['stoch_k'] = 100.0 * (out_df['close'] - lowest_low) / (highest_high - lowest_low).replace(0, 1e-9)
    out_df['stoch_d'] = out_df['stoch_k'].rolling(3).mean()
    
    # DMI / ADX: 14-period directional movement engine
    up_move = out_df['high'].diff()
    down_move = -out_df['low'].diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr14 = pd.Series(np.nan, index=out_df.index)
    pdm14 = pd.Series(np.nan, index=out_df.index)
    mdm14 = pd.Series(np.nan, index=out_df.index)
    
    if len(out_df) >= 14:
        tr14.iloc[13] = tr.iloc[0:14].mean()
        pdm14.iloc[13] = plus_dm[0:14].sum() / 14.0
        mdm14.iloc[13] = minus_dm[0:14].sum() / 14.0
        
        for i in range(14, len(out_df)):
            tr14.iloc[i] = (tr14.iloc[i - 1] * 13 + tr.iloc[i]) / 14.0
            pdm14.iloc[i] = (pdm14.iloc[i - 1] * 13 + plus_dm[i]) / 14.0
            mdm14.iloc[i] = (mdm14.iloc[i - 1] * 13 + minus_dm[i]) / 14.0
            
    out_df['plus_di'] = 100.0 * (pdm14 / tr14.replace(0, 1e-9))
    out_df['minus_di'] = 100.0 * (mdm14 / tr14.replace(0, 1e-9))
    
    dx = 100.0 * (out_df['plus_di'] - out_df['minus_di']).abs() / (out_df['plus_di'] + out_df['minus_di']).replace(0, 1e-9)
    
    adx = pd.Series(np.nan, index=out_df.index)
    if len(out_df) >= 27:
        adx.iloc[26] = dx.iloc[13:27].mean()
        for i in range(27, len(out_df)):
            adx.iloc[i] = (adx.iloc[i - 1] * 13 + dx.iloc[i]) / 14.0
    out_df['adx'] = adx
    
    # Volume Delta Engine:
    candle_range = out_df['high'] - out_df['low']
    buy_volume = np.where(candle_range > 0, out_df['volume'] * (out_df['close'] - out_df['low']) / candle_range, out_df['volume'] * 0.5)
    sell_volume = np.where(candle_range > 0, out_df['volume'] * (out_df['high'] - out_df['close']) / candle_range, out_df['volume'] * 0.5)
    
    out_df['volume_delta'] = buy_volume - sell_volume
    out_df['delta_ema'] = out_df['volume_delta'].ewm(span=10, adjust=False).mean()
    out_df['delta_momentum'] = out_df['volume_delta'] > out_df['delta_ema']
    
    # 20-period SMA volume for Volume Component score
    out_df['vol_sma20'] = out_df['volume'].rolling(20).mean()
    
    return out_df

def calculate_predicta_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Predicta V4 Scores (Bullish & Bearish) for each row.
    Returns the DataFrame with additional columns:
    'raw_bull_score', 'raw_bear_score', 'atr_percent_rank', 'vol_multiplier',
    'final_bull_percentage', 'final_bear_percentage'.
    """
    # 1. Rolling 100-period Percent Rank of ATR
    def rank_func(x):
        val = x[-1]
        if np.isnan(val):
            return np.nan
        valid = x[~np.isnan(x)]
        if len(valid) == 0:
            return np.nan
        return (valid <= val).sum() / len(valid) * 100.0
        
    df['atr_percent_rank'] = df['atr'].rolling(100, min_periods=1).apply(rank_func, raw=True)
    
    # 2. Establish Volatility Multiplier
    df['vol_multiplier'] = 1.0
    df.loc[df['atr_percent_rank'] > 75.0, 'vol_multiplier'] = 0.85
    df.loc[df['atr_percent_rank'] < 25.0, 'vol_multiplier'] = 1.15
    
    # Pre-allocate score arrays
    raw_bull_scores = []
    raw_bear_scores = []
    
    # We need to iterate to correctly check current vs previous rows (e.g. ema8[0] vs ema8[-1])
    for idx in range(len(df)):
        if idx == 0:
            raw_bull_scores.append(50.0)
            raw_bear_scores.append(50.0)
            continue
            
        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]
        
        # --- BULLISH SCORE ---
        # 1. Trend Component (23% Weight)
        if row['ema8'] > row['ema21'] and row['ema21'] > row['ema50']:
            trend_bull = 100.0
        elif row['ema8'] > row['ema21']:
            trend_bull = 80.0
        elif row['ema8'] > prev_row['ema8']:
            trend_bull = 60.0
        else:
            trend_bull = 0.0
            
        # 2. MACD Component (18% Weight)
        if row['macd_line'] > row['signal_line'] and row['macd_hist'] > 0:
            macd_bull = 100.0
        elif row['macd_line'] > row['signal_line']:
            macd_bull = 70.0
        elif row['macd_hist'] > 0:
            macd_bull = 50.0
        else:
            macd_bull = 20.0
            
        # 3. Delta Component (15% Weight)
        if row['volume_delta'] > 0 and row['delta_momentum']:
            delta_bull = 100.0
        elif row['volume_delta'] > 0:
            delta_bull = 75.0
        elif row['volume_delta'] > -abs(row['delta_ema']):
            delta_bull = 40.0
        else:
            delta_bull = 20.0
            
        # 4. RSI Component (12% Weight)
        if row['rsi'] < 30.0:
            rsi_bull = 100.0
        elif row['rsi'] < 40.0:
            rsi_bull = 85.0
        elif row['rsi'] < 50.0:
            rsi_bull = 70.0
        elif row['rsi'] < 60.0:
            rsi_bull = 50.0
        else:
            rsi_bull = 25.0
            
        # 5. Stoch Component (12% Weight)
        if row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < 20.0:
            stoch_bull = 100.0
        elif row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < 50.0:
            stoch_bull = 85.0
        elif row['stoch_k'] > row['stoch_d']:
            stoch_bull = 65.0
        else:
            stoch_bull = 25.0
            
        # 6. ADX Component (10% Weight)
        if row['adx'] > 35.0:
            adx_bull = 100.0
        elif row['adx'] > 30.0:
            adx_bull = 85.0
        elif row['adx'] > 25.0:
            adx_bull = 70.0
        elif row['adx'] > 20.0:
            adx_bull = 50.0
        else:
            adx_bull = 30.0
            
        # 7. Volume Component (10% Weight)
        if pd.isna(row['vol_sma20']) or row['vol_sma20'] == 0:
            vol_ratio = 1.0
        else:
            vol_ratio = row['volume'] / row['vol_sma20']
            
        if vol_ratio > 2.0:
            vol_bull = 100.0
        elif vol_ratio > 1.5:
            vol_bull = 80.0
        elif vol_ratio > 1.0:
            vol_bull = 60.0
        elif vol_ratio > 0.8:
            vol_bull = 45.0
        else:
            vol_bull = 25.0
            
        # High-Conviction Breakout Override (Bypasses Overbought Lockout)
        is_strong_trend_bull = (
            (row['close'] > row['trend_baseline']) and 
            (row['trend_baseline'] > prev_row['trend_baseline']) and 
            (row['ema9'] > row['ema21'])
        )
        non_oscillator_bull = (
            0.23 * trend_bull + 
            0.18 * macd_bull + 
            0.15 * delta_bull + 
            0.10 * adx_bull + 
            0.10 * vol_bull
        ) / 0.76
        
        if is_strong_trend_bull and non_oscillator_bull >= 75.0 and vol_ratio > 1.2:
            rsi_bull = 80.0
            stoch_bull = 80.0
            
        raw_bull = (
            0.23 * trend_bull +
            0.18 * macd_bull +
            0.15 * delta_bull +
            0.12 * rsi_bull +
            0.12 * stoch_bull +
            0.10 * adx_bull +
            0.10 * vol_bull
        )
        raw_bull_scores.append(raw_bull)
        
        # --- BEARISH SCORE (Mirrored Rules) ---
        # 1. Trend Component (23% Weight)
        if row['ema8'] < row['ema21'] and row['ema21'] < row['ema50']:
            trend_bear = 100.0
        elif row['ema8'] < row['ema21']:
            trend_bear = 80.0
        elif row['ema8'] < prev_row['ema8']:
            trend_bear = 60.0
        else:
            trend_bear = 0.0
            
        # 2. MACD Component (18% Weight)
        if row['macd_line'] < row['signal_line'] and row['macd_hist'] < 0:
            macd_bear = 100.0
        elif row['macd_line'] < row['signal_line']:
            macd_bear = 70.0
        elif row['macd_hist'] < 0:
            macd_bear = 50.0
        else:
            macd_bear = 20.0
            
        # 3. Delta Component (15% Weight)
        # Mirror: volume_delta < 0 and delta_momentum is False
        if row['volume_delta'] < 0 and not row['delta_momentum']:
            delta_bear = 100.0
        elif row['volume_delta'] < 0:
            delta_bear = 75.0
        elif row['volume_delta'] < abs(row['delta_ema']):
            delta_bear = 40.0
        else:
            delta_bear = 20.0
            
        # 4. RSI Component (12% Weight) - Mirrored around 50
        if row['rsi'] > 70.0:
            rsi_bear = 100.0
        elif row['rsi'] > 60.0:
            rsi_bear = 85.0
        elif row['rsi'] > 50.0:
            rsi_bear = 70.0
        elif row['rsi'] > 40.0:
            rsi_bear = 50.0
        else:
            rsi_bear = 25.0
            
        # 5. Stoch Component (12% Weight) - Mirrored around 50/80
        if row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > 80.0:
            stoch_bear = 100.0
        elif row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > 50.0:
            stoch_bear = 85.0
        elif row['stoch_k'] < row['stoch_d']:
            stoch_bear = 65.0
        else:
            stoch_bear = 25.0
            
        # 6. ADX Component (10% Weight) - Non-directional
        adx_bear = adx_bull
        
        # 7. Volume Component (10% Weight) - Non-directional
        vol_bear = vol_bull
        
        # High-Conviction Breakout Override (Bypasses Oversold Lockout)
        is_strong_trend_bear = (
            (row['close'] < row['trend_baseline']) and 
            (row['trend_baseline'] < prev_row['trend_baseline']) and 
            (row['ema9'] < row['ema21'])
        )
        non_oscillator_bear = (
            0.23 * trend_bear + 
            0.18 * macd_bear + 
            0.15 * delta_bear + 
            0.10 * adx_bear + 
            0.10 * vol_bear
        ) / 0.76
        
        if is_strong_trend_bear and non_oscillator_bear >= 75.0 and vol_ratio > 1.2:
            rsi_bear = 80.0
            stoch_bear = 80.0
            
        raw_bear = (
            0.23 * trend_bear +
            0.18 * macd_bear +
            0.15 * delta_bear +
            0.12 * rsi_bear +
            0.12 * stoch_bear +
            0.10 * adx_bear +
            0.10 * vol_bear
        )
        raw_bear_scores.append(raw_bear)

    df['raw_bull_score'] = raw_bull_scores
    df['raw_bear_score'] = raw_bear_scores
    
    # 3. Apply Volatility Normalization & Clip
    long_score = (df['raw_bull_score'] * df['vol_multiplier']).clip(0, 100)
    short_score = (df['raw_bear_score'] * df['vol_multiplier']).clip(0, 100)
    
    # 4. Final output percentages
    sum_scores = long_score + short_score
    
    # Safe division to prevent Division by Zero
    final_bull = np.where(sum_scores > 0, (long_score / sum_scores) * 100.0, 50.0)
    
    df['final_bull_percentage'] = np.round(final_bull).astype(int)
    df['final_bear_percentage'] = 100 - df['final_bull_percentage']
    
    return df
