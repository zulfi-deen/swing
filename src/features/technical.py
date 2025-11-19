"""Technical indicator features"""

import pandas as pd
import numpy as np
import talib as ta
from typing import Optional


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators using TA-Lib."""
    
    if df.empty or len(df) < 50:
        return df
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    # Trend indicators
    df['sma_20'] = ta.SMA(df['close'], timeperiod=20)
    df['sma_50'] = ta.SMA(df['close'], timeperiod=50)
    df['ema_12'] = ta.EMA(df['close'], timeperiod=12)
    df['ema_26'] = ta.EMA(df['close'], timeperiod=26)
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(
        df['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    # RSI
    df['rsi_14'] = ta.RSI(df['close'], timeperiod=14)
    
    # Bollinger Bands
    df['bbands_upper'], df['bbands_middle'], df['bbands_lower'] = ta.BBANDS(
        df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
    )
    df['bbands_pct'] = (df['close'] - df['bbands_lower']) / (
        df['bbands_upper'] - df['bbands_lower']
    )
    
    # ATR (volatility)
    df['atr_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Stochastic
    df['stoch_k'], df['stoch_d'] = ta.STOCH(
        df['high'], df['low'], df['close'],
        fastk_period=14, slowk_period=3, slowd_period=3
    )
    
    # ADX (trend strength)
    df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Money Flow Index
    df['mfi'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    
    return df


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based features."""
    
    if df.empty:
        return df
    
    # Volume z-score (20-day rolling)
    df['volume_mean_20'] = df['volume'].rolling(20).mean()
    df['volume_std_20'] = df['volume'].rolling(20).std()
    df['volume_z_score'] = (
        (df['volume'] - df['volume_mean_20']) / df['volume_std_20']
    ).fillna(0)
    
    # Average dollar volume
    df['dollar_volume'] = df['close'] * df['volume']
    df['avg_dollar_volume_20'] = df['dollar_volume'].rolling(20).mean()
    
    # On-Balance Volume
    df['obv'] = ta.OBV(df['close'], df['volume'])
    
    # VWAP deviation
    if 'vwap' in df.columns:
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
    else:
        df['vwap_deviation'] = 0.0
    
    return df


def compute_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price action features."""
    
    if df.empty or len(df) < 2:
        return df
    
    # Gap %
    df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Intraday range
    df['intraday_range_pct'] = (df['high'] - df['low']) / df['open']
    
    # Distance to 52-week high/low
    df['high_52w'] = df['high'].rolling(252, min_periods=1).max()
    df['low_52w'] = df['low'].rolling(252, min_periods=1).min()
    df['distance_to_52w_high'] = (df['close'] - df['high_52w']) / df['high_52w']
    df['distance_to_52w_low'] = (df['close'] - df['low_52w']) / df['low_52w']
    
    # Close position in daily range
    range_size = df['high'] - df['low']
    df['close_range_position'] = np.where(
        range_size > 0,
        (df['close'] - df['low']) / range_size,
        0.5
    )
    
    return df


def compute_fibonacci_levels(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Fibonacci retracement levels."""
    
    if df.empty or len(df) < lookback:
        return df
    
    high = df['high'].rolling(lookback, min_periods=1).max()
    low = df['low'].rolling(lookback, min_periods=1).min()
    diff = high - low
    
    # Retracement levels
    df['fib_0.236'] = high - 0.236 * diff
    df['fib_0.382'] = high - 0.382 * diff
    df['fib_0.500'] = high - 0.500 * diff
    df['fib_0.618'] = high - 0.618 * diff
    
    # Distance to nearest fib level
    fib_levels = df[['fib_0.236', 'fib_0.382', 'fib_0.500', 'fib_0.618']].values
    current_price = df['close'].values[:, None]
    distances = np.abs(fib_levels - current_price)
    df['distance_to_nearest_fib'] = np.min(distances, axis=1) / df['close']
    
    # Confluence flag (multiple fib levels nearby)
    df['fib_confluence'] = (df['distance_to_nearest_fib'] < 0.01)
    
    return df


def compute_all_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical features."""
    
    df = compute_technical_indicators(df)
    df = compute_volume_features(df)
    df = compute_price_action_features(df)
    df = compute_fibonacci_levels(df)
    
    return df

