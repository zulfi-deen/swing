"""Technical indicator features"""

import pandas as pd
import numpy as np
from typing import Optional

try:  # pragma: no cover - optional dependency
    import talib as ta

    TA_LIB_AVAILABLE = True
except Exception:  # pragma: no cover - TA-Lib not installed
    ta = None
    TA_LIB_AVAILABLE = False


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators using TA-Lib."""
    
    if df.empty or len(df) < 50:
        return df
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    if TA_LIB_AVAILABLE:
        df['sma_20'] = ta.SMA(df['close'], timeperiod=20)
        df['sma_50'] = ta.SMA(df['close'], timeperiod=50)
        df['ema_12'] = ta.EMA(df['close'], timeperiod=12)
        df['ema_26'] = ta.EMA(df['close'], timeperiod=26)
        df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        df['rsi_14'] = ta.RSI(df['close'], timeperiod=14)
        df['bbands_upper'], df['bbands_middle'], df['bbands_lower'] = ta.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        df['atr_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['stoch_k'], df['stoch_d'] = ta.STOCH(
            df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3
        )
        df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['mfi'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    else:
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        rolling_mean = df['close'].rolling(20).mean()
        rolling_std = df['close'].rolling(20).std()
        df['bbands_upper'] = rolling_mean + 2 * rolling_std
        df['bbands_middle'] = rolling_mean
        df['bbands_lower'] = rolling_mean - 2 * rolling_std
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))),
        )
        df['atr_14'] = tr.rolling(14).mean()
        df['stoch_k'] = 100 * (df['close'] - df['low'].rolling(14).min()) / (
            df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-9
        )
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        df['adx'] = (abs(df['close'].diff()) / (df['close'] + 1e-9)).rolling(14).mean()
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0.0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0.0)
        money_ratio = positive_flow.rolling(14).sum() / (negative_flow.rolling(14).sum() + 1e-9)
        df['mfi'] = 100 - (100 / (1 + money_ratio))
    
    df['bbands_pct'] = (df['close'] - df['bbands_lower']) / (
        df['bbands_upper'] - df['bbands_lower']
    )
    
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
    
    if TA_LIB_AVAILABLE:
        df['obv'] = ta.OBV(df['close'], df['volume'])
    else:
        df['obv'] = (np.sign(df['close'].diff()).fillna(0) * df['volume']).cumsum()
    
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

