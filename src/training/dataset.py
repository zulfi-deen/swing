"""TFT Dataset Preparation

Creates TimeSeriesDataSet for pytorch-forecasting Temporal Fusion Transformer.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

try:
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
except ImportError:
    TimeSeriesDataSet = None
    GroupNormalizer = None
    logging.warning("pytorch-forecasting not installed. TFT dataset creation will fail.")

logger = logging.getLogger(__name__)


def create_tft_dataset(
    df: pd.DataFrame,
    max_encoder_length: int = 60,
    max_prediction_length: int = 5,
    target_col: str = "return_5d",
    group_id_col: str = "ticker",
    time_col: str = "time"
) -> TimeSeriesDataSet:
    """
    Create TimeSeriesDataSet for TFT.
    
    Args:
        df: DataFrame with time series data
        max_encoder_length: Maximum encoder length (lookback)
        max_prediction_length: Maximum prediction length (forecast horizon)
        target_col: Name of target column
        group_id_col: Name of group ID column (e.g., ticker)
        time_col: Name of time column
    
    Returns:
        TimeSeriesDataSet ready for TFT training
    """
    
    if TimeSeriesDataSet is None:
        raise ImportError("pytorch-forecasting is required. Install with: pip install pytorch-forecasting")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Add time_idx (required by TFT) - days since minimum time
    df['time_idx'] = (df[time_col] - df[time_col].min()).dt.days
    
    # Ensure target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Ensure group_id exists
    if group_id_col not in df.columns:
        raise ValueError(f"Group ID column '{group_id_col}' not found in DataFrame")
    
    # Define feature columns based on what's available
    static_categoricals = []
    static_reals = []
    time_varying_known_categoricals = []
    time_varying_known_reals = []
    time_varying_unknown_categoricals = []
    time_varying_unknown_reals = []
    
    # Static categoricals
    if 'sector_id' in df.columns:
        static_categoricals.append('sector_id')
    if 'market_cap_bucket' in df.columns:
        static_categoricals.append('market_cap_bucket')
    
    # Time-varying known categoricals (known in future)
    if 'day_of_week' in df.columns:
        time_varying_known_categoricals.append('day_of_week')
    if 'month' in df.columns:
        time_varying_known_categoricals.append('month')
    
    # Time-varying known reals (known in future)
    if 'days_to_earnings' in df.columns:
        time_varying_known_reals.append('days_to_earnings')
    
    # Time-varying unknown reals (not known in future - features)
    feature_candidates = [
        'close', 'open', 'high', 'low', 'volume', 'vwap',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bbands_upper', 'bbands_middle', 'bbands_lower', 'bbands_pct',
        'atr_14', 'stoch_k', 'stoch_d', 'adx', 'mfi',
        'volume_z_score', 'vwap_deviation', 'gap_pct',
        'intraday_range_pct', 'distance_to_52w_high',
        'sentiment_score', 'news_intensity_score',
        'return_rank_5d', 'return_rank_20d',
        'sector_relative_strength', 'correlation_to_spy'
    ]
    
    for col in feature_candidates:
        if col in df.columns and col != target_col:
            time_varying_unknown_reals.append(col)
    
    # Create dataset
    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target_col,
        group_ids=[group_id_col],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals if static_categoricals else None,
        static_reals=static_reals if static_reals else None,
        time_varying_known_categoricals=time_varying_known_categoricals if time_varying_known_categoricals else None,
        time_varying_known_reals=time_varying_known_reals if time_varying_known_reals else None,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals if time_varying_unknown_categoricals else None,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=[group_id_col]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    logger.info(f"Created TimeSeriesDataSet with {len(df)} samples, {len(df[group_id_col].unique())} groups")
    logger.info(f"Encoder length: {max_encoder_length}, Prediction length: {max_prediction_length}")
    logger.info(f"Time-varying unknown reals: {len(time_varying_unknown_reals)} features")
    
    return dataset


def prepare_dataframe_for_tft(
    prices_df: pd.DataFrame,
    features_df: pd.DataFrame,
    tickers: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Prepare merged DataFrame for TFT dataset creation.
    
    Args:
        prices_df: Price data with OHLCV
        features_df: Feature data with technical indicators
        tickers: Optional list of tickers to filter
    
    Returns:
        Merged DataFrame ready for TFT
    """
    
    # Merge prices and features
    if 'time' in prices_df.columns and 'time' in features_df.columns:
        merge_cols = ['time', 'ticker']
    elif 'date' in prices_df.columns and 'date' in features_df.columns:
        merge_cols = ['date', 'ticker']
        prices_df = prices_df.rename(columns={'date': 'time'})
        features_df = features_df.rename(columns={'date': 'time'})
    else:
        raise ValueError("Cannot find time/date column in both DataFrames")
    
    df = prices_df.merge(features_df, on=merge_cols, how='inner')
    
    # Filter tickers if provided
    if tickers:
        df = df[df['ticker'].isin(tickers)]
    
    # Calculate target: 5-day forward return
    if 'return_5d' not in df.columns:
        df['return_5d'] = df.groupby('ticker')['close'].pct_change(5).shift(-5)
    
    # Calculate hit target (positive return = hit)
    if 'hit_target_long' not in df.columns:
        df['hit_target_long'] = (df['return_5d'] > 0).astype(float)
    
    # Add time-based features if missing
    if 'time' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['time'].dt.dayofweek
        if 'month' not in df.columns:
            df['month'] = df['time'].dt.month
    
    # Add sector_id if missing (placeholder - would come from tickers.py)
    if 'sector_id' not in df.columns:
        try:
            from src.utils.tickers import get_ticker_sector, SECTOR_TO_ID
            df['sector_id'] = df['ticker'].apply(
                lambda x: SECTOR_TO_ID.get(get_ticker_sector(x), 0)
            )
        except:
            logger.warning("Could not add sector_id, using default 0")
            df['sector_id'] = 0
    
    # Add liquidity_regime if missing (placeholder)
    if 'liquidity_regime' not in df.columns:
        # Would be computed from volume characteristics
        df['liquidity_regime'] = 1  # Default: mid
    
    # Add market_regime if missing (placeholder)
    if 'market_regime' not in df.columns:
        # Would be computed from market indicators
        df['market_regime'] = 1  # Default: sideways
    
    # Sort by ticker and time
    df = df.sort_values(['ticker', 'time']).reset_index(drop=True)
    
    # Remove rows with NaN targets
    df = df.dropna(subset=['return_5d', 'hit_target_long'])
    
    logger.info(f"Prepared DataFrame: {len(df)} rows, {df['ticker'].nunique()} tickers")
    
    return df


