"""Macro Economic Feature Engineering

Adds macro economic indicators as features to stock data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from src.data.sources import FREDClient, VIXClient

logger = logging.getLogger(__name__)


def add_macro_features(
    features_df: pd.DataFrame,
    macro_data: Optional[pd.DataFrame] = None,
    vix_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Add macro economic features to stock features DataFrame.
    
    Args:
        features_df: Stock features DataFrame (must have 'time' column)
        macro_data: DataFrame with macro indicators (from FRED)
        vix_data: DataFrame with VIX data (must have 'time' and 'close' columns)
    
    Returns:
        DataFrame with macro features added
    """
    
    df = features_df.copy()
    
    # Ensure time column is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    elif 'date' in df.columns:
        df['time'] = pd.to_datetime(df['date'])
    else:
        logger.warning("No time/date column found, cannot add macro features")
        return df
    
    # Add VIX features
    if vix_data is not None and not vix_data.empty:
        if 'time' in vix_data.columns:
            vix_data['time'] = pd.to_datetime(vix_data['time'])
            # Merge VIX on time
            if 'close' in vix_data.columns:
                vix_features = vix_data[['time', 'close']].rename(columns={'close': 'vix'})
                df = df.merge(vix_features, on='time', how='left')
                
                # Add VIX-based features
                if 'vix' in df.columns:
                    # VIX percentile (rolling 252 days)
                    df['vix_percentile'] = df.groupby('ticker')['vix'].transform(
                        lambda x: x.rolling(252, min_periods=1).apply(
                            lambda y: (y.iloc[-1] > y.quantile(0.5)).astype(float) if len(y) > 0 else 0.5
                        )
                    )
                    # VIX change
                    df['vix_change'] = df.groupby('ticker')['vix'].pct_change()
                    logger.info("Added VIX features")
    
    # Add macro indicators
    if macro_data is not None and not macro_data.empty:
        if 'time' in macro_data.columns:
            macro_data['time'] = pd.to_datetime(macro_data['time'])
            
            # Merge macro indicators
            macro_cols = [col for col in macro_data.columns if col != 'time']
            df = df.merge(macro_data[['time'] + macro_cols], on='time', how='left')
            
            # Forward fill macro data (macro indicators are typically monthly/quarterly)
            for col in macro_cols:
                if col in df.columns:
                    df[col] = df.groupby('ticker')[col].ffill()
            
            logger.info(f"Added macro features: {macro_cols}")
    
    return df


def compute_macro_features_for_date(
    date: str,
    lookback_days: int = 60
) -> Dict:
    """
    Compute macro features for a specific date.
    
    Args:
        date: Date string (YYYY-MM-DD)
        lookback_days: Number of days to look back
    
    Returns:
        Dict with macro feature values
    """
    
    from datetime import datetime, timedelta
    
    end_date = datetime.strptime(date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=lookback_days)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    features = {}
    
    # Fetch VIX
    try:
        vix_client = VIXClient()
        vix_data = vix_client.get_vix_data(start_str, end_str)
        if not vix_data.empty and 'close' in vix_data.columns:
            features['vix'] = float(vix_data['close'].iloc[-1])
            features['vix_change'] = float(vix_data['close'].pct_change().iloc[-1]) if len(vix_data) > 1 else 0.0
    except Exception as e:
        logger.warning(f"Failed to fetch VIX data: {e}")
    
    # Fetch macro indicators
    try:
        fred_client = FREDClient()
        macro_data = fred_client.get_macro_indicators(start_str, end_str)
        if not macro_data.empty:
            # Get latest values
            for col in macro_data.columns:
                if col != 'time':
                    latest_val = macro_data[col].dropna().iloc[-1] if not macro_data[col].dropna().empty else None
                    if latest_val is not None:
                        features[col] = float(latest_val)
    except Exception as e:
        logger.warning(f"Failed to fetch macro data: {e}")
    
    return features


