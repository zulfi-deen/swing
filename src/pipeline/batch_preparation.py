"""Batch Preparation for Model Inference

Prepares batches from DataFrames for digital twin inference.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional
import logging

from src.utils.tickers import get_ticker_sector, SECTOR_TO_ID
from src.features.normalization import FeatureNormalizer

logger = logging.getLogger(__name__)


def prepare_batch_for_twin(
    features_df: pd.DataFrame,
    ticker: str,
    normalizer: Optional[FeatureNormalizer] = None,
    config: Optional[Dict] = None
) -> Dict:
    """
    Prepare batch for digital twin inference.
    
    Args:
        features_df: DataFrame with features (must have 'ticker' and 'time' columns)
        ticker: Ticker symbol to prepare batch for
        normalizer: Optional FeatureNormalizer (if None, features assumed already normalized)
        config: Configuration dict with model parameters
    
    Returns:
        Dict with batch data compatible with foundation model
    """
    
    if config is None:
        config = {
            'models': {
                'foundation': {
                    'tft': {
                        'max_encoder_length': 60
                    }
                }
            }
        }
    
    # Filter data for this ticker
    ticker_data = features_df[features_df['ticker'] == ticker].copy()
    
    if ticker_data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
    # Sort by time
    if 'time' in ticker_data.columns:
        ticker_data = ticker_data.sort_values('time')
    elif 'date' in ticker_data.columns:
        ticker_data = ticker_data.sort_values('date')
        ticker_data['time'] = pd.to_datetime(ticker_data['date'])
    else:
        raise ValueError("DataFrame must have 'time' or 'date' column")
    
    # Normalize features if normalizer provided
    if normalizer is not None:
        ticker_data = normalizer.transform(ticker_data)
    
    # Extract last N days for encoder
    max_encoder_length = config.get('models', {}).get('foundation', {}).get('tft', {}).get('max_encoder_length', 60)
    encoder_data = ticker_data.tail(max_encoder_length).copy()
    
    if len(encoder_data) < max_encoder_length:
        # Pad with zeros if insufficient data
        padding_needed = max_encoder_length - len(encoder_data)
        padding_df = pd.DataFrame(0, index=range(padding_needed), columns=encoder_data.columns)
        encoder_data = pd.concat([padding_df, encoder_data], ignore_index=True)
    
    # Get sector ID
    sector = get_ticker_sector(ticker)
    sector_id = SECTOR_TO_ID.get(sector, 0) if sector else 0
    
    # Get liquidity regime (default to mid if not present)
    if 'liquidity_regime' in encoder_data.columns:
        liquidity_regime = int(encoder_data['liquidity_regime'].iloc[-1]) if not encoder_data['liquidity_regime'].isna().iloc[-1] else 1
    else:
        liquidity_regime = 1  # Default: mid
    
    # Get market regime (default to sideways if not present)
    if 'market_regime' in encoder_data.columns:
        market_regime = int(encoder_data['market_regime'].iloc[-1]) if not encoder_data['market_regime'].isna().iloc[-1] else 1
    else:
        market_regime = 1  # Default: sideways
    
    # Add time_idx if not present
    if 'time_idx' not in encoder_data.columns:
        encoder_data['time_idx'] = range(len(encoder_data))
    
    # Prepare time-varying continuous features
    time_varying_features = [
        'close', 'open', 'high', 'low', 'volume', 'vwap',
        'rsi_14', 'macd', 'macd_signal', 'atr_14',
        'stoch_k', 'adx', 'mfi',
        'volume_z_score', 'sentiment_score',
        'return_rank_5d', 'sector_relative_strength'
    ]
    
    available_features = [f for f in time_varying_features if f in encoder_data.columns]
    
    if not available_features:
        logger.warning(f"No time-varying features found for {ticker}, using placeholder")
        encoder_cont = torch.zeros(max_encoder_length, 10, dtype=torch.float32)
    else:
        encoder_cont = torch.tensor(
            encoder_data[available_features].fillna(0).values,
            dtype=torch.float32
        )
        # Ensure correct shape: (seq_len, num_features)
        if encoder_cont.dim() == 1:
            encoder_cont = encoder_cont.unsqueeze(0)
    
    # Prepare time-varying categorical features
    time_varying_cat_features = []
    if 'day_of_week' in encoder_data.columns:
        day_of_week = torch.tensor(encoder_data['day_of_week'].fillna(0).astype(int).values, dtype=torch.long)
        time_varying_cat_features.append(day_of_week)
    if 'month' in encoder_data.columns:
        month = torch.tensor(encoder_data['month'].fillna(1).astype(int).values, dtype=torch.long)
        time_varying_cat_features.append(month)
    
    # Prepare batch dict
    batch = {
        'time_idx': torch.tensor(encoder_data['time_idx'].values, dtype=torch.long),
        'group_ids': torch.zeros(len(encoder_data), dtype=torch.long),  # ticker_id (would be mapped in real scenario)
        'sector_id': torch.tensor([sector_id] * len(encoder_data), dtype=torch.long),
        'liquidity_regime': torch.tensor([liquidity_regime] * len(encoder_data), dtype=torch.long),
        'market_regime': torch.tensor([market_regime] * len(encoder_data), dtype=torch.long),
        'encoder_cont': encoder_cont.unsqueeze(0) if encoder_cont.dim() == 2 else encoder_cont,  # Add batch dim if needed
    }
    
    # Add categorical features if available
    if time_varying_cat_features:
        batch['encoder_cat'] = torch.stack(time_varying_cat_features, dim=0).unsqueeze(0) if len(time_varying_cat_features) > 0 else None
    
    # Add static features
    batch['static_cat'] = torch.tensor([[sector_id]], dtype=torch.long)
    batch['static_cont'] = torch.zeros(1, 0, dtype=torch.float32)  # No static continuous features
    
    return batch


def prepare_batches_for_multiple_tickers(
    features_df: pd.DataFrame,
    tickers: List[str],
    normalizer: Optional[FeatureNormalizer] = None,
    config: Optional[Dict] = None
) -> Dict[str, Dict]:
    """
    Prepare batches for multiple tickers.
    
    Args:
        features_df: DataFrame with features
        tickers: List of ticker symbols
        normalizer: Optional FeatureNormalizer
        config: Configuration dict
    
    Returns:
        Dict mapping ticker to batch dict
    """
    
    batches = {}
    
    for ticker in tickers:
        try:
            batch = prepare_batch_for_twin(features_df, ticker, normalizer, config)
            batches[ticker] = batch
        except Exception as e:
            logger.error(f"Failed to prepare batch for {ticker}: {e}")
            continue
    
    logger.info(f"Prepared batches for {len(batches)}/{len(tickers)} tickers")
    
    return batches


