"""Tests for batch preparation"""

import pytest
import pandas as pd
import numpy as np
import torch

from src.pipeline.batch_preparation import prepare_batch_for_twin, prepare_batches_for_multiple_tickers


def test_prepare_batch_for_twin():
    """Test batch preparation for a single ticker."""
    
    # Create sample features
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='B')
    data = []
    for date in dates:
        data.append({
            'time': date,
            'ticker': 'AAPL',
            'close': 100.0 + np.random.randn() * 2,
            'volume': 1000000,
            'rsi_14': 50.0,
            'macd': 0.5,
            'atr_14': 2.0,
            'sentiment_score': 0.1,
            'return_rank_5d': 100,
            'liquidity_regime': 1,
            'market_regime': 1
        })
    
    features_df = pd.DataFrame(data)
    
    config = {
        'models': {
            'foundation': {
                'tft': {
                    'max_encoder_length': 20
                }
            }
        }
    }
    
    batch = prepare_batch_for_twin(features_df, 'AAPL', normalizer=None, config=config)
    
    assert 'time_idx' in batch
    assert 'sector_id' in batch
    assert 'encoder_cont' in batch
    assert isinstance(batch['encoder_cont'], torch.Tensor)


def test_prepare_batches_for_multiple_tickers():
    """Test batch preparation for multiple tickers."""
    
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='B')
    tickers = ['AAPL', 'MSFT']
    
    data = []
    for ticker in tickers:
        for date in dates:
            data.append({
                'time': date,
                'ticker': ticker,
                'close': 100.0,
                'volume': 1000000,
                'rsi_14': 50.0,
                'liquidity_regime': 1,
                'market_regime': 1
            })
    
    features_df = pd.DataFrame(data)
    
    batches = prepare_batches_for_multiple_tickers(features_df, tickers, normalizer=None, config={})
    
    assert len(batches) == len(tickers)
    assert 'AAPL' in batches
    assert 'MSFT' in batches


