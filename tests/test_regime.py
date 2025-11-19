"""Tests for Regime Detection"""

import pytest
import pandas as pd
import numpy as np
from src.models.regime import (
    detect_regime_features,
    get_regime_name,
    get_regime_trading_strategy,
    REGIME_TRENDING,
    REGIME_MEAN_REVERTING,
    REGIME_CHOPPY,
    REGIME_VOLATILE
)


def test_regime_names():
    """Test regime name mapping."""
    assert get_regime_name(REGIME_TRENDING) == 'Trending'
    assert get_regime_name(REGIME_MEAN_REVERTING) == 'MeanReverting'
    assert get_regime_name(REGIME_CHOPPY) == 'Choppy'
    assert get_regime_name(REGIME_VOLATILE) == 'Volatile'


def test_regime_detection():
    """Test regime detection on synthetic data."""
    # Create synthetic price data
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    prices = 100 + np.cumsum(np.random.randn(30) * 0.5)
    
    stock_data = pd.DataFrame({
        'time': dates,
        'close': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'volume': np.random.randint(1e6, 1e7, 30)
    })
    
    stock_chars = {'beta': 1.0, 'mean_reversion_strength': 0.5}
    
    regime = detect_regime_features(stock_data, stock_chars)
    
    assert regime in [REGIME_TRENDING, REGIME_MEAN_REVERTING, REGIME_CHOPPY, REGIME_VOLATILE]


def test_regime_trading_strategy():
    """Test trading strategy for each regime."""
    for regime_id in [REGIME_TRENDING, REGIME_MEAN_REVERTING, REGIME_CHOPPY, REGIME_VOLATILE]:
        strategy = get_regime_trading_strategy(regime_id)
        assert 'strategy' in strategy
        assert 'stop_loss_multiplier' in strategy
        assert 'position_size_multiplier' in strategy


