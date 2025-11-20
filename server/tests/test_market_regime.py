"""Tests for market regime detection"""

import pytest
import pandas as pd
import numpy as np

from src.models.market_regime import detect_market_regime, get_market_regime_name, get_market_regime_features


def test_detect_market_regime():
    """Test market regime detection."""
    
    # Create bullish scenario (upward trend, low volatility)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
    prices = 100 + np.cumsum(np.random.normal(0.1, 0.01, 100))  # Upward trend
    
    spy_prices = pd.DataFrame({
        'time': dates,
        'close': prices
    })
    
    regime = detect_market_regime(spy_prices)
    assert regime in [0, 1, 2]  # Bear, Sideways, Bull


def test_get_market_regime_name():
    """Test regime name retrieval."""
    
    assert get_market_regime_name(0) == 'Bear'
    assert get_market_regime_name(1) == 'Sideways'
    assert get_market_regime_name(2) == 'Bull'


def test_get_market_regime_features():
    """Test market regime features extraction."""
    
    dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
    prices = 100 + np.cumsum(np.random.normal(0.05, 0.02, 100))
    
    spy_prices = pd.DataFrame({
        'time': dates,
        'close': prices
    })
    
    features = get_market_regime_features(spy_prices)
    
    assert 'regime' in features
    assert 'regime_name' in features
    assert 'volatility' in features
    assert 'trend_strength' in features
    assert features['regime'] in [0, 1, 2]


