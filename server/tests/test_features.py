"""Tests for feature engineering"""

import pytest
import pandas as pd
import numpy as np
from src.features.technical import compute_all_technical_features


def test_technical_indicators():
    """Test technical indicator computation."""
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'time': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + 1,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - 1,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000000, 10000000, 100),
        'vwap': 100 + np.cumsum(np.random.randn(100) * 0.5)
    })
    
    # Compute features
    result = compute_all_technical_features(df)
    
    # Check that features were added
    assert 'rsi_14' in result.columns
    assert 'macd' in result.columns
    assert 'bbands_pct' in result.columns
    assert 'atr_14' in result.columns


if __name__ == "__main__":
    pytest.main([__file__])

