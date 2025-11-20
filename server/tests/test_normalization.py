"""Tests for feature normalization"""

import pytest
import pandas as pd
import numpy as np

from src.features.normalization import FeatureNormalizer


def test_feature_normalizer_fit_transform():
    """Test FeatureNormalizer fit and transform."""
    
    # Create sample data
    df = pd.DataFrame({
        'close': np.random.randn(100) * 10 + 100,
        'volume': np.random.randint(1000000, 10000000, 100),
        'rsi_14': np.random.uniform(0, 100, 100),
        'return_5d': np.random.randn(100) * 0.02,
        'sentiment_score': np.random.uniform(-1, 1, 100)
    })
    
    normalizer = FeatureNormalizer()
    df_scaled = normalizer.fit_transform(df)
    
    assert len(df_scaled) == len(df)
    assert 'close' in df_scaled.columns
    assert normalizer.fitted


def test_feature_normalizer_save_load(tmp_path):
    """Test saving and loading normalizer."""
    
    df = pd.DataFrame({
        'close': np.random.randn(100) * 10 + 100,
        'volume': np.random.randint(1000000, 10000000, 100),
        'rsi_14': np.random.uniform(0, 100, 100)
    })
    
    normalizer = FeatureNormalizer()
    normalizer.fit(df)
    
    save_path = tmp_path / "normalizer.pkl"
    normalizer.save(str(save_path))
    
    # Load
    normalizer2 = FeatureNormalizer()
    normalizer2.load(str(save_path))
    
    assert normalizer2.fitted
    assert len(normalizer2.scalers) > 0


