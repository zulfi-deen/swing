"""Tests for RL state building with options features"""

import pytest
import pandas as pd
import numpy as np
from src.data.rl_state_builder import RLStateBuilder


def test_rl_state_builder_with_options():
    """Test RL state builder with options features."""
    
    config = {
        'rl_portfolio': {
            'options_enabled': True
        }
    }
    
    state_builder = RLStateBuilder(config)
    
    # Create sample twin predictions
    twin_predictions = {
        'AAPL': {
            'expected_return': 0.02,
            'hit_prob': 0.65,
            'volatility': 0.03,
            'regime': 0,
            'idiosyncratic_alpha': 0.01,
            'quantile_10': -0.05,
            'quantile_90': 0.05
        },
        'MSFT': {
            'expected_return': 0.015,
            'hit_prob': 0.60,
            'volatility': 0.025,
            'regime': 1,
            'idiosyncratic_alpha': 0.005,
            'quantile_10': -0.04,
            'quantile_90': 0.04
        }
    }
    
    # Create sample features
    features_df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'rsi_14': [55.0, 45.0],
        'macd_signal': [0.5, -0.3],
        'volume_z_score': [1.2, -0.5]
    })
    
    # Create sample portfolio state
    portfolio_state = {
        'cash': 0.35,
        'num_positions': 2,
        'positions': {
            'AAPL': {'size': 0.1, 'entry_price': 150.0, 'days_held': 3}
        },
        'sector_exposure': {'Technology': 0.15},
        'portfolio_value': 100000.0,
        'peak_value': 105000.0
    }
    
    # Create sample macro context
    macro_context = {
        'vix': 20.0,
        'spy_return_5d': 0.01,
        'treasury_10y': 4.0,
        'market_regime': 'bull'
    }
    
    # Create sample options features
    options_features = {
        'AAPL': {
            'trend_signal': 0.8,
            'sentiment_signal': 0.6,
            'gamma_signal': 1,
            'pcr_zscore': -1.5,
            'pcr_extreme_bullish': True,
            'pcr_extreme_bearish': False,
            'max_pain_distance_pct': 0.02,
            'iv_percentile': 0.75,
            'net_delta': 0.3
        },
        'MSFT': {
            'trend_signal': 0.5,
            'sentiment_signal': 0.4,
            'gamma_signal': 0,
            'pcr_zscore': 0.5,
            'pcr_extreme_bullish': False,
            'pcr_extreme_bearish': False,
            'max_pain_distance_pct': -0.01,
            'iv_percentile': 0.5,
            'net_delta': 0.1
        }
    }
    
    # Build state
    state = state_builder.build_state(
        twin_predictions=twin_predictions,
        features_df=features_df,
        portfolio_state=portfolio_state,
        macro_context=macro_context,
        prices_df=None,
        date=None,
        options_features=options_features
    )
    
    # Verify state structure
    assert 'twin_predictions' in state
    assert 'features' in state
    assert 'portfolio' in state
    assert 'macro' in state
    assert 'tickers' in state
    assert 'options_features' in state
    
    # Verify options features are included
    assert len(state['options_features']) == 2
    assert 'AAPL' in state['options_features']
    assert 'MSFT' in state['options_features']
    
    # Verify options feature values
    aapl_options = state['options_features']['AAPL']
    assert aapl_options['trend_signal'] == 0.8
    assert aapl_options['pcr_extreme_bullish'] == True


def test_rl_state_builder_without_options():
    """Test RL state builder without options features."""
    
    config = {
        'rl_portfolio': {
            'options_enabled': False
        }
    }
    
    state_builder = RLStateBuilder(config)
    
    # Create minimal sample data
    twin_predictions = {
        'AAPL': {
            'expected_return': 0.02,
            'hit_prob': 0.65,
            'volatility': 0.03,
            'regime': 0,
            'idiosyncratic_alpha': 0.01,
            'quantile_10': -0.05,
            'quantile_90': 0.05
        }
    }
    
    features_df = pd.DataFrame({
        'ticker': ['AAPL'],
        'rsi_14': [55.0]
    })
    
    portfolio_state = {
        'cash': 0.5,
        'num_positions': 0,
        'positions': {},
        'sector_exposure': {},
        'portfolio_value': 100000.0,
        'peak_value': 100000.0
    }
    
    macro_context = {
        'vix': 20.0,
        'spy_return_5d': 0.0,
        'treasury_10y': 4.0,
        'market_regime': 'bull'
    }
    
    # Build state without options
    state = state_builder.build_state(
        twin_predictions=twin_predictions,
        features_df=features_df,
        portfolio_state=portfolio_state,
        macro_context=macro_context,
        prices_df=None,
        date=None,
        options_features=None
    )
    
    # Verify state structure
    assert 'options_features' in state
    assert state['options_features'] == {}  # Empty dict when no options


if __name__ == "__main__":
    pytest.main([__file__])

