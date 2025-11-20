"""Tests for Digital Twin Model"""

import pytest
import torch
from src.models.foundation import StockTwinFoundation
from src.models.digital_twin import StockDigitalTwin


def test_digital_twin_initialization():
    """Test digital twin can be initialized."""
    foundation_config = {
        'tft': {'hidden_size': 256},
        'gnn': {'hidden_dim': 64, 'num_heads': 8, 'num_layers': 2, 'dropout': 0.1},
        'embeddings': {'sector_dim': 32, 'liquidity_dim': 16, 'market_regime_dim': 16},
        'backbone': {'hidden_dims': [256, 128], 'dropout': [0.15, 0.1]}
    }
    
    foundation = StockTwinFoundation(foundation_config)
    foundation.freeze()
    
    stock_chars = {
        'ticker': 'AAPL',
        'sector': 'Technology',
        'beta': 1.0,
        'mean_reversion_strength': 0.5
    }
    
    twin_config = {
        'adapter_rank': 16,
        'stock_embedding_dim': 64,
        'regime_embedding_dim': 32
    }
    
    twin = StockDigitalTwin(foundation, 'AAPL', stock_chars, twin_config)
    assert twin.ticker == 'AAPL'
    assert twin.embedding_dim == 128


def test_digital_twin_forward():
    """Test digital twin forward pass (simplified)."""
    foundation_config = {
        'tft': {'hidden_size': 256},
        'gnn': {'hidden_dim': 64, 'num_heads': 8, 'num_layers': 2, 'dropout': 0.1},
        'embeddings': {'sector_dim': 32, 'liquidity_dim': 16, 'market_regime_dim': 16},
        'backbone': {'hidden_dims': [256, 128], 'dropout': [0.15, 0.1]}
    }
    
    foundation = StockTwinFoundation(foundation_config)
    foundation.freeze()
    
    stock_chars = {'ticker': 'AAPL', 'beta': 1.0}
    twin = StockDigitalTwin(foundation, 'AAPL', stock_chars)
    
    # Note: Full forward pass requires TFT to be initialized, which needs a dataset
    # This is a placeholder test
    assert twin is not None


