"""Tests for Twin Manager"""

import pytest
from src.models.twin_manager import TwinManager
from src.models.foundation import StockTwinFoundation


def test_twin_manager_initialization():
    """Test TwinManager can be initialized."""
    foundation_config = {
        'tft': {'hidden_size': 256},
        'gnn': {'hidden_dim': 64, 'num_heads': 8, 'num_layers': 2, 'dropout': 0.1},
        'embeddings': {'sector_dim': 32, 'liquidity_dim': 16, 'market_regime_dim': 16},
        'backbone': {'hidden_dims': [256, 128], 'dropout': [0.15, 0.1]}
    }
    
    foundation = StockTwinFoundation(foundation_config)
    foundation.freeze()
    
    pilot_tickers = ['AAPL', 'MSFT']
    
    config = {
        'models': {
            'twins': {
                'pilot_tickers': pilot_tickers,
                'checkpoint_dir': 'models/twins/',
                'adapter_rank': 16,
                'stock_embedding_dim': 64,
                'regime_embedding_dim': 32
            }
        }
    }
    
    manager = TwinManager(
        foundation_model=foundation,
        pilot_tickers=pilot_tickers,
        config=config
    )
    
    assert len(manager.pilot_tickers) == 2
    assert manager.foundation is not None


def test_twin_manager_list_twins():
    """Test TwinManager can list twins."""
    foundation_config = {
        'tft': {'hidden_size': 256},
        'gnn': {'hidden_dim': 64, 'num_heads': 8, 'num_layers': 2, 'dropout': 0.1},
        'embeddings': {'sector_dim': 32, 'liquidity_dim': 16, 'market_regime_dim': 16},
        'backbone': {'hidden_dims': [256, 128], 'dropout': [0.15, 0.1]}
    }
    
    foundation = StockTwinFoundation(foundation_config)
    foundation.freeze()
    
    pilot_tickers = ['AAPL']
    config = {
        'models': {
            'twins': {
                'pilot_tickers': pilot_tickers,
                'checkpoint_dir': 'models/twins/'
            }
        }
    }
    
    manager = TwinManager(foundation, pilot_tickers, config)
    twins_list = manager.list_twins()
    
    assert isinstance(twins_list, list)
    assert len(twins_list) == 1


