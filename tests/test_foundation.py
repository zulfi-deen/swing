"""Tests for Foundation Model"""

import pytest
import torch
from src.models.foundation import StockTwinFoundation


def test_foundation_model_initialization():
    """Test foundation model can be initialized."""
    config = {
        'tft': {
            'hidden_size': 256,
            'lstm_layers': 2,
            'attention_head_size': 8,
            'dropout': 0.1,
            'max_encoder_length': 60,
            'max_prediction_length': 5,
        },
        'gnn': {
            'hidden_dim': 64,
            'num_heads': 8,
            'num_layers': 2,
            'dropout': 0.1,
        },
        'embeddings': {
            'sector_dim': 32,
            'liquidity_dim': 16,
            'market_regime_dim': 16,
        },
        'backbone': {
            'hidden_dims': [256, 128],
            'dropout': [0.15, 0.1],
        }
    }
    
    model = StockTwinFoundation(config)
    assert model.embedding_dim == 128
    assert model.tft_hidden_size == 256


def test_foundation_freeze():
    """Test foundation model can be frozen."""
    config = {
        'tft': {'hidden_size': 256},
        'gnn': {'hidden_dim': 64, 'num_heads': 8, 'num_layers': 2, 'dropout': 0.1},
        'embeddings': {'sector_dim': 32, 'liquidity_dim': 16, 'market_regime_dim': 16},
        'backbone': {'hidden_dims': [256, 128], 'dropout': [0.15, 0.1]}
    }
    
    model = StockTwinFoundation(config)
    model.freeze()
    
    for param in model.parameters():
        assert not param.requires_grad


