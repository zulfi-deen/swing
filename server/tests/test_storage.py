"""Tests for data storage round-trip functionality"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from src.data.storage import save_to_local, load_from_local
from src.utils.config import load_config


def test_storage_round_trip_dataframe():
    """Test saving and loading DataFrame."""
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create sample DataFrame
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'close': [150.0, 300.0, 2500.0],
            'volume': [1000000, 2000000, 500000]
        })
        
        # Create config with temp directory
        config = {
            'storage': {
                'local': {
                    'data_dir': temp_dir
                }
            }
        }
        
        # Save
        save_to_local(df, "test/prices", config)
        
        # Load
        loaded_df = load_from_local("test/prices", config)
        
        # Verify
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == 3
        assert list(loaded_df['ticker']) == ['AAPL', 'MSFT', 'GOOGL']
        assert loaded_df['close'].iloc[0] == 150.0
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_storage_round_trip_dict():
    """Test saving and loading dict."""
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create sample dict
        data = {
            'AAPL': [
                {'strike': 150, 'oi': 1000},
                {'strike': 155, 'oi': 2000}
            ],
            'MSFT': [
                {'strike': 300, 'oi': 500}
            ]
        }
        
        # Create config with temp directory
        config = {
            'storage': {
                'local': {
                    'data_dir': temp_dir
                }
            }
        }
        
        # Save
        save_to_local(data, "test/options_raw", config)
        
        # Load
        loaded_data = load_from_local("test/options_raw", config)
        
        # Verify
        assert isinstance(loaded_data, dict)
        assert 'AAPL' in loaded_data
        assert len(loaded_data['AAPL']) == 2
        assert loaded_data['AAPL'][0]['strike'] == 150
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_storage_round_trip_options_data():
    """Test saving and loading options data (dict of DataFrames converted to records)."""
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create sample options data (dict of DataFrames)
        options_data = {
            'AAPL': pd.DataFrame({
                'strike_price': [150, 155],
                'option_type': ['call', 'put'],
                'open_interest': [1000, 2000],
                'volume': [500, 600]
            }),
            'MSFT': pd.DataFrame({
                'strike_price': [300],
                'option_type': ['call'],
                'open_interest': [500],
                'volume': [200]
            })
        }
        
        # Convert to serializable format (as done in ingestion.py)
        options_data_serializable = {}
        for ticker, chain_df in options_data.items():
            if isinstance(chain_df, pd.DataFrame):
                if not chain_df.empty:
                    options_data_serializable[ticker] = chain_df.to_dict('records')
                else:
                    options_data_serializable[ticker] = []
        
        # Create config with temp directory
        config = {
            'storage': {
                'local': {
                    'data_dir': temp_dir
                }
            }
        }
        
        # Save
        save_to_local(options_data_serializable, "test/options_raw", config)
        
        # Load
        loaded_data = load_from_local("test/options_raw", config)
        
        # Verify
        assert isinstance(loaded_data, dict)
        assert 'AAPL' in loaded_data
        assert isinstance(loaded_data['AAPL'], list)
        assert len(loaded_data['AAPL']) == 2
        assert loaded_data['AAPL'][0]['strike_price'] == 150
        
        # Convert back to DataFrame (as done in orchestrator.py)
        if isinstance(loaded_data['AAPL'], list) and len(loaded_data['AAPL']) > 0:
            df = pd.DataFrame(loaded_data['AAPL'])
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert 'strike_price' in df.columns
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])

