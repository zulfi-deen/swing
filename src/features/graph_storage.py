"""Parquet-based graph storage for correlation graphs

Replaces Neo4j with lightweight parquet file storage for daily correlation graphs.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_graph_storage_dir(config: Optional[Dict] = None) -> Path:
    """
    Get graph storage directory from config or default.
    
    Args:
        config: Configuration dict
    
    Returns:
        Path to graph storage directory
    """
    if config:
        storage_dir = config.get('storage', {}).get('local', {}).get('data_dir', 'data/')
    else:
        storage_dir = 'data/'
    
    graph_dir = Path(storage_dir) / 'graphs'
    return graph_dir


def save_correlation_graph(
    correlations: Dict[Tuple[str, str], float],
    date: str,
    ticker_to_idx: Dict[str, int],
    lookback_days: int = 30,
    threshold: float = 0.3,
    config: Optional[Dict] = None
) -> bool:
    """
    Save correlation graph edges to parquet file.
    
    Args:
        correlations: Dict mapping (ticker1, ticker2) -> correlation value
        date: Date string (YYYY-MM-DD)
        ticker_to_idx: Mapping of ticker to node index
        lookback_days: Number of days used for correlation
        threshold: Correlation threshold used
        config: Configuration dict
    
    Returns:
        True if successful, False otherwise
    """
    try:
        graph_dir = get_graph_storage_dir(config)
        correlations_dir = graph_dir / 'correlations'
        metadata_dir = graph_dir / 'metadata'
        
        # Create directories if they don't exist
        correlations_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert correlations to DataFrame
        edges = []
        for (ticker1, ticker2), corr_value in correlations.items():
            edges.append({
                'ticker1': ticker1,
                'ticker2': ticker2,
                'correlation': corr_value,
                'abs_correlation': abs(corr_value)
            })
        
        if not edges:
            logger.warning(f"No edges to save for {date}")
            return False
        
        df = pd.DataFrame(edges)
        
        # Save correlations to parquet
        correlation_file = correlations_dir / f'correlations_{date}.parquet'
        df.to_parquet(correlation_file, index=False, engine='pyarrow')
        
        # Save metadata
        metadata = {
            'date': date,
            'lookback_days': lookback_days,
            'threshold': threshold,
            'num_edges': len(edges),
            'num_nodes': len(ticker_to_idx),
            'computation_time': datetime.now().isoformat()
        }
        
        metadata_file = metadata_dir / f'metadata_{date}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved correlation graph for {date}: {len(edges)} edges, {len(ticker_to_idx)} nodes")
        return True
        
    except Exception as e:
        logger.error(f"Error saving correlation graph for {date}: {e}")
        return False


def load_correlation_graph(
    date: str,
    tickers: Optional[List[str]] = None,
    threshold: float = 0.3,
    config: Optional[Dict] = None
) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    Load correlation graph from parquet file.
    
    Args:
        date: Date string (YYYY-MM-DD)
        tickers: Optional list of tickers to filter
        threshold: Minimum absolute correlation to include
        config: Configuration dict
    
    Returns:
        Tuple of (edges DataFrame, ticker_to_idx mapping) or (None, None) if not found
    """
    try:
        graph_dir = get_graph_storage_dir(config)
        correlation_file = graph_dir / 'correlations' / f'correlations_{date}.parquet'
        
        if not correlation_file.exists():
            logger.debug(f"Correlation graph file not found for {date}: {correlation_file}")
            return None, None
        
        # Load edges
        df = pd.read_parquet(correlation_file)
        
        # Filter by threshold
        df = df[df['abs_correlation'] >= threshold]
        
        # Filter by tickers if provided
        if tickers:
            df = df[df['ticker1'].isin(tickers) & df['ticker2'].isin(tickers)]
        
        # Build ticker_to_idx mapping from unique tickers
        all_tickers = sorted(set(df['ticker1'].unique().tolist() + df['ticker2'].unique().tolist()))
        ticker_to_idx = {ticker: i for i, ticker in enumerate(all_tickers)}
        
        logger.info(f"Loaded correlation graph for {date}: {len(df)} edges, {len(ticker_to_idx)} nodes")
        return df, ticker_to_idx
        
    except Exception as e:
        logger.error(f"Error loading correlation graph for {date}: {e}")
        return None, None


def get_graph_metadata(
    date: str,
    config: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Get metadata for a correlation graph.
    
    Args:
        date: Date string (YYYY-MM-DD)
        config: Configuration dict
    
    Returns:
        Metadata dict or None if not found
    """
    try:
        graph_dir = get_graph_storage_dir(config)
        metadata_file = graph_dir / 'metadata' / f'metadata_{date}.json'
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error loading graph metadata for {date}: {e}")
        return None


def compute_and_cache_graph(
    prices: pd.DataFrame,
    date: str,
    threshold: float = 0.3,
    lookback_days: int = 30,
    sector_map: Optional[Dict[str, str]] = None,
    config: Optional[Dict] = None
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, int]]:
    """
    Compute correlation graph and cache to parquet.
    
    Args:
        prices: Historical price data
        date: Current date
        threshold: Correlation threshold
        lookback_days: Number of days to look back
        sector_map: Optional mapping of ticker -> sector
        config: Configuration dict
    
    Returns:
        Tuple of (correlations dict, ticker_to_idx mapping)
    """
    from src.features.graph import compute_correlation_matrix
    
    # Compute correlation matrix
    corr_matrix, tickers = compute_correlation_matrix(prices, date, lookback_days)
    
    if corr_matrix.empty or not tickers:
        logger.warning(f"Could not compute correlation matrix for {date}")
        return {}, {}
    
    # Build correlations dict
    ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
    correlations = {}
    
    for i, ticker_i in enumerate(tickers):
        for j, ticker_j in enumerate(tickers):
            if i < j:  # Only store once per pair
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    correlations[(ticker_i, ticker_j)] = float(corr_value)
    
    # Save to parquet
    save_correlation_graph(
        correlations,
        date,
        ticker_to_idx,
        lookback_days,
        threshold,
        config
    )
    
    return correlations, ticker_to_idx


def get_or_build_graph(
    prices: pd.DataFrame,
    date: str,
    threshold: float = 0.3,
    lookback_days: int = 30,
    sector_map: Optional[Dict[str, str]] = None,
    config: Optional[Dict] = None,
    force_recompute: bool = False
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, int]]:
    """
    Get graph from cache or compute fresh.
    
    Args:
        prices: Historical price data (used if cache miss)
        date: Current date
        threshold: Correlation threshold
        lookback_days: Number of days to look back
        sector_map: Optional mapping of ticker -> sector
        config: Configuration dict
        force_recompute: If True, recompute even if cache exists
    
    Returns:
        Tuple of (correlations dict, ticker_to_idx mapping)
    """
    # Try to load from cache first
    if not force_recompute:
        edges_df, ticker_to_idx = load_correlation_graph(date, threshold=threshold, config=config)
        
        if edges_df is not None and ticker_to_idx is not None:
            # Convert DataFrame back to correlations dict
            correlations = {}
            for _, row in edges_df.iterrows():
                correlations[(row['ticker1'], row['ticker2'])] = row['correlation']
            
            logger.info(f"Loaded graph from cache for {date}")
            return correlations, ticker_to_idx
    
    # Cache miss or force recompute - compute fresh
    logger.info(f"Computing fresh graph for {date}")
    return compute_and_cache_graph(
        prices,
        date,
        threshold,
        lookback_days,
        sector_map,
        config
    )

