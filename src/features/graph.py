"""Graph construction for GNN with parquet-based storage"""

import torch
from torch_geometric.data import Data, HeteroData
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
import logging

from src.utils.tickers import TICKER_TO_SECTOR, SECTOR_TO_ID
from src.features.graph_storage import get_or_build_graph

logger = logging.getLogger(__name__)


def compute_correlation_matrix(
    prices: pd.DataFrame,
    date: str,
    lookback_days: int = 30
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute correlation matrix from price data.
    
    Args:
        prices: Historical price data
        date: Current date
        lookback_days: Number of days to look back
    
    Returns:
        Tuple of (correlation_matrix, ticker_list)
    """
    
    # Get last N days of returns
    if 'date' not in prices.columns and 'time' in prices.columns:
        prices['date'] = pd.to_datetime(prices['time']).dt.date
    
    date_col = 'date' if 'date' in prices.columns else None
    
    if date_col:
        lookback_start = pd.to_datetime(date) - pd.Timedelta(days=lookback_days)
        recent_prices = prices[pd.to_datetime(prices[date_col]) >= lookback_start]
    else:
        recent_prices = prices.tail(lookback_days)
    
    if recent_prices.empty:
        return pd.DataFrame(), []
    
    # Pivot to ticker columns
    if date_col:
        returns = recent_prices.pivot_table(
            index=date_col,
            columns='ticker',
            values='close'
        ).pct_change().dropna()
    else:
        returns = recent_prices.pivot(columns='ticker', values='close').pct_change().dropna()
    
    if returns.empty or len(returns.columns) < 2:
        return pd.DataFrame(), []
    
    # Compute correlation matrix
    corr_matrix = returns.corr()
    tickers = list(corr_matrix.columns)
    
    return corr_matrix, tickers


def build_correlation_graph(
    prices: pd.DataFrame,
    date: str,
    threshold: float = 0.3,
    cache_to_parquet: bool = True,
    config: Optional[Dict] = None,
    sector_map: Optional[Dict[str, str]] = None,
    force_recompute: bool = False
) -> Tuple[Data, Dict[str, int]]:
    """
    Build dynamic graph based on rolling correlation and cache to parquet.
    
    Args:
        prices: Historical price data
        date: Current date
        threshold: Correlation threshold for edge creation
        cache_to_parquet: Whether to cache graph to parquet
        config: Configuration dict (for storage paths)
        sector_map: Optional mapping of ticker -> sector (unused, kept for compatibility)
        force_recompute: If True, recompute even if cache exists
    
    Returns:
        PyTorch Geometric Data object and ticker_to_idx mapping
    """
    
    lookback_days = 30
    
    # Get or build graph (from cache or compute fresh)
    if cache_to_parquet:
        correlations, ticker_to_idx = get_or_build_graph(
            prices,
            date,
            threshold,
            lookback_days,
            sector_map,
            config,
            force_recompute
        )
    else:
        # Compute fresh without caching
        corr_matrix, tickers = compute_correlation_matrix(prices, date, lookback_days)
        
        if corr_matrix.empty or not tickers:
            return Data(x=torch.zeros((1, 128))), {}
        
        ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
        correlations = {}
        
        for i, ticker_i in enumerate(tickers):
            for j, ticker_j in enumerate(tickers):
                if i < j:
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > threshold:
                        correlations[(ticker_i, ticker_j)] = float(corr_value)
    
    if not correlations:
        # No edges found, return graph with isolated nodes
        if not ticker_to_idx:
            return Data(x=torch.zeros((1, 128))), {}
        num_nodes = len(ticker_to_idx)
        x = torch.zeros((num_nodes, 128))  # Placeholder
        return Data(x=x), ticker_to_idx
    
    # Build edge_index and edge_attr from correlations
    ticker_list = sorted(ticker_to_idx.keys())
    edge_index = []
    edge_attr = []
    
    for (ticker_i, ticker_j), corr_value in correlations.items():
        idx_i = ticker_to_idx.get(ticker_i)
        idx_j = ticker_to_idx.get(ticker_j)
        
        if idx_i is not None and idx_j is not None:
            edge_index.append([idx_i, idx_j])
            edge_attr.append(corr_value)
            # Add reverse edge for undirected graph
            edge_index.append([idx_j, idx_i])
            edge_attr.append(corr_value)
    
    if not edge_index:
        num_nodes = len(ticker_to_idx)
        x = torch.zeros((num_nodes, 128))  # Placeholder
        return Data(x=x), ticker_to_idx
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
    
    # Node features (will be filled by TFT embeddings later)
    num_nodes = len(ticker_to_idx)
    x = torch.zeros((num_nodes, 128))  # Placeholder
    
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    logger.info(f"Built correlation graph: {num_nodes} nodes, {len(edge_index[0]) // 2} edges")
    
    return graph, ticker_to_idx


def build_correlation_graph_from_parquet(
    date: str,
    tickers: Optional[List[str]] = None,
    threshold: float = 0.3,
    config: Optional[Dict] = None
) -> Tuple[Data, Dict[str, int]]:
    """
    Build correlation graph from parquet cache instead of computing from scratch.
    
    Args:
        date: Date of the correlation graph
        tickers: Optional list of tickers to filter
        threshold: Minimum absolute correlation
        config: Configuration dict (for storage paths)
    
    Returns:
        PyTorch Geometric Data object and ticker_to_idx mapping
    """
    
    from src.features.graph_storage import load_correlation_graph
    
    # Load edges from parquet
    edges_df, ticker_to_idx = load_correlation_graph(date, tickers, threshold, config)
    
    if edges_df is None or not ticker_to_idx:
        logger.warning(f"No graph found in cache for {date}")
        if ticker_to_idx:
            num_nodes = len(ticker_to_idx)
            return Data(x=torch.zeros((num_nodes, 128))), ticker_to_idx
        return Data(x=torch.zeros((1, 128))), {}
    
    # Convert to PyTorch Geometric format
    edge_index = []
    edge_attr = []
    
    for _, row in edges_df.iterrows():
        ticker1 = row['ticker1']
        ticker2 = row['ticker2']
        corr_value = row['correlation']
        
        idx1 = ticker_to_idx.get(ticker1)
        idx2 = ticker_to_idx.get(ticker2)
        
        if idx1 is not None and idx2 is not None:
            edge_index.append([idx1, idx2])
            edge_attr.append(corr_value)
            # Add reverse edge for undirected graph
            edge_index.append([idx2, idx1])
            edge_attr.append(corr_value)
    
    if not edge_index:
        num_nodes = len(ticker_to_idx)
        return Data(x=torch.zeros((num_nodes, 128))), ticker_to_idx
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
    
    num_nodes = len(ticker_to_idx)
    x = torch.zeros((num_nodes, 128))  # Placeholder
    
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    logger.info(f"Built graph from parquet cache: {num_nodes} nodes, {len(edge_index[0]) // 2} edges")
    
    return graph, ticker_to_idx


def build_heterogeneous_graph(
    prices_df: pd.DataFrame,
    date: str,
    threshold: float = 0.3,
    config: Optional[Dict] = None
) -> HeteroData:
    """
    Build heterogeneous graph with stocks, sectors, and macro nodes.
    
    Args:
        prices_df: Historical price data
        date: Current date
        threshold: Correlation threshold for stock-stock edges
        config: Optional config dict for graph storage paths
    
    Returns:
        HeteroData object with multiple node types and edge types
    """
    
    data = HeteroData()
    
    # 1. Stock nodes
    tickers = prices_df['ticker'].unique().tolist()
    num_stocks = len(tickers)
    data['stock'].x = torch.randn(num_stocks, 256)  # Placeholder, will be filled by TFT
    
    # 2. Sector nodes (11 S&P sectors)
    sectors = list(set(TICKER_TO_SECTOR.values()))
    num_sectors = len(sectors)
    sector_to_idx = {sector: i for i, sector in enumerate(sectors)}
    data['sector'].x = torch.randn(num_sectors, 64)
    
    # 3. Macro regime node (single node representing market state)
    # In production, this would use actual macro data (FRED, VIX)
    # For now, use placeholder that will be filled with actual macro features
    macro_features = torch.randn(1, 32)  # Placeholder
    # TODO: Replace with actual macro features from FRED/VIX data
    data['macro'].x = macro_features
    
    # 4. Stock-Stock edges (correlations)
    corr_matrix, _ = compute_correlation_matrix(prices_df, date, lookback_days=60)
    edge_index_stock_stock = []
    edge_weights = []
    
    if not corr_matrix.empty:
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                if i < len(corr_matrix.index) and j < len(corr_matrix.columns):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > threshold:
                        edge_index_stock_stock.append([i, j])
                        edge_index_stock_stock.append([j, i])  # Undirected
                        edge_weights.extend([float(corr_value), float(corr_value)])
    
    if edge_index_stock_stock:
        data['stock', 'correlates_with', 'stock'].edge_index = torch.tensor(edge_index_stock_stock, dtype=torch.long).t().contiguous()
        data['stock', 'correlates_with', 'stock'].edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
    
    # 5. Stock-Sector edges (belongs_to)
    edge_index_stock_sector = []
    for i, ticker in enumerate(tickers):
        sector = TICKER_TO_SECTOR.get(ticker, 'Unknown')
        if sector in sector_to_idx:
            edge_index_stock_sector.append([i, sector_to_idx[sector]])
    
    if edge_index_stock_sector:
        data['stock', 'belongs_to', 'sector'].edge_index = torch.tensor(edge_index_stock_sector, dtype=torch.long).t().contiguous()
    
    # 6. Sector-Macro edges (influenced_by)
    edge_index_sector_macro = [[i, 0] for i in range(num_sectors)]
    data['sector', 'influenced_by', 'macro'].edge_index = torch.tensor(edge_index_sector_macro, dtype=torch.long).t().contiguous()
    
    logger.info(f"Built heterogeneous graph: {num_stocks} stocks, {num_sectors} sectors, 1 macro node")
    logger.info(f"Stock-stock edges: {len(edge_index_stock_stock) // 2 if edge_index_stock_stock else 0}")
    logger.info(f"Stock-sector edges: {len(edge_index_stock_sector)}")
    logger.info(f"Sector-macro edges: {num_sectors}")
    
    return data

