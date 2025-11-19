"""Cross-sectional features (relative to universe)"""

import pandas as pd
import numpy as np
from typing import Dict, List


def compute_cross_sectional_features(universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectional features across all stocks.
    
    Args:
        universe_df: DataFrame with all stocks for a given day
    """
    
    if universe_df.empty:
        return universe_df
    
    # Ensure date column exists
    if 'date' not in universe_df.columns and 'time' in universe_df.columns:
        universe_df['date'] = pd.to_datetime(universe_df['time']).dt.date
    
    # Return rankings (within universe)
    universe_df['return_1d'] = universe_df.groupby('ticker')['close'].pct_change(1)
    universe_df['return_5d'] = universe_df.groupby('ticker')['close'].pct_change(5)
    universe_df['return_20d'] = universe_df.groupby('ticker')['close'].pct_change(20)
    
    # Rank within universe (1 = best, N = worst)
    if 'date' in universe_df.columns:
        date_col = 'date'
    else:
        date_col = universe_df.index.name if universe_df.index.name else None
    
    if date_col:
        universe_df['return_rank_1d'] = universe_df.groupby(date_col)['return_1d'].rank(
            ascending=False, method='dense'
        )
        universe_df['return_rank_5d'] = universe_df.groupby(date_col)['return_5d'].rank(
            ascending=False, method='dense'
        )
        universe_df['return_rank_20d'] = universe_df.groupby(date_col)['return_20d'].rank(
            ascending=False, method='dense'
        )
    else:
        # Fallback: rank across all rows
        universe_df['return_rank_1d'] = universe_df['return_1d'].rank(ascending=False)
        universe_df['return_rank_5d'] = universe_df['return_5d'].rank(ascending=False)
        universe_df['return_rank_20d'] = universe_df['return_20d'].rank(ascending=False)
    
    return universe_df


def compute_correlation_features(prices: pd.DataFrame, spy_ticker: str = "SPY") -> pd.DataFrame:
    """Compute correlation to SPY and other indices."""
    
    if prices.empty:
        return pd.DataFrame()
    
    # Pivot to ticker columns
    if 'date' not in prices.columns and 'time' in prices.columns:
        prices['date'] = pd.to_datetime(prices['time']).dt.date
    
    date_col = 'date' if 'date' in prices.columns else prices.index.name
    
    if date_col:
        returns = prices.pivot_table(
            index=date_col,
            columns='ticker',
            values='close'
        ).pct_change()
    else:
        returns = prices.pivot(columns='ticker', values='close').pct_change()
    
    # Rolling 20-day correlation to SPY
    if spy_ticker not in returns.columns:
        return pd.DataFrame()
    
    spy_returns = returns[spy_ticker]
    
    corr_features = {}
    for ticker in returns.columns:
        if ticker == spy_ticker:
            continue
        
        corr = returns[ticker].rolling(20).corr(spy_returns)
        corr_features[f'{ticker}_corr_spy'] = corr
    
    if not corr_features:
        return pd.DataFrame()
    
    # Convert to long format
    corr_df = pd.DataFrame(corr_features)
    corr_df = corr_df.reset_index().melt(
        id_vars=[date_col] if date_col else [],
        var_name='ticker',
        value_name='correlation_to_spy'
    )
    
    # Extract ticker from column name
    corr_df['ticker'] = corr_df['ticker'].str.replace('_corr_spy', '')
    
    return corr_df


def compute_peer_features(
    ticker: str,
    universe_df: pd.DataFrame,
    sector_map: Dict[str, str] = None
) -> Dict:
    """Analyze peer stocks (same sector)."""
    
    if sector_map is None:
        # Default: all stocks are peers
        peers = universe_df['ticker'].unique().tolist()
    else:
        sector = sector_map.get(ticker)
        if sector:
            peers = [t for t, s in sector_map.items() if s == sector and t != ticker]
        else:
            peers = []
    
    if not peers or ticker not in universe_df['ticker'].values:
        return {
            'peer_median_return_5d': 0.0,
            'peer_outperformance': 0.0,
            'peer_percentile': 0.5
        }
    
    # Peer performance
    peer_mask = universe_df['ticker'].isin(peers)
    ticker_mask = universe_df['ticker'] == ticker
    
    if 'return_5d' not in universe_df.columns:
        return {
            'peer_median_return_5d': 0.0,
            'peer_outperformance': 0.0,
            'peer_percentile': 0.5
        }
    
    peer_returns = universe_df[peer_mask]['return_5d'].dropna()
    ticker_return = universe_df[ticker_mask]['return_5d'].values
    
    if len(peer_returns) == 0 or len(ticker_return) == 0:
        return {
            'peer_median_return_5d': 0.0,
            'peer_outperformance': 0.0,
            'peer_percentile': 0.5
        }
    
    ticker_return_val = ticker_return[0]
    
    features = {
        'peer_median_return_5d': peer_returns.median(),
        'peer_outperformance': ticker_return_val - peer_returns.median(),
        'peer_percentile': (peer_returns < ticker_return_val).mean()
    }
    
    return features

