"""RelatedStockAgent - Identify peer and sympathy plays"""

import pandas as pd
from typing import List, Dict
from src.utils.tickers import get_sector_mapping

class RelatedStockAgent:
    """
    Identifies stocks related to the target ticker via:
    1. Sector correlation
    2. Supply chain (placeholder for knowledge graph)
    3. Competitor analysis
    """
    
    def __init__(self):
        from src.utils.tickers import TICKER_TO_SECTOR
        self.ticker_to_sector = TICKER_TO_SECTOR

    def find_peers(self, ticker: str, universe_df: pd.DataFrame) -> List[Dict]:
        """
        Find top 5 correlated peers in the same sector.
        """
        if universe_df.empty:
            return []

        # 1. Identify Sector
        target_sector = self.ticker_to_sector.get(ticker)
        
        if not target_sector:
            # Unknown sector, return empty
            return []
            
        # 2. Correlation Analysis
        # Pivot to get returns matrix
        if 'date' in universe_df.columns:
            pivot = universe_df.pivot(index='date', columns='ticker', values='close')
        else:
            # If single day, cannot compute correlation. Return empty.
            return []
            
        returns = pivot.pct_change()
        
        if ticker not in returns.columns:
            return []
            
        correlations = returns.corrwith(returns[ticker]).sort_values(ascending=False)
        
        # Filter out self and get top 5
        peers = correlations.drop(ticker).head(5)
        
        results = []
        for peer_ticker, corr in peers.items():
            results.append({
                "ticker": peer_ticker,
                "correlation": round(corr, 2),
                "relationship": "Statistical Correlation"
            })
            
        return results

