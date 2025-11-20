"""Ticker utilities"""

import pandas as pd
from typing import List, Dict, Optional

# Sector ETF mapping
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financial',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrial',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLU': 'Utilities',
    'XLC': 'Communication'
}

# Sector to ID mapping (for embeddings)
SECTORS = ['Technology', 'Financial', 'Energy', 'Healthcare', 'Consumer Discretionary',
           'Consumer Staples', 'Industrial', 'Materials', 'Real Estate', 'Utilities', 'Communication']
SECTOR_TO_ID = {sector: idx for idx, sector in enumerate(SECTORS)}
ID_TO_SECTOR = {idx: sector for idx, sector in enumerate(SECTORS)}

# Ticker to Sector mapping (sample - in production, fetch from Polygon/Finnhub)
TICKER_TO_SECTOR = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'AMZN': 'Technology', 'NVDA': 'Technology', 'META': 'Technology', 'AVGO': 'Technology',
    'ADBE': 'Technology', 'CSCO': 'Technology', 'ACN': 'Technology', 'AMD': 'Technology',
    'QCOM': 'Technology', 'INTU': 'Technology', 'AMAT': 'Technology', 'TXN': 'Technology',
    # Financial
    'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial',
    'MS': 'Financial', 'C': 'Financial', 'BLK': 'Financial', 'SCHW': 'Financial',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'SLB': 'Energy', 'EOG': 'Energy',
    # Healthcare
    'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'LLY': 'Healthcare', 'ABBV': 'Healthcare',
    'MRK': 'Healthcare', 'TMO': 'Healthcare', 'AMGN': 'Healthcare', 'ISRG': 'Healthcare',
    # Consumer Discretionary
    'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary', 'BKNG': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary',
    # Consumer Staples
    'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'COST': 'Consumer Staples', 'KO': 'Consumer Staples',
    # Industrial
    'HON': 'Industrial', 'RTX': 'Industrial', 'GE': 'Industrial', 'CAT': 'Industrial',
    # Materials
    'LIN': 'Materials', 'APD': 'Materials',
    # Communication
    'VZ': 'Communication', 'CMCSA': 'Communication', 'T': 'Communication',
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities',
    # Real Estate
    'AMT': 'Real Estate', 'PLD': 'Real Estate',
    # Other
    'V': 'Financial', 'MA': 'Financial', 'BRK.B': 'Financial',
    'DIS': 'Communication', 'NFLX': 'Communication', 'PM': 'Consumer Staples',
    'SPGI': 'Financial', 'AXP': 'Financial'
}


def get_sp500_tickers() -> List[str]:
    """
    Get S&P 500 ticker list.
    
    In production, this would fetch from a reliable source.
    For now, returns a sample list.
    """
    
    # Sample S&P 500 tickers (top 50 by market cap)
    sample_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
        'V', 'UNH', 'XOM', 'JNJ', 'JPM', 'WMT', 'MA', 'PG', 'CVX', 'LLY',
        'HD', 'ABBV', 'MRK', 'AVGO', 'COST', 'PEP', 'ADBE', 'TMO', 'MCD',
        'CSCO', 'ACN', 'NFLX', 'LIN', 'AMD', 'DIS', 'PM', 'NKE', 'TXN',
        'CMCSA', 'HON', 'AMGN', 'RTX', 'QCOM', 'INTU', 'AMAT', 'GE', 'ISRG',
        'VZ', 'BKNG', 'LOW', 'SPGI', 'AXP'
    ]
    
    return sample_tickers


def get_sector_mapping() -> Dict[str, str]:
    """Map tickers to sector ETFs."""
    return SECTOR_ETFS.copy()


def get_ticker_sector(ticker: str) -> Optional[str]:
    """
    Get sector for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Sector name or None if not found
    """
    return TICKER_TO_SECTOR.get(ticker)


def get_sector_etf(sector: str) -> Optional[str]:
    """
    Get sector ETF ticker for a sector name.
    
    Args:
        sector: Sector name (e.g., 'Technology')
    
    Returns:
        ETF ticker (e.g., 'XLK') or None
    """
    for etf, sec in SECTOR_ETFS.items():
        if sec == sector:
            return etf
    return None


def get_sector_tickers(sector: str) -> List[str]:
    """
    Get all tickers in a sector.
    
    Args:
        sector: Sector name
    
    Returns:
        List of ticker symbols
    """
    return [ticker for ticker, sec in TICKER_TO_SECTOR.items() if sec == sector]


def get_sector_etfs() -> List[str]:
    """Get list of all sector ETF tickers."""
    return list(SECTOR_ETFS.keys())

