"""Synthetic Data Generator for Stock Digital Twins

Generates realistic OHLCV data with trends, volatility regimes,
sector correlations, and macro sensitivity for demo/testing.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def generate_synthetic_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    base_price: float = 100.0,
    beta: float = 1.0,
    volatility: float = 0.02,
    trend: float = 0.0,
    sector_correlation: float = 0.5,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for a stock.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        base_price: Starting price
        beta: Beta coefficient (sensitivity to market)
        volatility: Daily volatility
        trend: Daily trend (drift)
        sector_correlation: Correlation to sector
        seed: Random seed
    
    Returns:
        DataFrame with OHLCV data
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends
    
    n_days = len(dates)
    
    # Generate returns with trend and volatility
    returns = np.random.normal(trend, volatility, n_days)
    
    # Add some autocorrelation (momentum/mean reversion)
    for i in range(1, n_days):
        returns[i] += 0.1 * returns[i-1]  # Slight momentum
    
    # Generate prices
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = prices[1:]  # Remove initial base_price
    
    # Generate OHLC from close
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # High/Low around close
        daily_range = close * volatility * np.random.uniform(0.5, 2.0)
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        # Open (between prev close and current close)
        if i > 0:
            prev_close = prices[i-1]
            open_price = prev_close + (close - prev_close) * np.random.uniform(0.2, 0.8)
        else:
            open_price = close * np.random.uniform(0.98, 1.02)
        
        # Volume (correlated with volatility)
        volume = int(1e6 * (1 + abs(returns[i]) * 10) * np.random.uniform(0.5, 2.0))
        
        # VWAP (volume-weighted average price)
        vwap = (high + low + close) / 3
        
        data.append({
            'time': date,
            'ticker': ticker,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'vwap': vwap
        })
    
    df = pd.DataFrame(data)
    return df


def generate_synthetic_universe(
    tickers: List[str],
    start_date: str,
    end_date: str,
    sector_map: Optional[Dict[str, str]] = None,
    base_prices: Optional[Dict[str, float]] = None,
    betas: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic price data for multiple stocks with correlations.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        sector_map: Optional mapping of ticker -> sector
        base_prices: Optional mapping of ticker -> base price
        betas: Optional mapping of ticker -> beta
        seed: Random seed
    
    Returns:
        DataFrame with OHLCV data for all tickers
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Default values
    if base_prices is None:
        base_prices = {ticker: 100.0 * np.random.uniform(0.5, 2.0) for ticker in tickers}
    
    if betas is None:
        betas = {ticker: np.random.uniform(0.5, 2.0) for ticker in tickers}
    
    # Generate market factor (common to all stocks)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]
    market_returns = np.random.normal(0.0005, 0.015, len(dates))  # Slight positive drift
    
    all_data = []
    
    for ticker in tickers:
        # Stock-specific parameters
        base_price = base_prices.get(ticker, 100.0)
        beta = betas.get(ticker, 1.0)
        volatility = 0.015 + beta * 0.01  # Higher beta = higher vol
        
        # Generate base returns
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]
        n_days = len(dates)
        
        # Idiosyncratic returns
        idiosyncratic = np.random.normal(0, volatility * 0.5, n_days)
        
        # Market component (beta-adjusted)
        market_component = beta * market_returns[:n_days]
        
        # Sector component (if applicable)
        sector_component = np.zeros(n_days)
        if sector_map and ticker in sector_map:
            # Add sector correlation
            sector_component = np.random.normal(0, volatility * 0.3, n_days)
        
        # Combined returns
        returns = market_component + idiosyncratic + sector_component * 0.3
        
        # Generate prices
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        prices = prices[1:]
        
        # Generate OHLCV
        for i, (date, close) in enumerate(zip(dates, prices)):
            daily_range = close * volatility * np.random.uniform(0.5, 2.0)
            high = close + daily_range * np.random.uniform(0.3, 1.0)
            low = close - daily_range * np.random.uniform(0.3, 1.0)
            
            if i > 0:
                prev_close = prices[i-1]
                open_price = prev_close + (close - prev_close) * np.random.uniform(0.2, 0.8)
            else:
                open_price = close * np.random.uniform(0.98, 1.02)
            
            volume = int(1e6 * (1 + abs(returns[i]) * 10) * np.random.uniform(0.5, 2.0))
            vwap = (high + low + close) / 3
            
            all_data.append({
                'time': date,
                'ticker': ticker,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'vwap': vwap
            })
    
    df = pd.DataFrame(all_data)
    return df


def generate_synthetic_features(
    prices_df: pd.DataFrame,
    tickers: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate synthetic features from price data.
    
    This is a placeholder - in production, would compute real technical indicators.
    For now, generates random features that look realistic.
    """
    
    if tickers is None:
        tickers = prices_df['ticker'].unique()
    
    all_features = []
    
    for ticker in tickers:
        ticker_data = prices_df[prices_df['ticker'] == ticker].copy()
        
        if ticker_data.empty:
            continue
        
        # Generate synthetic technical indicators
        ticker_data['rsi_14'] = np.random.uniform(30, 70, len(ticker_data))
        ticker_data['macd'] = np.random.normal(0, 0.5, len(ticker_data))
        ticker_data['macd_signal'] = ticker_data['macd'] * 0.9
        ticker_data['bbands_pct'] = np.random.uniform(0, 1, len(ticker_data))
        ticker_data['atr_14'] = ticker_data['close'] * np.random.uniform(0.01, 0.03, len(ticker_data))
        ticker_data['volume_z_score'] = np.random.normal(0, 1, len(ticker_data))
        
        # Cross-sectional features
        ticker_data['return_rank_5d'] = np.random.randint(1, 501, len(ticker_data))
        ticker_data['sector_relative_strength'] = np.random.uniform(-0.1, 0.1, len(ticker_data))
        ticker_data['correlation_to_spy'] = np.random.uniform(0.3, 0.9, len(ticker_data))
        
        # Text features (synthetic)
        ticker_data['sentiment_score'] = np.random.uniform(-1, 1, len(ticker_data))
        ticker_data['news_intensity'] = np.random.choice(['low', 'medium', 'high'], len(ticker_data))
        
        all_features.append(ticker_data)
    
    features_df = pd.concat(all_features, ignore_index=True)
    return features_df


def create_synthetic_dataset(
    tickers: List[str],
    start_date: str,
    end_date: str,
    sector_map: Optional[Dict[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Create complete synthetic dataset (prices + features).
    
    Returns:
        Dict with 'prices' and 'features' DataFrames
    """
    
    logger.info(f"Generating synthetic data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    # Generate prices
    prices_df = generate_synthetic_universe(
        tickers,
        start_date,
        end_date,
        sector_map=sector_map
    )
    
    # Generate features
    features_df = generate_synthetic_features(prices_df, tickers)
    
    return {
        'prices': prices_df,
        'features': features_df
    }


