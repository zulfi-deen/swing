"""DEPRECATED: Export training data for Colab training

This script is deprecated. The codebase now uses Lightning.ai for training and deployment.
Training data is accessed directly from TimescaleDB or local storage.

This file is kept for backward compatibility only and will be removed in a future version.
"""

import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import argparse

from src.data.storage import get_timescaledb_engine
from src.utils.config import load_config
from src.utils.colab_utils import get_data_paths, is_colab

logger = logging.getLogger(__name__)


def export_prices(
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_dir: str,
    engine
) -> str:
    """
    Export price data for tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory
        engine: Database engine
    
    Returns:
        Path to exported file
    """
    
    query = """
        SELECT * FROM prices
        WHERE time >= %s AND time <= %s
        AND ticker = ANY(%s)
        ORDER BY time, ticker
    """
    
    logger.info(f"Exporting prices for {len(tickers)} tickers from {start_date} to {end_date}")
    df = pd.read_sql(query, engine, params=(start_date, end_date, tickers))
    
    if df.empty:
        logger.warning("No price data found")
        return None
    
    # Save to parquet
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"prices_{start_date}_{end_date}.parquet")
    df.to_parquet(output_path, index=False, compression='snappy')
    
    logger.info(f"Exported {len(df)} price records to {output_path}")
    return output_path


def export_stock_characteristics(
    tickers: List[str],
    output_dir: str,
    engine
) -> str:
    """
    Export stock characteristics.
    
    Args:
        tickers: List of ticker symbols
        output_dir: Output directory
        engine: Database engine
    
    Returns:
        Path to exported file
    """
    
    query = """
        SELECT * FROM stock_characteristics
        WHERE ticker = ANY(%s)
    """
    
    logger.info(f"Exporting stock characteristics for {len(tickers)} tickers")
    df = pd.read_sql(query, engine, params=(tickers,))
    
    if df.empty:
        logger.warning("No stock characteristics found")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "stock_characteristics.parquet")
    df.to_parquet(output_path, index=False, compression='snappy')
    
    logger.info(f"Exported {len(df)} stock characteristics to {output_path}")
    return output_path


def export_training_data(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookback_days: int = 1095,  # 3 years
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None
):
    """
    Export training data for Colab.
    
    Args:
        tickers: List of tickers to export (defaults to pilot tickers)
        start_date: Start date (YYYY-MM-DD), defaults to lookback_days ago
        end_date: End date (YYYY-MM-DD), defaults to today
        lookback_days: Number of days to look back if start_date not provided
        output_dir: Output directory (defaults to Google Drive path or local data/training)
        config_path: Path to config file
    """
    
    config = load_config(config_path)
    
    # Get output directory
    if output_dir is None:
        if is_colab():
            data_path, _ = get_data_paths(config)
            output_dir = os.path.join(data_path, 'training')
        else:
            # On MacBook, save to local data/training
            data_path = config.get('storage', {}).get('local', {}).get('data_dir', 'data/')
            project_root = Path(__file__).parent.parent
            output_dir = str(project_root / data_path / 'training')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get tickers
    if tickers is None:
        tickers = config.get('models', {}).get('twins', {}).get('pilot_tickers', [])
        if not tickers:
            logger.warning("No pilot tickers configured, using default tickers")
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Get date range
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    logger.info(f"Exporting training data for {len(tickers)} tickers")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Output directory: {output_dir}")
    
    # Get database engine
    engine = get_timescaledb_engine()
    
    # Export prices
    prices_path = export_prices(tickers, start_date, end_date, output_dir, engine)
    
    # Export stock characteristics
    chars_path = export_stock_characteristics(tickers, output_dir, engine)
    
    # Create metadata file
    metadata = {
        'tickers': tickers,
        'start_date': start_date,
        'end_date': end_date,
        'export_date': datetime.now().isoformat(),
        'prices_file': os.path.basename(prices_path) if prices_path else None,
        'characteristics_file': os.path.basename(chars_path) if chars_path else None,
        'num_tickers': len(tickers)
    }
    
    import json
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Training data export complete. Metadata saved to {metadata_path}")
    logger.info(f"Files exported:")
    if prices_path:
        logger.info(f"  - {prices_path}")
    if chars_path:
        logger.info(f"  - {chars_path}")
    
    return output_dir


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Export training data for Colab')
    parser.add_argument('--tickers', nargs='+', help='List of tickers to export')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--lookback-days', type=int, default=1095, help='Days to look back (default: 1095)')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--config', help='Path to config file')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    export_training_data(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        output_dir=args.output_dir,
        config_path=args.config
    )


if __name__ == '__main__':
    main()

