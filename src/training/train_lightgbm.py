"""Train LightGBM Cross-Sectional Ranker

Trains a LightGBM model to rank stocks based on cross-sectional features.
This complements the digital twin predictions by adding cross-sectional analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import joblib
from pathlib import Path

from src.models.ensemble import LightGBMRanker
from src.data.storage import get_timescaledb_engine
from src.utils.config import load_config
from src.utils.mlflow_logger import log_training_run, log_model

logger = logging.getLogger(__name__)


def load_training_data(start_date: str, end_date: str):
    """
    Load historical features and returns for LightGBM training.
    
    Returns:
        train_df, val_df DataFrames with features and target (return_5d)
    """
    
    engine = get_timescaledb_engine()
    
    logger.info(f"Loading data from {start_date} to {end_date}")
    
    # Load features and prices
    query = f"""
        SELECT 
            f.time,
            f.ticker,
            p.close,
            -- Technical features
            f.rsi_14,
            f.macd,
            f.macd_signal,
            f.atr_14,
            f.volume_z_score,
            f.sentiment_score,
            f.pattern_confidence,
            -- Cross-sectional features
            f.return_rank_5d,
            f.return_rank_20d,
            f.sector_relative_strength,
            f.correlation_to_spy,
            -- Stock characteristics
            sc.beta,
            sc.sector,
            sc.liquidity_regime
        FROM features f
        INNER JOIN prices p ON f.time = p.time AND f.ticker = p.ticker
        LEFT JOIN stock_characteristics sc ON f.ticker = sc.ticker
        WHERE f.time >= '{start_date}' AND f.time <= '{end_date}'
        ORDER BY f.time, f.ticker
    """
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        logger.error("No data loaded from database")
        return None, None
    
    logger.info(f"Loaded {len(df)} rows for {df['ticker'].nunique()} tickers")
    
    # Compute target: 5-day forward return
    df['return_5d'] = df.groupby('ticker')['close'].pct_change(periods=5).shift(-5)
    
    # Drop rows without target
    df = df.dropna(subset=['return_5d'])
    
    # Fill missing features
    df['rsi_14'] = df['rsi_14'].fillna(50)
    df['macd'] = df['macd'].fillna(0)
    df['macd_signal'] = df['macd_signal'].fillna(0)
    df['atr_14'] = df['atr_14'].fillna(0.02)
    df['volume_z_score'] = df['volume_z_score'].fillna(0)
    df['sentiment_score'] = df['sentiment_score'].fillna(0.5)
    df['pattern_confidence'] = df['pattern_confidence'].fillna(0)
    df['return_rank_5d'] = df['return_rank_5d'].fillna(250)
    df['return_rank_20d'] = df['return_rank_20d'].fillna(250)
    df['sector_relative_strength'] = df['sector_relative_strength'].fillna(0)
    df['correlation_to_spy'] = df['correlation_to_spy'].fillna(0.5)
    df['beta'] = df['beta'].fillna(1.0)
    df['liquidity_regime'] = df['liquidity_regime'].fillna(1)
    
    # Split train/val (80/20 by time)
    split_date = df['time'].quantile(0.8)
    train_df = df[df['time'] <= split_date].copy()
    val_df = df[df['time'] > split_date].copy()
    
    logger.info(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")
    
    return train_df, val_df


def train_lightgbm_ranker(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict
) -> LightGBMRanker:
    """
    Train LightGBM ranker on historical data.
    
    Args:
        train_df: Training data
        val_df: Validation data
        config: Configuration dict
    
    Returns:
        Trained LightGBMRanker
    """
    
    logger.info("Training LightGBM ranker...")
    
    # Feature columns
    feature_cols = [
        'rsi_14', 'macd', 'macd_signal', 'atr_14',
        'volume_z_score', 'return_rank_5d', 'return_rank_20d',
        'sector_relative_strength', 'correlation_to_spy',
        'sentiment_score', 'pattern_confidence', 'beta', 'liquidity_regime'
    ]
    
    # Filter to available columns
    available_cols = [col for col in feature_cols if col in train_df.columns]
    
    logger.info(f"Using {len(available_cols)} features: {available_cols}")
    
    X_train = train_df[available_cols].fillna(0)
    y_train = train_df['return_5d']
    
    X_val = val_df[available_cols].fillna(0)
    y_val = val_df['return_5d']
    
    # Create query groups (by date)
    # LightGBM ranker needs to know which rows belong to the same query (date)
    train_groups = train_df.groupby('time').size().values
    val_groups = val_df.groupby('time').size().values
    
    logger.info(f"Train groups: {len(train_groups)}, Val groups: {len(val_groups)}")
    
    # Get LightGBM config
    lgbm_config = config.get('models', {}).get('lightgbm', {})
    
    # Train
    ranker = LightGBMRanker(**lgbm_config)
    
    ranker.fit(
        X_train,
        y_train,
        groups=train_groups,
        eval_set=[(X_val, y_val)],
        eval_groups=[val_groups]
    )
    
    logger.info("LightGBM training completed")
    
    # Log feature importance
    feature_importance = ranker.get_feature_importance()
    logger.info(f"Top 10 features:\n{feature_importance.head(10)}")
    
    return ranker


def main():
    """Main training function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LightGBM ranker')
    parser.add_argument('--start-date', type=str, default='2022-01-01', help='Training start date')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='Training end date')
    parser.add_argument('--output-dir', type=str, default='models/ensemble', help='Output directory')
    parser.add_argument('--version', type=str, default='v1.0', help='Model version')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config()
    
    # Load data
    train_df, val_df = load_training_data(args.start_date, args.end_date)
    
    if train_df is None:
        logger.error("Failed to load training data")
        return
    
    # Train
    ranker = train_lightgbm_ranker(train_df, val_df, config)
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f'lgbm_ranker_{args.version}.pkl'
    joblib.dump(ranker, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    # Log to MLflow
    try:
        feature_importance = ranker.get_feature_importance()
        
        metrics = {
            'num_features': len(feature_importance),
            'top_feature_importance': feature_importance['importance'].iloc[0] if not feature_importance.empty else 0,
        }
        
        params = {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'train_size': len(train_df),
            'val_size': len(val_df),
            **ranker.params
        }
        
        log_training_run(
            metrics=metrics,
            params=params,
            model_version=args.version,
            tags={'model_type': 'lightgbm_ranker'}
        )
        
        log_model(ranker, 'lightgbm_ranker', model_type='lightgbm')
        
        logger.info("Logged to MLflow")
    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()

