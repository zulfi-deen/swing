"""Training script for RL Portfolio Agent"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

from src.agents.portfolio_rl_agent import PortfolioRLAgent, PPOTrainer
from src.training.rl_environment import TradingEnvironment
from src.training.train_rl_portfolio_lightning import train_rl_agent as train_rl_agent_lightning
from src.data.storage import get_timescaledb_engine, load_from_local
from src.utils.config import load_config
from src.utils.mlflow_logger import log_metrics, log_model

logger = logging.getLogger(__name__)


def load_training_data(
    start_date: str,
    end_date: str,
    tickers: Optional[List[str]] = None
) -> Dict:
    """
    Load historical data for RL training.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        tickers: Optional list of tickers to filter
    
    Returns:
        Dict with predictions, prices, features, macro_data
    """
    
    engine = get_timescaledb_engine()
    
    # Load predictions
    pred_query = f"""
        SELECT 
            time::date as date,
            ticker,
            expected_return,
            hit_prob_long as hit_probability,
            volatility_forecast as volatility,
            quantile_10,
            quantile_90,
            regime
        FROM predictions
        WHERE time >= '{start_date}' AND time <= '{end_date}'
    """
    
    if tickers:
        ticker_list = "', '".join(tickers)
        pred_query += f" AND ticker IN ('{ticker_list}')"
    
    predictions = pd.read_sql(pred_query, engine)
    
    # Load prices
    price_query = f"""
        SELECT 
            time::date as date,
            ticker,
            close,
            volume
        FROM prices
        WHERE time >= '{start_date}' AND time <= '{end_date}'
    """
    
    if tickers:
        price_query += f" AND ticker IN ('{ticker_list}')"
    
    prices = pd.read_sql(price_query, engine)
    
    # Load features
    feature_query = f"""
        SELECT 
            time::date as date,
            ticker,
            rsi_14,
            macd_signal,
            volume_z_score,
            sentiment_score,
            pattern_confidence,
            days_to_earnings
        FROM features
        WHERE time >= '{start_date}' AND time <= '{end_date}'
    """
    
    if tickers:
        feature_query += f" AND ticker IN ('{ticker_list}')"
    
    features = pd.read_sql(feature_query, engine)
    
    # Load macro data (simplified - would come from external source)
    # For now, create placeholder
    dates = sorted(predictions['date'].unique())
    macro_data = pd.DataFrame({
        'date': dates,
        'vix': 20.0,
        'spy_return_5d': 0.0,
        'treasury_10y': 4.0,
        'market_regime': 'bull'
    })
    
    # Load options features (if available)
    options_features_df = pd.DataFrame()
    try:
        options_query = f"""
            SELECT 
                time::date as date,
                ticker,
                trend_signal,
                sentiment_signal,
                gamma_signal,
                pcr_zscore,
                pcr_extreme_bullish,
                pcr_extreme_bearish,
                max_pain_distance_pct,
                iv_percentile,
                net_delta
            FROM options_features
            WHERE time >= '{start_date}' AND time <= '{end_date}'
        """
        if tickers:
            options_query += f" AND ticker IN ('{ticker_list}')"
        options_features_df = pd.read_sql(options_query, engine)
        logger.info(f"Loaded {len(options_features_df)} options features")
    except Exception as e:
        logger.warning(f"Could not load options features: {e}")
    
    logger.info(f"Loaded training data: {len(predictions)} predictions, {len(prices)} prices, {len(features)} features")
    
    return {
        'predictions': predictions,
        'prices': prices,
        'features': features,
        'macro_data': macro_data,
        'options_features': options_features_df
    }


def train_rl_agent(
    agent: PortfolioRLAgent,
    env: TradingEnvironment,
    num_episodes: int = 1000,
    episode_length: int = 60,
    checkpoint_dir: str = 'models/rl_portfolio/',
    config: Optional[Dict] = None
):
    """
    Train RL agent using Lightning (delegates to Lightning implementation).
    
    Each episode = 60 trading days (3 months)
    """
    # Update config with episode_length
    if config is None:
        config = {}
    if 'rl_portfolio' not in config:
        config['rl_portfolio'] = {}
    config['rl_portfolio']['episode_length'] = episode_length
    
    # Use Lightning-based training
    train_rl_agent_lightning(agent, env, config, num_episodes, checkpoint_dir)


def main():
    """Main training function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RL Portfolio Agent')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Training start date')
    parser.add_argument('--end-date', type=str, default='2024-01-01', help='Training end date')
    parser.add_argument('--num-episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--episode-length', type=int, default=60, help='Episode length (trading days)')
    parser.add_argument('--checkpoint-dir', type=str, default='models/rl_portfolio/', help='Checkpoint directory')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config(args.config)
    
    # Initialize MLflow
    try:
        import mlflow
        from src.utils.mlflow_logger import init_mlflow
        init_mlflow(experiment_name='rl_portfolio_training')
        mlflow.start_run(run_name=f"rl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Log hyperparameters
        mlflow.log_params({
            'start_date': args.start_date,
            'end_date': args.end_date,
            'num_episodes': args.num_episodes,
            'episode_length': args.episode_length,
            **config.get('rl_portfolio', {})
        })
    except Exception as e:
        logger.warning(f"Could not initialize MLflow: {e}")
    
    # Load training data
    logger.info(f"Loading training data from {args.start_date} to {args.end_date}")
    data = load_training_data(args.start_date, args.end_date)
    
    # Create environment
    env = TradingEnvironment(
        historical_predictions=data['predictions'],
        historical_prices=data['prices'],
        historical_features=data['features'],
        macro_data=data['macro_data'],
        initial_capital=100000.0,
        max_positions=15,
        max_position_size=0.10,
        max_sector_exposure=0.25,
        transaction_cost=0.001,
        config=config,
        historical_options_features=data.get('options_features', pd.DataFrame())
    )
    
    # Create agent
    rl_config = config.get('rl_portfolio', {})
    agent = PortfolioRLAgent(config=rl_config)
    
    # Train
    logger.info("Starting RL training...")
    train_rl_agent(
        agent=agent,
        env=env,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        checkpoint_dir=args.checkpoint_dir,
        config=rl_config.get('ppo', {})
    )
    
    logger.info("Training complete!")
    
    # End MLflow run
    try:
        import mlflow
        mlflow.end_run()
    except:
        pass


if __name__ == '__main__':
    main()

