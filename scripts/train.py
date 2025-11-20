"""Unified training CLI using PyTorch Lightning

Supports training:
- foundation: Foundation model pre-training
- twin: Digital twin fine-tuning
- rl: RL portfolio agent
- lightgbm: LightGBM cross-sectional ranker
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_foundation import train_foundation_model
from src.training.train_twins_lightning import fine_tune_twin
from src.training.train_rl_portfolio import load_training_data, train_rl_agent
from src.training.train_lightgbm_lightning import train_lightgbm_ranker
from src.models.foundation import load_foundation_model
from src.agents.portfolio_rl_agent import PortfolioRLAgent
from src.training.rl_environment import TradingEnvironment
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def train_foundation(args):
    """Train foundation model."""
    logger.info("Training foundation model...")
    train_foundation_model(
        config_path=args.config,
        use_synthetic_data=args.use_synthetic
    )


def train_twin(args):
    """Train digital twin for a ticker."""
    logger.info(f"Training twin for {args.ticker}...")
    
    config = load_config(args.config)
    
    # Load foundation model
    foundation_path = config.get('models', {}).get('foundation', {}).get('checkpoint_path')
    if not foundation_path:
        logger.error("Foundation model checkpoint path not configured")
        return
    
    foundation_model = load_foundation_model(foundation_path, config.get('models', {}).get('foundation', {}))
    
    # Fine-tune twin
    fine_tune_twin(
        ticker=args.ticker,
        foundation_model=foundation_model,
        config=config,
        lookback_days=args.lookback_days
    )


def train_rl(args):
    """Train RL portfolio agent."""
    logger.info("Training RL portfolio agent...")
    
    config = load_config(args.config)
    
    # Load training data
    data = load_training_data(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
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
        historical_options_features=data.get('options_features', None)
    )
    
    # Create agent
    rl_config = config.get('rl_portfolio', {})
    agent = PortfolioRLAgent(config=rl_config)
    
    # Train
    train_rl_agent(
        agent=agent,
        env=env,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        checkpoint_dir=args.checkpoint_dir,
        config=rl_config
    )


def train_lightgbm(args):
    """Train LightGBM ranker."""
    logger.info("Training LightGBM ranker...")
    
    config = load_config(args.config)
    
    train_lightgbm_ranker(
        config=config,
        start_date=args.start_date,
        end_date=args.end_date,
        model_version=args.version,
        output_dir=args.output_dir
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Unified training CLI')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Training command')
    
    # Foundation training
    foundation_parser = subparsers.add_parser('foundation', help='Train foundation model')
    foundation_parser.add_argument('--use-synthetic', action='store_true', help='Use synthetic data')
    foundation_parser.set_defaults(func=train_foundation)
    
    # Twin training
    twin_parser = subparsers.add_parser('twin', help='Train digital twin')
    twin_parser.add_argument('--ticker', type=str, required=True, help='Stock ticker')
    twin_parser.add_argument('--lookback-days', type=int, default=180, help='Lookback days')
    twin_parser.set_defaults(func=train_twin)
    
    # RL training
    rl_parser = subparsers.add_parser('rl', help='Train RL portfolio agent')
    rl_parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date')
    rl_parser.add_argument('--end-date', type=str, default='2024-01-01', help='End date')
    rl_parser.add_argument('--num-episodes', type=int, default=1000, help='Number of episodes')
    rl_parser.add_argument('--episode-length', type=int, default=60, help='Episode length')
    rl_parser.add_argument('--checkpoint-dir', type=str, default='models/rl_portfolio/', help='Checkpoint directory')
    rl_parser.set_defaults(func=train_rl)
    
    # LightGBM training
    lgbm_parser = subparsers.add_parser('lightgbm', help='Train LightGBM ranker')
    lgbm_parser.add_argument('--start-date', type=str, default='2022-01-01', help='Start date')
    lgbm_parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date')
    lgbm_parser.add_argument('--version', type=str, default='v1.0', help='Model version')
    lgbm_parser.add_argument('--output-dir', type=str, default='models/ensemble', help='Output directory')
    lgbm_parser.set_defaults(func=train_lightgbm)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()

