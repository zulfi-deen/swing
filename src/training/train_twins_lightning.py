"""Digital Twin Fine-Tuning with PyTorch Lightning"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional
import logging
import os
from datetime import datetime

from src.models.foundation import StockTwinFoundation, load_foundation_model
from src.models.digital_twin import StockDigitalTwin
from src.training.data_modules import TwinDataModule
from src.training.loss import TwinLoss
from src.training.lightning_utils import create_callbacks, create_logger, create_trainer
from src.data.storage import get_stock_characteristics, compute_stock_characteristics, get_timescaledb_engine
from src.utils.config import load_config
from src.utils.colab_utils import setup_colab_environment
import pandas as pd

logger = logging.getLogger(__name__)


class TwinLightningModule(pl.LightningModule):
    """
    Lightning module for digital twin fine-tuning.
    """
    
    def __init__(
        self,
        foundation_model: StockTwinFoundation,
        ticker: str,
        stock_characteristics: Dict,
        config: Dict,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-5
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['foundation_model', 'config'])
        self.ticker = ticker
        self.config = config
        
        # Create twin
        twin_config = config.get('models', {}).get('twins', {})
        self.twin = StockDigitalTwin(
            foundation_model,
            ticker,
            stock_characteristics,
            twin_config
        )
        
        # Loss function
        self.loss_fn = TwinLoss()
        
        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def forward(self, batch, graph=None, options_features=None):
        """Forward pass."""
        return self.twin(batch, graph, options_features)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        features, targets = batch
        
        # Convert to dict format expected by twin
        batch_dict = {'features': features}
        
        # Forward pass
        predictions = self.twin(batch_dict)
        
        # Prepare targets dict
        targets_dict = {
            'return_5d': targets,
            'hit_target': (targets > 0).float()
        }
        
        # Compute loss
        loss_dict = self.loss_fn(predictions, targets_dict)
        loss = loss_dict['total']
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_return_loss', loss_dict['return'], on_step=False, on_epoch=True)
        self.log('train_prob_loss', loss_dict['prob'], on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        features, targets = batch
        
        batch_dict = {'features': features}
        predictions = self.twin(batch_dict)
        
        targets_dict = {
            'return_5d': targets,
            'hit_target': (targets > 0).float()
        }
        
        loss_dict = self.loss_fn(predictions, targets_dict)
        loss = loss_dict['total']
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_return_loss', loss_dict['return'], on_step=False, on_epoch=True)
        self.log('val_prob_loss', loss_dict['prob'], on_step=False, on_epoch=True)
        
        # Compute additional metrics
        mae = torch.mean(torch.abs(predictions['expected_return'] - targets))
        hit_rate = torch.mean((predictions['hit_prob'] > 0.5).float() == targets_dict['hit_target'])
        
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        self.log('val_hit_rate', hit_rate, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.twin.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.get('training', {}).get('twins', {}).get('fine_tune_epochs', 20)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


def fine_tune_twin(
    ticker: str,
    foundation_model: StockTwinFoundation,
    config: Optional[Dict] = None,
    lookback_days: int = 180
) -> StockDigitalTwin:
    """
    Fine-tune twin for a ticker using Lightning.
    
    Args:
        ticker: Stock ticker
        foundation_model: Pre-loaded foundation model
        config: Configuration dict
        lookback_days: Number of days to look back
    
    Returns:
        Fine-tuned twin
    """
    
    config = config or load_config()
    
    # Get stock characteristics
    stock_chars = get_stock_characteristics(ticker)
    if not stock_chars:
        logger.info(f"Computing stock characteristics for {ticker}")
        engine = get_timescaledb_engine()
        stock_query = f"""
            SELECT * FROM prices
            WHERE ticker = '{ticker}'
            ORDER BY time DESC
            LIMIT {lookback_days}
        """
        prices_df = pd.read_sql(stock_query, engine)
        
        # Fetch market (SPY) data for alpha calculation
        market_query = f"""
            SELECT * FROM prices
            WHERE ticker = 'SPY'
            ORDER BY time DESC
            LIMIT {lookback_days}
        """
        market_df = pd.read_sql(market_query, engine)
        
        stock_chars = compute_stock_characteristics(prices_df, market_df if not market_df.empty else None)
    
    # Training config
    training_config = config.get('training', {}).get('twins', {})
    twin_config = config.get('models', {}).get('twins', {})
    
    # Create data module
    datamodule = TwinDataModule(
        ticker=ticker,
        lookback_days=lookback_days,
        batch_size=training_config.get('batch_size', 32),
        options_enabled=twin_config.get('options_enabled', False)
    )
    
    # Create model
    model = TwinLightningModule(
        foundation_model=foundation_model,
        ticker=ticker,
        stock_characteristics=stock_chars,
        config=config,
        learning_rate=training_config.get('learning_rate', 5e-4),
        weight_decay=training_config.get('weight_decay', 1e-5)
    )
    
    # Setup data module
    datamodule.setup('fit')
    
    # Get checkpoint directory
    _, models_path = setup_colab_environment(config)
    checkpoint_dir = os.path.join(models_path, 'twins', ticker)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create callbacks and logger
    callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        patience=training_config.get('early_stopping_patience', 10),
        filename=f'{ticker}-twin-{{epoch:02d}}-{{val_loss:.4f}}'
    )
    
    logger_obj = create_logger(
        experiment_name=f'twin_training_{ticker}',
        tracking_uri=config.get('mlflow', {}).get('tracking_uri', 'file:./mlruns')
    )
    
    # Create trainer
    trainer = create_trainer(
        max_epochs=training_config.get('fine_tune_epochs', 20),
        callbacks=callbacks,
        logger=logger_obj,
        gradient_clip_val=1.0
    )
    
    # Train
    logger.info(f"Fine-tuning twin for {ticker}")
    trainer.fit(model, datamodule)
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'twin_{datetime.now().strftime("%Y-%m-%d")}.pt')
    model.twin.save_checkpoint(checkpoint_path)
    
    # Also save as latest
    latest_path = os.path.join(checkpoint_dir, 'twin_latest.pt')
    model.twin.save_checkpoint(latest_path)
    
    logger.info(f"Fine-tuning complete for {ticker}. Model saved to {checkpoint_dir}")
    
    return model.twin

