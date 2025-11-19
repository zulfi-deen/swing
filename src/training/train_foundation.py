"""Foundation Model Pre-Training Script

Trains the universal foundation model on all stocks.
Uses PyTorch Lightning for training orchestration.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from typing import Dict, Optional
import logging
import os

from src.models.foundation import StockTwinFoundation
from src.training.synthetic_data import create_synthetic_dataset
from src.utils.config import load_config
from src.utils.colab_utils import setup_colab_environment, is_colab

logger = logging.getLogger(__name__)


class FoundationLoss(nn.Module):
    """Loss function for foundation model pre-training."""
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions: Dict, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            predictions: Dict with 'return' and 'prob' predictions
            targets: Target returns (batch_size,)
        
        Returns:
            Total loss
        """
        
        # Return prediction loss (MSE)
        return_pred = predictions.get('return', torch.zeros_like(targets))
        return_loss = nn.functional.mse_loss(return_pred, targets)
        
        # Hit probability loss (BCE with targets > 0)
        prob_pred = predictions.get('prob', torch.zeros_like(targets))
        hit_targets = (targets > 0).float()
        prob_loss = nn.functional.binary_cross_entropy(prob_pred, hit_targets)
        
        # Combined loss
        total_loss = return_loss + 0.5 * prob_loss
        
        return total_loss


class FoundationTrainingModule(pl.LightningModule):
    """
    PyTorch Lightning module for foundation model training.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.save_hyperparameters()
        self.config = config
        
        # Model
        foundation_config = config.get('models', {}).get('foundation', {})
        self.foundation = StockTwinFoundation(foundation_config)
        
        # Loss function
        self.loss_fn = FoundationLoss()
        
        # Training config
        training_config = config.get('training', {}).get('foundation', {})
        self.learning_rate = training_config.get('learning_rate', 1e-3)
        self.weight_decay = training_config.get('weight_decay', 1e-5)
    
    def forward(self, batch: Dict, graph: Optional[torch.Tensor] = None):
        """Forward pass."""
        return self.foundation.pretrain_forward(batch, graph)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        predictions = self(batch['features'], batch.get('graph'))
        loss = self.loss_fn(predictions, batch['targets'])
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        predictions = self(batch['features'], batch.get('graph'))
        loss = self.loss_fn(predictions, batch['targets'])
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.get('training', {}).get('foundation', {}).get('num_epochs', 100)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


def create_tft_dataset_from_db(
    tickers: list,
    start_date: str,
    end_date: str,
    max_encoder_length: int = 60,
    max_prediction_length: int = 5
):
    """
    Create pytorch-forecasting TimeSeriesDataSet from TimescaleDB.
    
    Returns:
        Training and validation TimeSeriesDataSet objects
    """
    try:
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
    except ImportError:
        logger.error("pytorch-forecasting not installed. Run: pip install pytorch-forecasting")
        return None, None
    
    from src.data.storage import get_timescaledb_engine
    import pandas as pd
    
    logger.info(f"Loading data from TimescaleDB for {len(tickers)} tickers")
    
    engine = get_timescaledb_engine()
    
    # Load prices and features
    ticker_list = "', '".join(tickers)
    query = f"""
        SELECT 
            p.time,
            p.ticker,
            p.close,
            p.volume,
            p.open,
            p.high,
            p.low,
            f.rsi_14,
            f.macd,
            f.atr_14,
            f.volume_z_score,
            f.sentiment_score,
            sc.beta,
            sc.sector,
            sc.liquidity_regime
        FROM prices p
        LEFT JOIN features f ON p.time = f.time AND p.ticker = f.ticker
        LEFT JOIN stock_characteristics sc ON p.ticker = sc.ticker
        WHERE p.ticker IN ('{ticker_list}')
        AND p.time >= '{start_date}' AND p.time <= '{end_date}'
        ORDER BY p.ticker, p.time
    """
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        logger.error("No data loaded from database")
        return None, None
    
    # Prepare data for TimeSeriesDataSet
    df['time_idx'] = (df['time'] - df['time'].min()).dt.days
    df['target'] = df.groupby('ticker')['close'].pct_change(periods=5).shift(-5)  # 5-day forward return
    df = df.dropna(subset=['target'])
    
    # Fill missing features
    df['rsi_14'] = df['rsi_14'].fillna(50)
    df['macd'] = df['macd'].fillna(0)
    df['atr_14'] = df['atr_14'].fillna(0.02)
    df['volume_z_score'] = df['volume_z_score'].fillna(0)
    df['sentiment_score'] = df['sentiment_score'].fillna(0.5)
    df['beta'] = df['beta'].fillna(1.0)
    df['liquidity_regime'] = df['liquidity_regime'].fillna(1)
    
    # Split train/val (80/20)
    split_time = df['time_idx'].quantile(0.8)
    train_df = df[df['time_idx'] <= split_time]
    val_df = df[df['time_idx'] > split_time]
    
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # Create TimeSeriesDataSet
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target",
        group_ids=["ticker"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["ticker", "sector"],
        static_reals=["beta"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "target",
            "close",
            "volume",
            "rsi_14",
            "macd",
            "atr_14",
            "volume_z_score",
            "sentiment_score"
        ],
        target_normalizer=GroupNormalizer(groups=["ticker"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)
    
    return training, validation


def train_foundation_model(
    config_path: Optional[str] = None,
    use_synthetic_data: bool = False
):
    """
    Train foundation model.
    
    Args:
        config_path: Path to config file
        use_synthetic_data: Whether to use synthetic data (for demo)
    """
    
    config = load_config(config_path)
    
    # Setup Colab environment if needed
    data_path, models_path = setup_colab_environment(config)
    logger.info(f"Using data_path: {data_path}, models_path: {models_path}")
    
    # Update checkpoint path based on environment
    if is_colab():
        foundation_config = config.get('models', {}).get('foundation', {})
        foundation_config['checkpoint_path'] = os.path.join(models_path, 'foundation', 'foundation_v1.0.pt')
        os.makedirs(os.path.dirname(foundation_config['checkpoint_path']), exist_ok=True)
    
    # Load data and create TFT dataset
    tickers = config.get('models', {}).get('twins', {}).get('pilot_tickers', [])
    if not tickers:
        # Load S&P 500 tickers (or subset for faster training)
        from src.utils.tickers import get_sp500_tickers
        tickers = get_sp500_tickers()[:100]  # Use first 100 for faster training
    
    logger.info(f"Training on {len(tickers)} tickers")
    
    # Create TFT dataset
    training_dataset, validation_dataset = create_tft_dataset_from_db(
        tickers,
        start_date='2022-01-01',
        end_date='2024-12-31',
        max_encoder_length=config.get('models', {}).get('foundation', {}).get('tft', {}).get('max_encoder_length', 60),
        max_prediction_length=config.get('models', {}).get('foundation', {}).get('tft', {}).get('max_prediction_length', 5)
    )
    
    if training_dataset is None:
        logger.error("Failed to create TFT dataset. Check database and data availability.")
        return
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    batch_size = config.get('training', {}).get('foundation', {}).get('batch_size', 128)
    
    train_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders: train batches={len(train_dataloader)}, val batches={len(val_dataloader)}")
    
    # Training setup
    training_config = config.get('training', {}).get('foundation', {})
    
    model = FoundationTrainingModule(config)
    
    # CRITICAL: Initialize TFT from dataset
    logger.info("Initializing Foundation TFT from dataset...")
    model.foundation.initialize_tft(training_dataset)
    logger.info("TFT initialized successfully")
    
    # Get models path for checkpoint directory
    _, models_path = setup_colab_environment(config)
    checkpoint_dir = os.path.join(models_path, 'foundation')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='foundation-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=training_config.get('early_stopping_patience', 15),
        mode='min'
    )
    
    # Logger
    mlflow_logger = MLFlowLogger(
        experiment_name='foundation_training',
        tracking_uri=config.get('mlflow', {}).get('tracking_uri', 'file:./mlruns')
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=training_config.get('num_epochs', 100),
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=mlflow_logger,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=10,
        gradient_clip_val=1.0
    )
    
    # Train
    logger.info("Starting foundation model training...")
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, 'foundation_v1.0.pt')
    torch.save({
        'model_state_dict': model.foundation.state_dict(),
        'config': config,
        'tft_initialized': True,
    }, final_checkpoint_path)
    
    logger.info(f"Foundation training complete. Model saved to {final_checkpoint_path}")


if __name__ == '__main__':
    train_foundation_model(use_synthetic_data=True)


