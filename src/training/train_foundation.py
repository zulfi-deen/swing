"""Foundation Model Pre-Training Script

Trains the universal foundation model on all stocks.
Uses PyTorch Lightning for training orchestration.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from typing import Dict, Optional
import logging
import os

from src.models.foundation import StockTwinFoundation
from src.training.data_modules import FoundationDataModule
from src.training.lightning_utils import create_callbacks, create_logger, create_trainer
from src.utils.config import load_config

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
    
    def __init__(
        self,
        config: Dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        num_epochs: int = 100
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['config'])
        self.config = config
        
        # Model
        foundation_config = config.get('models', {}).get('foundation', {})
        self.foundation = StockTwinFoundation(foundation_config)
        
        # Loss function
        self.loss_fn = FoundationLoss()
        
        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        
        # TFT initialization flag
        self.tft_initialized = False
    
    def on_train_start(self):
        """Initialize TFT from dataset when training starts."""
        if not self.tft_initialized:
            datamodule = self.trainer.datamodule
            if datamodule and datamodule.training_dataset:
                logger.info("Initializing Foundation TFT from dataset...")
                self.foundation.initialize_tft(datamodule.training_dataset)
                self.tft_initialized = True
                logger.info("TFT initialized successfully")
    
    def forward(self, batch, graph: Optional[torch.Tensor] = None):
        """Forward pass."""
        # Handle TFT batch format
        if isinstance(batch, dict):
            return self.foundation.pretrain_forward(batch, graph)
        else:
            # TFT TimeSeriesDataSet returns batch directly
            return self.foundation.pretrain_forward({'features': batch}, graph)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # TFT dataset returns (x, y) tuple
        if isinstance(batch, tuple):
            x, y = batch
            predictions = self(x)
            targets = y
        else:
            predictions = self(batch)
            targets = batch.get('targets', batch.get('y'))
        
        # Extract targets from TFT format if needed
        if isinstance(targets, dict):
            targets = targets.get('target', targets.get('return_5d'))
        
        loss = self.loss_fn(predictions, targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if isinstance(batch, tuple):
            x, y = batch
            predictions = self(x)
            targets = y
        else:
            predictions = self(batch)
            targets = batch.get('targets', batch.get('y'))
        
        if isinstance(targets, dict):
            targets = targets.get('target', targets.get('return_5d'))
        
        loss = self.loss_fn(predictions, targets)
        
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
            T_max=self.num_epochs
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


def train_foundation_model(
    config_path: Optional[str] = None,
    use_synthetic_data: bool = False
):
    """
    Train foundation model using Lightning.
    
    Args:
        config_path: Path to config file
        use_synthetic_data: Whether to use synthetic data (for demo)
    """
    
    config = load_config(config_path)
    
    # Setup Colab environment if needed
    # Get paths from config
    storage_config = config.get('storage', {}).get('local', {})
    models_path = storage_config.get('models_dir', 'models/')
    if not os.path.isabs(models_path):
        models_path = os.path.abspath(models_path)
    logger.info(f"Using models_path: {models_path}")
    
    # Get tickers
    tickers = config.get('models', {}).get('twins', {}).get('pilot_tickers', [])
    if not tickers:
        from src.utils.tickers import get_sp500_tickers
        tickers = get_sp500_tickers()[:100]
    
    logger.info(f"Training on {len(tickers)} tickers")
    
    # Training config
    training_config = config.get('training', {}).get('foundation', {})
    foundation_config = config.get('models', {}).get('foundation', {})
    
    # Create data module
    datamodule = FoundationDataModule(
        tickers=tickers,
        start_date='2022-01-01',
        end_date='2024-12-31',
        max_encoder_length=foundation_config.get('tft', {}).get('max_encoder_length', 60),
        max_prediction_length=foundation_config.get('tft', {}).get('max_prediction_length', 5),
        batch_size=training_config.get('batch_size', 128),
        use_synthetic=use_synthetic_data
    )
    
    # Create model
    model = FoundationTrainingModule(
        config=config,
        learning_rate=training_config.get('learning_rate', 1e-3),
        weight_decay=training_config.get('weight_decay', 1e-5),
        num_epochs=training_config.get('num_epochs', 100)
    )
    
    # Setup data module (this creates datasets)
    datamodule.setup('fit')
    
    # Initialize TFT from dataset
    logger.info("Initializing Foundation TFT from dataset...")
    model.foundation.initialize_tft(datamodule.training_dataset)
    model.tft_initialized = True
    logger.info("TFT initialized successfully")
    
    # Get checkpoint directory
    checkpoint_dir = os.path.join(models_path, 'foundation')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create callbacks and logger
    callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        patience=training_config.get('early_stopping_patience', 15),
        filename='foundation-{epoch:02d}-{val_loss:.4f}'
    )
    
    logger_obj = create_logger(
        experiment_name='foundation_training',
        tracking_uri=config.get('mlflow', {}).get('tracking_uri', 'file:./mlruns')
    )
    
    # Create trainer
    trainer = create_trainer(
        max_epochs=training_config.get('num_epochs', 100),
        callbacks=callbacks,
        logger=logger_obj,
        gradient_clip_val=1.0
    )
    
    # Train
    logger.info("Starting foundation model training...")
    trainer.fit(model, datamodule)
    
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


