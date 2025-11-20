"""LightGBM Cross-Sectional Ranker Training with PyTorch Lightning"""

import pytorch_lightning as pl
import torch
from typing import Dict, Optional
import logging
import os
import joblib
from pathlib import Path

from src.models.ensemble import LightGBMRanker
from src.training.data_modules import LightGBMDataModule
from src.training.lightning_utils import create_callbacks, create_logger, create_trainer
from src.utils.config import load_config
# Legacy colab_utils removed - using config paths directly

logger = logging.getLogger(__name__)


class LightGBMLightningModule(pl.LightningModule):
    """
    Lightning module wrapper for LightGBM training.
    Allows LightGBM to benefit from Lightning's orchestration and logging.
    """
    
    def __init__(
        self,
        config: Dict,
        model_version: str = 'v1.0'
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['config'])
        self.config = config
        self.model_version = model_version
        
        # LightGBM config
        lgbm_config = config.get('models', {}).get('lightgbm', {})
        self.ranker = LightGBMRanker(**lgbm_config)
        
        # Training state
        self.trained = False
    
    def training_step(self, batch, batch_idx):
        """
        Training step - LightGBM training happens in on_train_start.
        This step is a placeholder for compatibility.
        """
        # LightGBM training is done in on_train_start
        # This step just logs that we're using LightGBM
        if batch_idx == 0:
            self.log('model_type', 'lightgbm', on_step=False, on_epoch=True)
        return torch.tensor(0.0, requires_grad=False)
    
    def on_train_start(self):
        """Train LightGBM model when training starts."""
        if self.trained:
            return
        
        datamodule = self.trainer.datamodule
        if datamodule is None:
            logger.warning("No datamodule available for LightGBM training")
            return
        
        logger.info("Training LightGBM ranker...")
        
        # Get training data
        X_train, y_train, train_groups = datamodule.get_train_data()
        X_val, y_val, val_groups = datamodule.get_val_data()
        
        # Train
        self.ranker.fit(
            X_train,
            y_train,
            groups=train_groups,
            eval_set=[(X_val, y_val)],
            eval_groups=[val_groups]
        )
        
        self.trained = True
        
        # Log feature importance
        feature_importance = self.ranker.get_feature_importance()
        logger.info(f"Top 10 features:\n{feature_importance.head(10)}")
        
        # Log metrics
        self.log('num_features', len(feature_importance), on_step=False, on_epoch=True)
        if not feature_importance.empty:
            self.log('top_feature_importance', feature_importance['importance'].iloc[0], on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        """No optimizer needed for LightGBM."""
        return None


def train_lightgbm_ranker(
    config: Optional[Dict] = None,
    start_date: str = '2022-01-01',
    end_date: str = '2024-12-31',
    model_version: str = 'v1.0',
    output_dir: str = 'models/ensemble'
):
    """
    Train LightGBM ranker using Lightning.
    
    Args:
        config: Configuration dict
        start_date: Training start date
        end_date: Training end date
        model_version: Model version string
        output_dir: Output directory for saved model
    """
    
    config = config or load_config()
    
    # Create data module
    datamodule = LightGBMDataModule(
        start_date=start_date,
        end_date=end_date,
        val_split=0.2
    )
    
    # Create model
    model = LightGBMLightningModule(
        config=config,
        model_version=model_version
    )
    
    # Setup data module
    datamodule.setup('fit')
    
    # Get output directory from config
    storage_config = config.get('storage', {}).get('local', {})
    models_path = storage_config.get('models_dir', 'models/')
    if not os.path.isabs(models_path):
        models_path = os.path.abspath(models_path)
    output_dir = os.path.join(models_path, 'ensemble')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger_obj = create_logger(
        experiment_name='lightgbm_training',
        tracking_uri=config.get('mlflow', {}).get('tracking_uri', 'file:./mlruns')
    )
    
    # Create trainer (LightGBM doesn't need many epochs, just one to trigger training)
    trainer = create_trainer(
        max_epochs=1,
        callbacks=[],  # No callbacks needed for LightGBM
        logger=logger_obj,
        enable_progress_bar=True
    )
    
    # Create dummy dataloader (LightGBM training happens in on_train_start)
    from torch.utils.data import DataLoader, Dataset
    
    class DummyDataset(Dataset):
        def __len__(self):
            return 1
        
        def __getitem__(self, idx):
            return idx
    
    dummy_dataset = DummyDataset()
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=1)
    
    # Train (this triggers on_train_start which trains LightGBM)
    logger.info("Starting LightGBM training...")
    trainer.fit(model, dummy_dataloader)
    
    # Save model
    model_path = os.path.join(output_dir, f'lgbm_ranker_{model_version}.pkl')
    joblib.dump(model.ranker, model_path)
    
    logger.info(f"LightGBM training complete. Model saved to {model_path}")
    
    return model.ranker

