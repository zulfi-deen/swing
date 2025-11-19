"""Per-Stock Twin Fine-Tuning Script

Fine-tunes stock-specific digital twins on last 6 months of data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional
import logging
import os
from datetime import datetime, timedelta

from src.models.foundation import StockTwinFoundation, load_foundation_model
from src.models.digital_twin import StockDigitalTwin
from src.training.train_twins_lightning import fine_tune_twin as fine_tune_twin_lightning
from src.data.storage import get_stock_characteristics, compute_stock_characteristics, get_timescaledb_engine
from src.utils.config import load_config
from src.utils.colab_utils import setup_colab_environment, is_colab
import pandas as pd

logger = logging.getLogger(__name__)


class TwinDataset(Dataset):
    """Dataset for twin fine-tuning."""
    
    def __init__(self, features_df: pd.DataFrame, targets: pd.Series):
        self.features_df = features_df
        self.targets = targets
    
    def __len__(self):
        return len(self.features_df)
    
    def __getitem__(self, idx):
        # Simplified - would need proper feature extraction
        features = torch.tensor(self.features_df.iloc[idx].values, dtype=torch.float32)
        target = torch.tensor(self.targets.iloc[idx], dtype=torch.float32)
        return features, target


class TwinFineTuner:
    """Fine-tuner for stock-specific twins."""
    
    def __init__(
        self,
        foundation_model: StockTwinFoundation,
        ticker: str,
        stock_characteristics: Dict,
        config: Optional[Dict] = None
    ):
        self.foundation = foundation_model
        self.ticker = ticker
        self.stock_characteristics = stock_characteristics
        self.config = config or load_config()
        
        # Initialize twin
        twin_config = self.config.get('models', {}).get('twins', {})
        self.twin = StockDigitalTwin(
            foundation_model,
            ticker,
            stock_characteristics,
            twin_config
        )
        
        # Training config
        training_config = self.config.get('training', {}).get('twins', {})
        self.learning_rate = training_config.get('learning_rate', 5e-4)
        self.weight_decay = training_config.get('weight_decay', 1e-5)
        self.epochs = training_config.get('fine_tune_epochs', 20)
    
    def fine_tune(
        self,
        stock_data: pd.DataFrame,
        epochs: Optional[int] = None,
        device: str = 'cpu'
    ) -> Dict:
        """
        Fine-tune twin on stock data.
        
        Args:
            stock_data: DataFrame with features and targets
            epochs: Number of epochs (overrides config)
            device: Device to train on
        
        Returns:
            Training metrics
        """
        
        epochs = epochs or self.epochs
        
        # Prepare data
        # In production, would properly format for TFT
        # For now, simplified training loop
        
        self.twin.train()
        self.twin.to(device)
        
        optimizer = torch.optim.AdamW(
            self.twin.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Simplified loss (would need proper targets)
        criterion = nn.MSELoss()
        
        train_losses = []
        
        for epoch in range(epochs):
            # Simplified training step
            # In production, would iterate over DataLoader
            
            # Placeholder: would use actual batch
            batch = {'features': torch.randn(1, 60, 10)}  # Placeholder
            graph = None
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.twin(batch, graph)
            
            # Placeholder target
            target_return = torch.tensor([0.01], device=device)
            
            # Loss
            loss = criterion(predictions['expected_return'], target_return)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"{self.ticker} - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        self.twin.eval()
        
        return {
            'final_loss': train_losses[-1] if train_losses else 0.0,
            'train_losses': train_losses
        }
    
    def validate(self, test_data: pd.DataFrame) -> Dict:
        """
        Validate twin on test data.
        
        Args:
            test_data: Test DataFrame
        
        Returns:
            Validation metrics
        """
        
        self.twin.eval()
        
        # Placeholder validation
        # In production, would compute proper metrics
        
        return {
            'mae': 0.0,
            'hit_rate': 0.0,
            'regime_accuracy': 0.0
        }
    
    def save(self, checkpoint_dir: str):
        """Save twin checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'twin_{datetime.now().strftime("%Y-%m-%d")}.pt')
        self.twin.save_checkpoint(checkpoint_path)
        
        # Also save as latest
        latest_path = os.path.join(checkpoint_dir, 'twin_latest.pt')
        self.twin.save_checkpoint(latest_path)
        
        logger.info(f"Saved twin for {self.ticker} to {checkpoint_dir}")


def fine_tune_twin(
    ticker: str,
    foundation_model: StockTwinFoundation,
    config: Optional[Dict] = None,
    lookback_days: int = 180
) -> StockDigitalTwin:
    """
    Fine-tune twin for a ticker (delegates to Lightning implementation).
    
    Args:
        ticker: Stock ticker
        foundation_model: Pre-loaded foundation model
        config: Configuration dict
        lookback_days: Number of days to look back
    
    Returns:
        Fine-tuned twin
    """
    # Use Lightning-based fine-tuning
    return fine_tune_twin_lightning(ticker, foundation_model, config, lookback_days)


