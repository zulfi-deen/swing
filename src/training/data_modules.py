"""PyTorch Lightning DataModules for training"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, List, Dict
import logging
import pandas as pd
import torch
from torch_geometric.data import Data

from src.training.dataset import create_tft_dataset, prepare_dataframe_for_tft
from src.data.storage import get_timescaledb_engine

logger = logging.getLogger(__name__)


class FoundationDataModule(pl.LightningDataModule):
    """
    DataModule for foundation model training.
    Handles TFT dataset creation and initialization.
    """
    
    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        max_encoder_length: int = 60,
        max_prediction_length: int = 5,
        batch_size: int = 128,
        num_workers: int = 2,
        pin_memory: bool = True,
        use_synthetic: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_synthetic = use_synthetic
        
        self.training_dataset = None
        self.validation_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Create TFT datasets."""
        if stage == "fit" or stage is None:
            if self.use_synthetic:
                from src.training.synthetic_data import create_synthetic_dataset
                train_df, val_df = create_synthetic_dataset(
                    self.tickers,
                    self.start_date,
                    self.end_date
                )
            else:
                train_df, val_df = self._load_from_db()
            
            if train_df is None or val_df is None:
                raise ValueError("Failed to load training data")
            
            # Create TFT datasets using pytorch-forecasting directly
            try:
                from pytorch_forecasting import TimeSeriesDataSet
                from pytorch_forecasting.data import GroupNormalizer
            except ImportError:
                raise ImportError("pytorch-forecasting not installed")
            
            # Create training dataset
            self.training_dataset = TimeSeriesDataSet(
                train_df,
                time_idx="time_idx",
                target="target",
                group_ids=["ticker"],
                min_encoder_length=self.max_encoder_length // 2,
                max_encoder_length=self.max_encoder_length,
                min_prediction_length=1,
                max_prediction_length=self.max_prediction_length,
                static_categoricals=["ticker", "sector"] if "sector" in train_df.columns else ["ticker"],
                static_reals=["beta"] if "beta" in train_df.columns else None,
                time_varying_known_reals=["time_idx"],
                time_varying_unknown_categoricals=[],
                time_varying_unknown_reals=[
                    col for col in ["target", "close", "volume", "rsi_14", "macd", "atr_14", 
                                   "volume_z_score", "sentiment_score"] if col in train_df.columns
                ],
                target_normalizer=GroupNormalizer(groups=["ticker"]),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )
            
            # Create validation dataset
            self.validation_dataset = TimeSeriesDataSet.from_dataset(
                self.training_dataset, 
                val_df, 
                predict=True, 
                stop_randomization=True
            )
            
            logger.info(f"Created datasets: train={len(train_df)}, val={len(val_df)}")
    
    def _load_from_db(self):
        """Load data from TimescaleDB."""
        try:
            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting.data import GroupNormalizer
        except ImportError:
            logger.error("pytorch-forecasting not installed")
            return None, None
        
        engine = get_timescaledb_engine()
        
        ticker_list = "', '".join(self.tickers)
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
            AND p.time >= '{self.start_date}' AND p.time <= '{self.end_date}'
            ORDER BY p.ticker, p.time
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            logger.error("No data loaded from database")
            return None, None
        
        # Prepare data
        df['time_idx'] = (pd.to_datetime(df['time']) - pd.to_datetime(df['time']).min()).dt.days
        df['target'] = df.groupby('ticker')['close'].pct_change(periods=5).shift(-5)
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
        
        return train_df, val_df
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )


class TwinDataModule(pl.LightningDataModule):
    """
    DataModule for digital twin fine-tuning.
    Handles per-stock dataset creation with optional options features.
    """
    
    def __init__(
        self,
        ticker: str,
        lookback_days: int = 180,
        batch_size: int = 32,
        num_workers: int = 0,
        options_enabled: bool = False,
        val_split: float = 0.2
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.ticker = ticker
        self.lookback_days = lookback_days
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.options_enabled = options_enabled
        self.val_split = val_split
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Load and prepare stock data."""
        if stage == "fit" or stage is None:
            from datetime import datetime, timedelta
            from torch.utils.data import Dataset
            
            class TwinDataset(Dataset):
                """Dataset for twin fine-tuning."""
                
                def __init__(self, features_df: pd.DataFrame, targets: pd.Series):
                    self.features_df = features_df
                    self.targets = targets
                
                def __len__(self):
                    return len(self.features_df)
                
                def __getitem__(self, idx):
                    features = torch.tensor(self.features_df.iloc[idx].values, dtype=torch.float32)
                    target = torch.tensor(self.targets.iloc[idx], dtype=torch.float32)
                    return features, target
            
            engine = get_timescaledb_engine()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            # Load prices and features
            query = f"""
                SELECT 
                    p.time,
                    p.close,
                    p.volume,
                    f.*
                FROM prices p
                LEFT JOIN features f ON p.time = f.time AND p.ticker = f.ticker
                WHERE p.ticker = '{self.ticker}'
                AND p.time >= '{start_date}' AND p.time <= '{end_date}'
                ORDER BY p.time
            """
            
            df = pd.read_sql(query, engine)
            
            if df.empty:
                raise ValueError(f"No data found for {self.ticker}")
            
            # Calculate target
            df['return_5d'] = df['close'].pct_change(periods=5).shift(-5)
            df = df.dropna(subset=['return_5d'])
            
            # Split train/val
            split_idx = int(len(df) * (1 - self.val_split))
            train_df = df.iloc[:split_idx]
            val_df = df.iloc[split_idx:]
            
            # Prepare features
            feature_cols = [col for col in df.columns if col not in ['time', 'ticker', 'return_5d', 'close']]
            train_features = train_df[feature_cols].fillna(0)
            train_targets = train_df['return_5d']
            
            val_features = val_df[feature_cols].fillna(0)
            val_targets = val_df['return_5d']
            
            self.train_dataset = TwinDataset(train_features, train_targets)
            self.val_dataset = TwinDataset(val_features, val_targets)
            
            logger.info(f"Created datasets for {self.ticker}: train={len(train_df)}, val={len(val_df)}")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class LightGBMDataModule(pl.LightningDataModule):
    """
    DataModule for LightGBM cross-sectional ranker.
    Handles feature engineering and temporal splits.
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        val_split: float = 0.2,
        feature_cols: Optional[List[str]] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.start_date = start_date
        self.end_date = end_date
        self.val_split = val_split
        self.feature_cols = feature_cols or [
            'rsi_14', 'macd', 'macd_signal', 'atr_14',
            'volume_z_score', 'return_rank_5d', 'return_rank_20d',
            'sector_relative_strength', 'correlation_to_spy',
            'sentiment_score', 'pattern_confidence', 'beta', 'liquidity_regime'
        ]
        
        self.train_df = None
        self.val_df = None
    
    def setup(self, stage: Optional[str] = None):
        """Load and prepare cross-sectional data."""
        if stage == "fit" or stage is None:
            engine = get_timescaledb_engine()
            
            query = f"""
                SELECT 
                    f.time,
                    f.ticker,
                    p.close,
                    f.rsi_14,
                    f.macd,
                    f.macd_signal,
                    f.atr_14,
                    f.volume_z_score,
                    f.sentiment_score,
                    f.pattern_confidence,
                    f.return_rank_5d,
                    f.return_rank_20d,
                    f.sector_relative_strength,
                    f.correlation_to_spy,
                    sc.beta,
                    sc.liquidity_regime
                FROM features f
                INNER JOIN prices p ON f.time = p.time AND f.ticker = p.ticker
                LEFT JOIN stock_characteristics sc ON f.ticker = sc.ticker
                WHERE f.time >= '{self.start_date}' AND f.time <= '{self.end_date}'
                ORDER BY f.time, f.ticker
            """
            
            df = pd.read_sql(query, engine)
            
            if df.empty:
                raise ValueError("No data loaded from database")
            
            # Compute target
            df['return_5d'] = df.groupby('ticker')['close'].pct_change(periods=5).shift(-5)
            df = df.dropna(subset=['return_5d'])
            
            # Fill missing features
            for col in self.feature_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            # Split train/val by time
            split_date = pd.to_datetime(df['time']).quantile(1 - self.val_split)
            self.train_df = df[pd.to_datetime(df['time']) <= split_date].copy()
            self.val_df = df[pd.to_datetime(df['time']) > split_date].copy()
            
            logger.info(f"Created datasets: train={len(self.train_df)}, val={len(self.val_df)}")
    
    def get_train_data(self):
        """Get training data as (X, y, groups)."""
        available_cols = [col for col in self.feature_cols if col in self.train_df.columns]
        X = self.train_df[available_cols]
        y = self.train_df['return_5d']
        groups = self.train_df.groupby('time').size().values
        return X, y, groups
    
    def get_val_data(self):
        """Get validation data as (X, y, groups)."""
        available_cols = [col for col in self.feature_cols if col in self.val_df.columns]
        X = self.val_df[available_cols]
        y = self.val_df['return_5d']
        groups = self.val_df.groupby('time').size().values
        return X, y, groups

