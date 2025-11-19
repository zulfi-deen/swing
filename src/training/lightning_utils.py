"""Shared PyTorch Lightning utilities and factories"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from typing import Optional, Dict, List
import logging
import os

logger = logging.getLogger(__name__)


def create_callbacks(
    checkpoint_dir: str,
    monitor: str = "val_loss",
    mode: str = "min",
    save_top_k: int = 3,
    patience: int = 15,
    filename: Optional[str] = None,
    enable_lr_monitor: bool = True
) -> List[pl.Callback]:
    """
    Create standard Lightning callbacks.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        monitor: Metric to monitor
        mode: 'min' or 'max'
        save_top_k: Number of best models to save
        patience: Early stopping patience
        filename: Checkpoint filename pattern
        enable_lr_monitor: Whether to log learning rate
    
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=checkpoint_dir,
        filename=filename or f"{{epoch:02d}}-{{{monitor}:.4f}}",
        save_top_k=save_top_k,
        mode=mode,
        save_last=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    if enable_lr_monitor:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
    
    return callbacks


def create_logger(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    use_mlflow: bool = True,
    log_dir: str = "lightning_logs"
) -> pl.loggers.Logger:
    """
    Create Lightning logger (MLflow or TensorBoard).
    
    Args:
        experiment_name: Experiment name
        tracking_uri: MLflow tracking URI (if None, uses file:./mlruns)
        use_mlflow: Whether to use MLflow (else TensorBoard)
        log_dir: Directory for TensorBoard logs
    
    Returns:
        Lightning logger
    """
    if use_mlflow:
        try:
            return MLFlowLogger(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri or "file:./mlruns"
            )
        except Exception as e:
            logger.warning(f"Failed to create MLflow logger: {e}, falling back to TensorBoard")
            use_mlflow = False
    
    if not use_mlflow:
        return TensorBoardLogger(
            save_dir=log_dir,
            name=experiment_name
        )


def create_trainer(
    max_epochs: int = 100,
    accelerator: str = "auto",
    devices: str = "auto",
    precision: str = "32",
    gradient_clip_val: Optional[float] = 1.0,
    log_every_n_steps: int = 10,
    callbacks: Optional[List[pl.Callback]] = None,
    logger: Optional[pl.loggers.Logger] = None,
    enable_progress_bar: bool = True,
    **kwargs
) -> pl.Trainer:
    """
    Create Lightning trainer with standard defaults.
    
    Args:
        max_epochs: Maximum training epochs
        accelerator: Accelerator type ('auto', 'gpu', 'cpu', etc.)
        devices: Device specification
        precision: Training precision ('32', '16', 'bf16')
        gradient_clip_val: Gradient clipping value
        log_every_n_steps: Logging frequency
        callbacks: Additional callbacks
        logger: Logger instance
        enable_progress_bar: Whether to show progress bar
        **kwargs: Additional trainer arguments
    
    Returns:
        Configured Trainer instance
    """
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks or [],
        logger=logger,
        enable_progress_bar=enable_progress_bar,
        **kwargs
    )
    
    return trainer

