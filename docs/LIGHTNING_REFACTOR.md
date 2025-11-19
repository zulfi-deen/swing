# PyTorch Lightning Refactor Summary

This document summarizes the complete refactor of all training code to use PyTorch Lightning.

## Overview

All training scripts have been refactored to use PyTorch Lightning for:
- Unified training orchestration
- Automatic checkpointing and early stopping
- MLflow/TensorBoard logging
- Gradient clipping and mixed precision
- Multi-GPU support (when available)
- Reproducible experiments

## Architecture Changes

### Before
- Manual training loops with custom checkpointing
- Inconsistent logging across models
- Ad-hoc data loading
- No unified CLI

### After
- Lightning modules for all models
- Lightning DataModules for data loading
- Shared utilities for common patterns
- Unified CLI via `scripts/train.py`

## New Components

### 1. Shared Utilities (`src/training/lightning_utils.py`)

Provides factory functions for:
- `create_callbacks()`: Model checkpointing, early stopping, LR monitoring
- `create_logger()`: MLflow or TensorBoard loggers
- `create_trainer()`: Standardized trainer configuration

### 2. Data Modules (`src/training/data_modules.py`)

Lightning DataModules for each training type:
- `FoundationDataModule`: TFT dataset creation from TimescaleDB or synthetic data
- `TwinDataModule`: Per-stock dataset preparation for twin fine-tuning
- `LightGBMDataModule`: Cross-sectional feature engineering and temporal splits

### 3. Lightning Modules

#### Foundation Training (`src/training/train_foundation.py`)
- `FoundationTrainingModule`: Handles TFT initialization and training
- Uses `FoundationDataModule` for data loading
- Automatic TFT initialization from dataset

#### Twin Training (`src/training/train_twins_lightning.py`)
- `TwinLightningModule`: Fine-tunes digital twins
- Uses `TwinDataModule` for per-stock data
- Logs per-ticker metrics (MAE, hit rate, etc.)

#### RL Training (`src/training/train_rl_portfolio_lightning.py`)
- `RLPortfolioModule`: Manages episode rollouts and PPO updates
- Integrates with `TradingEnvironment`
- Logs episode returns, portfolio values, and PPO losses

#### LightGBM Training (`src/training/train_lightgbm_lightning.py`)
- `LightGBMLightningModule`: Wraps LightGBM for Lightning orchestration
- Uses `LightGBMDataModule` for feature engineering
- Benefits from Lightning's logging and checkpointing

### 4. Unified CLI (`scripts/train.py`)

Single entry point for all training:

```bash
python scripts/train.py foundation --use-synthetic
python scripts/train.py twin --ticker AAPL
python scripts/train.py rl --num-episodes 1000
python scripts/train.py lightgbm --start-date 2022-01-01
```

## Migration Guide

### Old Training Scripts

Legacy scripts in `src/training/` now delegate to Lightning implementations:
- `train_foundation.py` → Uses `FoundationDataModule` and `FoundationTrainingModule`
- `train_twins.py` → Delegates to `train_twins_lightning.py`
- `train_rl_portfolio.py` → Delegates to `train_rl_portfolio_lightning.py`
- `train_lightgbm.py` → Delegates to `train_lightgbm_lightning.py`
- `train.py` → Deprecated, use `scripts/train.py`

### Weekly Retraining

`src/training/weekly_retrain.py` has been updated to use Lightning-based twin fine-tuning. The Prefect flow now:
1. Loads foundation model
2. Calls `fine_tune_twin()` which uses Lightning internally
3. Collects checkpoint artifacts from trainer's output directory

## Benefits

1. **Consistency**: All models use the same training infrastructure
2. **Reproducibility**: Hyperparameters saved automatically via `save_hyperparameters()`
3. **Observability**: Unified logging to MLflow/TensorBoard
4. **Maintainability**: Shared utilities reduce code duplication
5. **Scalability**: Easy to add multi-GPU, mixed precision, etc.
6. **Testing**: Lightning modules are easier to test in isolation

## Configuration

Training configuration remains in `config/config.yaml` under:
- `training.foundation`: Foundation model training params
- `training.twins`: Twin fine-tuning params
- `rl_portfolio.ppo`: PPO hyperparameters
- `models.lightgbm`: LightGBM params

Lightning-specific settings (accelerator, devices, precision) can be set via:
- Trainer arguments in code
- Environment variables
- Future: LightningCLI config files

## Future Enhancements

Potential improvements:
1. **LightningCLI**: Use Lightning's CLI for config-driven training
2. **Distributed Training**: Multi-GPU support via Lightning strategies
3. **Mixed Precision**: Automatic mixed precision training
4. **Hyperparameter Tuning**: Integration with Optuna/Ray Tune
5. **Model Registry**: MLflow model registry integration

## Testing

All Lightning modules can be tested with:
- Synthetic data modules
- Dry-run training (1-2 epochs)
- Unit tests for forward passes
- Integration tests for full training loops

See `tests/test_foundation.py` and `tests/test_twin_manager.py` for examples.

