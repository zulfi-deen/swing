# Colab Training Guide

This guide explains how to train models in Google Colab while using your local MacBook for storage, databases, and inference.

## Architecture Overview

- **Local MacBook**: Storage, databases (TimescaleDB, Redis), and inference
- **Google Colab**: Model training with GPU acceleration
- **Google Drive**: Data synchronization between MacBook and Colab

## Setup

### 1. Local MacBook Setup

Ensure your local environment is configured:

```bash
# Install dependencies
pip install -r requirements.txt

# Configure databases (TimescaleDB, Redis)
docker-compose up -d

# Set up configuration
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your settings
```

### 2. Google Drive Setup

1. Create a folder in Google Drive: `swing_trading`
2. Ensure Google Drive is syncing on your MacBook
3. Note the path to your Google Drive folder (e.g., `/Users/username/Google Drive/swing_trading`)

### 3. Export Training Data

On your MacBook, export training data to Google Drive:

```bash
# Export training data (defaults to pilot tickers, 3 years of data)
python scripts/export_training_data.py

# Or specify custom parameters
python scripts/export_training_data.py \
    --tickers AAPL MSFT GOOGL \
    --start-date 2021-01-01 \
    --end-date 2024-12-31 \
    --lookback-days 1095
```

This will:
- Query TimescaleDB for historical price data
- Export stock characteristics
- Save to `data/training/` directory
- Create metadata file

### 4. Sync to Google Drive

Sync the exported data to Google Drive:

```bash
# Dry run first to see what will be copied
python scripts/sync_to_drive.py \
    --drive-path "/Users/username/Google Drive/swing_trading" \
    --dry-run

# Actually sync
python scripts/sync_to_drive.py \
    --drive-path "/Users/username/Google Drive/swing_trading"
```

## Training in Colab

### 1. Create Colab Notebook

Create a new Google Colab notebook and install dependencies:

```python
# Install dependencies
!pip install torch pytorch-lightning pandas numpy scipy scikit-learn
!pip install pytorch-forecasting  # If using TFT
```

### 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Setup Project

```python
import sys
import os

# Add project to path (if you've uploaded the project to Drive)
sys.path.append('/content/drive/MyDrive/swing_trading')

# Or clone from repository
# !git clone https://github.com/your-repo/swing.git /content/drive/MyDrive/swing_trading
```

### 4. Run Training

The training scripts automatically detect Colab and use Google Drive paths:

```python
# Foundation model training
from src.training.train_foundation import train_foundation_model
from src.utils.config import load_config

config = load_config('/content/drive/MyDrive/swing_trading/config/config.yaml')
train_foundation_model(config_path='/content/drive/MyDrive/swing_trading/config/config.yaml')
```

Or for twin fine-tuning:

```python
from src.training.train_twins import fine_tune_twin
from src.models.foundation import load_foundation_model
from src.utils.config import load_config

config = load_config('/content/drive/MyDrive/swing_trading/config/config.yaml')

# Load foundation model
foundation = load_foundation_model(
    '/content/drive/MyDrive/swing_trading/models/foundation/foundation_v1.0.pt',
    config['models']['foundation']
)

# Fine-tune twin
twin = fine_tune_twin('AAPL', foundation, config)
```

### 5. Weekly Retraining

For automated weekly retraining:

```python
from src.training.weekly_retrain import weekly_twin_retrain_flow

weekly_twin_retrain_flow(config_path='/content/drive/MyDrive/swing_trading/config/config.yaml')
```

## Syncing Models Back to MacBook

After training in Colab, sync models back to your MacBook:

```bash
# Dry run first
python scripts/sync_from_drive.py \
    --drive-path "/Users/username/Google Drive/swing_trading" \
    --dry-run

# Actually sync
python scripts/sync_from_drive.py \
    --drive-path "/Users/username/Google Drive/swing_trading"
```

This will copy:
- Foundation model checkpoints from `models/foundation/`
- Twin model checkpoints from `models/twins/{TICKER}/`

## Workflow Summary

### Initial Setup (One-time)

1. Configure local environment
2. Set up Google Drive folder
3. Export initial training data
4. Sync to Google Drive

### Training Workflow (Repeat as needed)

1. **On MacBook**: Export latest training data
   ```bash
   python scripts/export_training_data.py
   ```

2. **On MacBook**: Sync to Google Drive
   ```bash
   python scripts/sync_to_drive.py --drive-path "/path/to/Google Drive/swing_trading"
   ```

3. **In Colab**: Mount Drive and run training
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Run training scripts
   ```

4. **On MacBook**: Sync trained models back
   ```bash
   python scripts/sync_from_drive.py --drive-path "/path/to/Google Drive/swing_trading"
   ```

## Configuration

The system automatically detects the environment:

- **Colab**: Uses Google Drive paths from `training.colab` config
- **Local**: Uses local paths from `storage.local` config

You can override with `training.environment`:
- `auto`: Auto-detect (default)
- `local`: Force local paths
- `colab`: Force Colab paths

## Troubleshooting

### Google Drive Not Mounting

If Google Drive mount fails in Colab:
1. Check authentication token
2. Ensure you have access to the Drive folder
3. Try remounting: `drive.mount('/content/drive', force_remount=True)`

### Models Not Found

If models aren't found:
1. Check that models were saved to correct path in Colab
2. Verify Google Drive sync completed
3. Check file paths in config match actual locations

### Data Export Issues

If data export fails:
1. Verify TimescaleDB is running: `docker-compose ps`
2. Check database connection in config
3. Ensure sufficient disk space for exports

## Best Practices

1. **Regular Exports**: Export training data regularly to keep Colab data current
2. **Version Control**: Tag model checkpoints with dates/versions
3. **Backup**: Keep backups of trained models
4. **Monitor Sync**: Use `--dry-run` before actual syncs
5. **Clean Up**: Remove old training data exports periodically

## File Structure

```
Google Drive/swing_trading/
├── data/
│   └── training/
│       ├── prices_2021-01-01_2024-12-31.parquet
│       ├── stock_characteristics.parquet
│       └── metadata.json
├── models/
│   ├── foundation/
│   │   └── foundation_v1.0.pt
│   └── twins/
│       ├── AAPL/
│       │   └── twin_latest.pt
│       └── MSFT/
│           └── twin_latest.pt
└── config/
    └── config.yaml
```

## Notes

- Training data exports can be large (several GB for 3 years of S&P 500 data)
- Google Drive has storage limits (15GB free, more with paid plans)
- Colab sessions timeout after inactivity - save checkpoints frequently
- GPU availability in Colab is limited - use GPU runtime for faster training

