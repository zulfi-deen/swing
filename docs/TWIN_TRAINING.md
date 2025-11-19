# Digital Twin Training Guide

## Overview

This guide covers the training procedures for the Stock Digital Twin system, including foundation pre-training and weekly twin fine-tuning.

## Foundation Model Pre-Training

### Setup

1. Ensure you have 3+ years of historical data for 500 stocks
2. Configure foundation model in `config/config.example.yaml`
3. Install dependencies: `pip install pytorch-lightning`

### Training

```bash
python src/training/train_foundation.py
```

**Configuration**:
- Learning rate: 1e-3
- Batch size: 128
- Epochs: 100
- Early stopping patience: 15
- Optimizer: AdamW with cosine annealing

**Output**: Foundation checkpoint saved to `models/foundation/foundation_v1.0.pt`

## Twin Fine-Tuning

### Per-Stock Fine-Tuning

```python
from src.training.train_twins import fine_tune_twin
from src.models.foundation import load_foundation_model
from src.utils.config import load_config

config = load_config()
foundation = load_foundation_model(
    config['models']['foundation']['checkpoint_path'],
    config['models']['foundation']
)

twin = fine_tune_twin('AAPL', foundation, config)
```

**Configuration**:
- Learning rate: 5e-4
- Batch size: 64
- Epochs: 20
- Lookback: 180 days (6 months)
- Weight decay: 1e-5
- Options enabled: Set `models.twins.options_enabled: true` in config (requires options features in training data)

**Output**: Twin checkpoint saved to `models/twins/{TICKER}/twin_latest.pt`

**Note**: If options are enabled, the twin will include:
- Options encoder (40 features → 32-dim embeddings)
- Gamma adjustment head (for max pain corrections)
- PCR sentiment gate (for contrarian signals)
- Options embeddings fused into regime detection and correction layers

## Weekly Retraining Pipeline

### Automated Weekly Fine-Tuning

The weekly retraining pipeline runs every Sunday at 2 AM via Prefect:

```bash
python src/training/weekly_retrain.py
```

Or schedule via Prefect:

```python
from prefect import serve
from src.training.weekly_retrain import weekly_twin_retrain_flow

# Schedule for Sundays at 2 AM
serve(weekly_twin_retrain_flow.to_deployment(
    name="weekly-twin-retraining",
    cron="0 2 * * 0"  # Sunday 2 AM
))
```

### Pipeline Steps

1. **Update Stock Characteristics**: Recompute beta, Hurst, liquidity regime
2. **Fine-Tune Twins**: For each pilot ticker, fine-tune on last 6 months
   - If options enabled: Include options features in training batches
   - Options encoder, gamma adjustment, and PCR gates are trained end-to-end
3. **Validate**: Compute metrics (MAE, hit rate, regime accuracy)
   - If options enabled: Validate options signal accuracy (PCR extremes, gamma zones)
4. **Save Checkpoints**: Upload to S3 or local storage
5. **Log Metrics**: Track in MLflow

### Monitoring

- MLflow tracks training metrics per twin
- Check logs for errors: `logs/weekly_retrain.log`
- Monitor training time (should be 5-10 min per stock)

## Synthetic Data (For Testing)

For demo/testing without real data:

```python
from src.training.synthetic_data import create_synthetic_dataset

dataset = create_synthetic_dataset(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2022-01-01',
    end_date='2024-12-31'
)

prices_df = dataset['prices']
features_df = dataset['features']
```

## Troubleshooting

### Foundation Training Fails

- Check TFT initialization: Requires pytorch-forecasting dataset
- Verify data format matches TFT requirements
- Check GPU memory if using GPU

### Twin Fine-Tuning Fails

- Ensure stock characteristics exist in database
- Verify sufficient historical data (minimum 60 days)
- Check checkpoint directory permissions

### Weekly Pipeline Fails

- Verify Prefect is running
- Check database connectivity
- Ensure foundation checkpoint exists
- Review logs for specific errors

## Best Practices

1. **Backup Checkpoints**: Regularly backup foundation and twin checkpoints
2. **Monitor Metrics**: Track hit rate, regime accuracy over time
3. **Gradual Rollout**: Start with 10-20 pilot stocks, expand gradually
4. **Version Control**: Tag model versions in MLflow
5. **A/B Testing**: Compare twin predictions vs baseline


