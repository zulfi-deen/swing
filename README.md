# Swing Trading Recommendation System

A production-grade, end-to-end swing trading recommendation system for S&P 500 stocks. The system leverages deep learning (Temporal Fusion Transformers + Graph Neural Networks), reinforcement learning for portfolio optimization, statistical models (ARIMA/GARCH), LLM agents for text processing, and classic technical analysis to generate high-probability trade recommendations daily.

## Architecture Overview

### Deployment Architecture (Lightning.ai + Local)

```
Data Sources (Polygon, Finnhub) 
  → Data Ingestion (Prefect)
    - Equity data (OHLCV)
    - Options data (OI, Greeks, IV) [if enabled]
  → Storage (Local Filesystem, TimescaleDB, Feast)
  → Feature Engineering
    - Technical indicators
    - Options features (40 features) [if enabled]
    - Cross-sectional features
  → Correlation Graph (Parquet) → GNN
  → Foundation Model (Universal Market Intelligence)
  → Digital Twins (Stock-Specific Predictors)
    - Options encoder [if enabled]
    - Gamma adjustment & PCR sentiment gates
  → Options Intelligence Layer
    - PCR extremes, gamma zones, IV regimes
  → RL Portfolio Manager (PPO-based portfolio construction)
    - Options-informed state encoding [if enabled]
  → LLM Agent Layer (Policy, Pattern, Explainer)
    - Options-aware explanations
  → Output (LitServe API via Lightning + optional React dashboard)
```

### Training & Serving Architecture

- **Lightning.ai**: Orchestrates training jobs (foundation, twins, RL) and serves the LitServe API via a LightningApp control plane.
- **Local Machine**: Optional local development environment for running the LightningApp, databases (TimescaleDB, Redis), and experimentation.

## Key Features

- **Four-Layer Architecture**: Foundation Model → Digital Twins → Options Intelligence → RL Portfolio Manager
- **Options Intelligence Layer**: Forward-looking market microstructure signals (PCR extremes, gamma zones, IV regimes, OI trends)
- **Reinforcement Learning Portfolio Optimization**: PPO-based agent learns optimal portfolio construction from data
- **Multi-Model Ensemble**: Combines TFT-GNN hybrid, LightGBM ranker, and ARIMA/GARCH
- **LLM Agents**: Text processing, policy enforcement, and options-aware explanation generation
- **Graph Neural Networks**: Captures inter-stock relationships via dynamic correlation graphs (built daily from price data, cached to parquet)
- **Local-First**: All databases and storage run locally on MacBook (no cloud dependencies)
- **Lightning-Native Training**: Train models via Lightning.ai or locally using PyTorch Lightning
- **Explainability**: Every recommendation includes human-readable rationale with options context

## Project Structure

```
swing/
├── src/
│   ├── data/           # Data ingestion and storage
│   ├── features/       # Feature engineering
│   ├── models/         # Model architectures
│   ├── agents/         # LLM agents + RL portfolio agent
│   ├── api/            # LitServe / FastAPI routes integrated into LightningApp
│   ├── utils/          # Utilities
│   ├── training/       # Training scripts (including RL)
│   └── evaluation/     # Backtesting and evaluation
├── config/             # Configuration files
├── tests/              # Unit and integration tests
├── scripts/            # Deployment and utility scripts
└── docs/               # Documentation
```

## Setup

### Quick Start (Local MacBook)

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

**Note**: The codebase uses PyTorch Lightning for all training. Key dependencies:
- `pytorch-lightning>=2.1.0` - Training orchestration
- `pytorch-forecasting>=1.0.0` - Temporal Fusion Transformer support
- `torch-geometric>=2.4.0` - Graph Neural Networks

2. **Configure Environment**
```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your API keys and database credentials
```

3. **Start Local Databases** (optional if running everything on Lightning.ai)
```bash
# Start TimescaleDB and Redis via Docker Compose
docker-compose up -d

# The database schema will be automatically initialized on first startup
# Verify databases are running
docker-compose ps
```

4. **Run Data Ingestion**
```bash
# Ingest initial data
prefect deployment run data-ingestion/daily
# Or run directly: python -m src.data.ingestion
```

5. **Start Lightning App (LitServe API + Control Plane)**
```bash
cd swing

# Run locally with Lightning
lightning run app app.py --open

# Or run on Lightning.ai cloud
lightning run app app.py --cloud
```

6. **(Optional) Web UI**

A React/dashboard UI is **not included** in this repository. You can build your own frontend and point it at the LitServe API exposed by the LightningApp.

### Detailed Setup Guides

- **Local Setup**: See [docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md) for complete local setup instructions
- **Lightning.ai Training**: See [docs/LIGHTNING_TRAINING.md](docs/LIGHTNING_TRAINING.md) for training and running models on Lightning.ai)

## Configuration

Key configuration options in `config/config.yaml`:
- API keys (Polygon, Finnhub, OpenAI, FRED)
- Options configuration:
  - Enable/disable options data (`data_sources.options.enabled`)
  - Options provider (`data_sources.options.provider`)
  - Ticker subset for options (`data_sources.options.tickers`)
- Storage configuration:
  - Local filesystem paths (`storage.local`)
- Database connections (TimescaleDB, Redis - local via Docker or Lightning volumes)
- Model parameters (foundation, twins with options encoder, legacy models)
- RL Portfolio configuration (with options signals support)
- Training configuration (local vs Lightning-managed)
- Risk rules and constraints

## Training

All model training uses **PyTorch Lightning** for unified orchestration, logging, and checkpointing.

### Unified Training CLI

Use `scripts/train.py` to train any model:

```bash
# Train foundation model
python scripts/train.py foundation --use-synthetic

# Train digital twin for a ticker
python scripts/train.py twin --ticker AAPL --lookback-days 180

# Train RL portfolio agent
python scripts/train.py rl --start-date 2023-01-01 --end-date 2024-01-01 --num-episodes 1000

# Train LightGBM ranker
python scripts/train.py lightgbm --start-date 2022-01-01 --end-date 2024-12-31
```

### Training Components

- **Foundation Model** (`src/training/train_foundation.py`): Universal market intelligence pre-training
- **Digital Twins** (`src/training/train_twins_lightning.py`): Per-stock fine-tuning with Lightning
- **RL Portfolio Agent** (`src/training/train_rl_portfolio_lightning.py`): PPO-based portfolio optimization
- **LightGBM Ranker** (`src/training/train_lightgbm_lightning.py`): Cross-sectional ranking model

All training scripts use:
- Lightning DataModules for data loading (`src/training/data_modules.py`)
- Shared utilities for callbacks, loggers, and trainers (`src/training/lightning_utils.py`)
- MLflow integration for experiment tracking
- Automatic checkpointing and early stopping

## Daily Pipeline

The system runs an end-of-day (EOD) pipeline that:
1. Ingests market data, options data (if enabled), news, and fundamentals
2. Computes features (technical, options [40 features], cross-sectional, text)
3. Generates predictions from Foundation Model and Digital Twins (with options encoder if enabled)
4. RL Portfolio Manager selects optimal stocks and position sizes using options signals (or falls back to priority scoring)
5. LLM agents curate and explain trades (with options context)
6. Outputs recommendations to database and API

## RL Portfolio Layer

The system includes a reinforcement learning portfolio manager that replaces heuristic priority scoring with a learned policy. See [docs/RL_PORTFOLIO_LAYER.md](docs/RL_PORTFOLIO_LAYER.md) for details.

Key benefits:
- Learns optimal portfolio construction from data
- Adapts to market regimes automatically
- Optimizes for portfolio-level objectives (Sharpe ratio, drawdown)
- Discovers non-linear signal interactions
- Incorporates options market signals (if enabled)

## Options Intelligence Layer

The system includes an options intelligence layer that provides forward-looking market microstructure signals. See [docs/OPTIONS_LAYER.md](docs/OPTIONS_LAYER.md) for complete documentation.

Key features:
- **40 Options Features**: Volume/OI, PCR, gamma, IV, Greeks, term structure, composite signals
- **Options Encoder in Twins**: 40 features → 32-dim embeddings with gamma adjustment and PCR sentiment gates
- **RL Integration**: Options signals inform portfolio construction decisions
- **Options-Aware Explanations**: Trade rationales include PCR extremes, gamma zones, and IV regimes

Expected performance improvements (per v4 architecture document):
- 24% annual return (vs. 18% without options)
- 2.5 Sharpe ratio (vs. 1.8 without options)
- 72% win rate (vs. 65% without options)
- -8% max drawdown (vs. -12% without options)

## License

Proprietary - All rights reserved

