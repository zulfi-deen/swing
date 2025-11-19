# Project Structure

## Overview

This document describes the complete structure of the Swing Trading Recommendation System, including both the original TFT-GNN architecture and the new Stock Digital Twin architecture.

## Directory Layout

```
swing/
├── docs/                                    # Documentation
│   ├── Complete Architecture & Design Document.pdf
│   ├── Stock Twin Complete Architecture & Design Document.pdf
│   ├── RL Portfolio layer.pdf               # RL Portfolio layer design document
│   ├── Options Complete Architecture & Design Document v4.pdf  # Options layer architecture
│   ├── PROJECT_STRUCTURE.md                 # This file
│   ├── STOCK_TWIN_ARCHITECTURE.md           # Stock Twin architecture details
│   ├── RL_PORTFOLIO_LAYER.md                # RL Portfolio layer documentation
│   ├── TWIN_TRAINING.md                     # Training procedures
│   ├── REGIME_DETECTION.md                  # Regime detection documentation
│   ├── NEO4J_SETUP.md                       # Neo4j setup guide (deprecated - now using parquet)
│   ├── LOCAL_SETUP.md                       # Local database and storage setup guide
│   ├── COLAB_TRAINING.md                    # Colab training workflow guide
│   └── design.docx
│
├── feature_repo/                            # Feast Feature Store
│   ├── feature_store.yaml
│   ├── definitions.py
│   └── data/
│
├── src/                                     # Source code
│   ├── __init__.py
│   │
│   ├── data/                                # Data ingestion and storage
│   │   ├── __init__.py
│   │   ├── ingestion.py                     # Prefect flows for data ingestion
│   │   ├── sources.py                       # API clients (Polygon, Finnhub, FRED, VIX)
│   │   ├── storage.py                       # Local filesystem and TimescaleDB storage
│   │   ├── neo4j_client.py                  # Neo4j client (deprecated - now using parquet)
│   │   ├── validation.py                    # Data quality checks and validation
│   │   └── rl_state_builder.py              # RL state builder for portfolio agent
│   │
│   ├── features/                            # Feature engineering
│   │   ├── __init__.py
│   │   ├── technical.py                     # Technical indicators (TA-Lib)
│   │   ├── options.py                       # Options feature engineering (40 features)
│   │   ├── cross_sectional.py               # Cross-sectional features
│   │   ├── graph.py                         # Graph construction for GNN (parquet-based storage)
│   │   ├── graph_storage.py                 # Parquet-based graph storage and retrieval
│   │   ├── macro.py                         # Macro economic feature engineering
│   │   ├── normalization.py                 # Feature normalization and scaling
│   │   └── feast_client.py                 # Feast feature store client
│   │
│   ├── models/                              # Model architectures
│   │   ├── __init__.py
│   │   ├── foundation.py                    # Stock Twin Foundation Model (Universal Market Intelligence)
│   │   ├── digital_twin.py                  # Stock-Specific Digital Twin Models
│   │   ├── twin_manager.py                  # TwinManager for orchestrating multiple twins
│   │   ├── regime.py                        # Stock regime detection (Trending/MeanReverting/Choppy/Volatile)
│   │   ├── market_regime.py                 # Market-level regime detection (Bull/Bear/Sideways)
│   │   ├── model_registry.py                # Model registry for caching and serving
│   │   ├── tft_gnn.py                       # Legacy TFT-GNN hybrid model (backward compatibility)
│   │   ├── ensemble.py                     # LightGBM ranker and ensemble
│   │   ├── arima_garch.py                   # ARIMA/GARCH baseline
│   │   └── patterns.py                      # Chart pattern detection
│   │
│   ├── agents/                              # LLM agents + RL portfolio agent
│   │   ├── __init__.py
│   │   ├── text_summarizer.py               # TextSummarizerAgent
│   │   ├── policy_agent.py                  # PolicyAgent (trade curation)
│   │   ├── explainer.py                     # ExplainerAgent (rationale)
│   │   ├── pattern_agent.py                 # PatternDetectorAgent (tech validation)
│   │   ├── related_agent.py                 # RelatedStockAgent (peer/sympathy)
│   │   └── portfolio_rl_agent.py            # PortfolioRLAgent (PPO-based portfolio manager)
│   │
│   ├── api/                                 # FastAPI backend
│   │   ├── __init__.py
│   │   └── main.py                          # API endpoints (including twin endpoints)
│   │
│   ├── ui/                                  # React frontend
│   │   ├── package.json
│   │   ├── public/
│   │   │   └── index.html
│   │   └── src/
│   │       ├── App.jsx
│   │       ├── App.css
│   │       ├── index.js
│   │       └── index.css
│   │
│   ├── pipeline/                            # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── orchestrator.py                  # Daily EOD pipeline orchestrator
│   │   └── batch_preparation.py             # Batch preparation for model inference
│   │
│   ├── training/                            # Training scripts
│   │   ├── __init__.py
│   │   ├── train.py                         # Legacy training pipeline
│   │   ├── train_foundation.py              # Foundation model pre-training
│   │   ├── train_twins.py                   # Digital twin fine-tuning
│   │   ├── train_rl_portfolio.py            # RL portfolio agent training
│   │   ├── weekly_retrain.py                # Weekly retraining pipeline (Prefect)
│   │   ├── dataset.py                       # TimeSeriesDataSet preparation
│   │   ├── loss.py                          # Loss functions (FoundationLoss, TwinLoss)
│   │   ├── synthetic_data.py                # Synthetic data generation for testing
│   │   └── rl_environment.py                # Trading environment for RL training
│   │
│   ├── evaluation/                          # Evaluation and backtesting
│   │   ├── __init__.py
│   │   ├── backtest.py                      # Backtesting framework
│   │   ├── paper_trading.py                 # Paper trading tracker
│   │   └── options_metrics.py               # Options prediction accuracy metrics
│   │
│   └── utils/                               # Utilities
│       ├── __init__.py
│       ├── tickers.py                       # Ticker utilities with complete sector mapping
│       ├── config.py                        # Configuration loader with environment variable support
│       ├── mlflow_logger.py                 # MLflow integration for experiment tracking
│       └── colab_utils.py                   # Colab environment detection and Google Drive mounting
│
├── config/                                  # Configuration files
│   └── config.example.yaml                  # Example configuration (includes foundation/twin config)
│
├── scripts/                                 # Utility scripts
│   ├── schema.sql                           # TimescaleDB schema (includes stock_characteristics, twin_predictions)
│   ├── run_pipeline.py                      # Daily pipeline runner
│   ├── export_training_data.py              # Export training data for Colab
│   ├── sync_to_drive.py                     # Sync data to Google Drive
│   └── sync_from_drive.py                   # Sync models from Google Drive
│
├── tests/                                   # Tests
│   ├── __init__.py
│   ├── test_foundation.py                   # Foundation model tests
│   ├── test_digital_twin.py                 # Digital twin tests
│   ├── test_twin_manager.py                 # TwinManager tests
│   ├── test_regime.py                       # Regime detection tests
│   ├── test_market_regime.py                # Market regime tests
│   ├── test_batch_preparation.py            # Batch preparation tests
│   ├── test_dataset.py                      # Dataset preparation tests
│   ├── test_normalization.py                # Feature normalization tests
│   └── test_features.py                     # Feature engineering tests
│
├── .gitignore
├── Dockerfile
├── docker-compose.yml                       # Docker services (TimescaleDB, Redis)
├── Makefile
├── README.md
└── requirements.txt
```

## Key Components

### 1. Data Layer (`src/data/` & `feature_repo/`)

#### Data Sources (`src/data/sources.py`)
- **PolygonClient**: Market data (OHLCV) for stocks, ETFs, and indexes
- **PolygonOptionsClient**: Options chain data (Greeks, OI, IV) for stocks with options
- **FinnhubClient**: News, sentiment, fundamentals, and earnings calendar
- **FREDClient**: Macro economic indicators (GDP, unemployment, interest rates, CPI, Treasury rates)
- **VIXClient**: Volatility index data via Polygon

#### Data Ingestion (`src/data/ingestion.py`)
- **daily_data_ingestion_flow**: Prefect flow orchestrating:
  - Market data fetching (Polygon)
  - Options data fetching (Polygon) - if enabled
  - News and sentiment (Finnhub)
  - Fundamentals (Finnhub)
  - Macro data (FRED, VIX)
- Saves to local filesystem (raw data lake) and TimescaleDB (time-series)

#### Storage (`src/data/storage.py`)
- **Local Filesystem**: Raw data lake (prices, news, fundamentals, macro) stored in `data/raw/`
  - `save_to_local()`: Save DataFrames/dicts to local filesystem (Parquet/JSON)
  - `load_from_local()`: Load data from local filesystem
  - Backward compatible with deprecated S3 functions
- **TimescaleDB**: Time-series database for prices, features, recommendations (local via Docker)
- **Graph Storage**: Parquet files in `data/graphs/` for correlation graphs (computed daily, no database needed)
- **Feast**: Feature store with local registry (`data/feast/registry.db`) and Redis online store

#### Data Validation (`src/data/validation.py`)
- Price data validation (OHLCV checks)
- Feature validation (missing features, NaN/Inf checks)
- Data freshness monitoring
- Feature drift detection
- Batch validation for TFT models

### 2. Feature Engineering (`src/features/`)

#### Technical Features (`src/features/technical.py`)
- Technical indicators: RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX, MFI
- Volume features: Volume z-scores, dollar volume
- Price action: Gaps, intraday range, distance to 52-week high/low
- Fibonacci levels

#### Options Features (`src/features/options.py`)
- **OptionsFeatureExtractor**: Extracts 40 options features per stock per day
  - Volume & Open Interest (8 features): call/put OI, volume, OI change, volume z-score
  - Put-Call Ratios (6 features): PCR OI/volume, z-scores, extreme flags, trend
  - Gamma Exposure (7 features): max pain strike, gamma concentration, flip levels
  - Implied Volatility (7 features): ATM IV, skew, percentile, rank, term structure
  - Net Greeks (5 features): net delta, gamma, vega, theta, delta strength
  - Term Structure (4 features): front/next month OI, roll ratio, curve slope
  - Composite Signals (3 features): trend confirmation, sentiment extreme, gamma zone
- **compute_options_features_batch**: Batch processing for multiple tickers
- Integrated into daily pipeline for options-enabled stocks

#### Cross-Sectional Features (`src/features/cross_sectional.py`)
- Return rankings (5d, 20d)
- Sector relative strength
- Correlation to SPY
- Peer comparisons

#### Graph Features (`src/features/graph.py`, `src/features/graph_storage.py`)
- **compute_correlation_matrix**: Computes rolling correlation matrix
- **build_correlation_graph**: Builds PyTorch Geometric graph with parquet caching
- **build_correlation_graph_from_parquet**: Retrieves graph from parquet cache
- **build_heterogeneous_graph**: Creates heterogeneous graph with stocks, sectors, and macro nodes
- **save_correlation_graph**: Saves graph edges to parquet
- **load_correlation_graph**: Loads graph from parquet cache
- **get_or_build_graph**: Gets from cache or computes fresh

#### Macro Features (`src/features/macro.py`)
- **add_macro_features**: Adds FRED indicators and VIX to feature DataFrame
- **compute_macro_features_for_date**: Computes macro features for a specific date
- Integrates GDP, unemployment, interest rates, CPI, Treasury rates, VIX

#### Feature Normalization (`src/features/normalization.py`)
- **FeatureNormalizer**: StandardScaler or RobustScaler for numerical features
- Handles fitting, transforming, and inverse transforming
- Supports feature groups (price-based, indicators, volume, returns, sentiment, cross-sectional)

### 3. Models (`src/models/`)

#### Foundation Model (`src/models/foundation.py`)
- **StockTwinFoundation**: Universal market intelligence engine
  - Temporal Fusion Transformer (TFT) for temporal patterns
  - Graph Attention Network (GAT) for cross-stock relationships
  - Shared embeddings (sector, liquidity, market regime)
  - Foundation backbone combining all representations
  - Outputs 128-dim embeddings transferable to any stock
- **initialize_tft**: Initializes TFT from TimeSeriesDataSet
- **load_foundation_model**: Loads pre-trained foundation model

#### Digital Twins (`src/models/digital_twin.py`)
- **StockDigitalTwin**: Stock-specific specialized model
  - LoRA-style adaptation layers (minimal parameters)
  - **Options Encoder** (if enabled): 40 options features → 32-dim embeddings
  - **Gamma Adjustment Head**: Max pain corrections for return predictions
  - **PCR Sentiment Gate**: Contrarian signal modulation for probability
  - Stock-specific embeddings
  - Regime detector (Trending/MeanReverting/Choppy/Volatile) - enhanced with options
  - Correction layers for stock-specific adjustments (fused with options)
  - Multiple prediction heads (return, probability, volatility, quantiles)
  - Idiosyncratic alpha tracking
- **save_checkpoint** / **load_checkpoint**: Saves/loads only trainable parameters

#### Twin Manager (`src/models/twin_manager.py`)
- **TwinManager**: Orchestrates multiple digital twins
  - Loads foundation model (frozen)
  - Initializes twins for pilot stocks
  - Caches twins in memory
  - Batch predictions across multiple twins
  - Lists available twins with metadata

#### Regime Detection (`src/models/regime.py`)
- **detect_regime_features**: Classifies stock regimes (Trending/MeanReverting/Choppy/Volatile)
- **get_regime_name**: Human-readable regime names
- **get_regime_trading_strategy**: Suggested trading strategies per regime

#### Market Regime (`src/models/market_regime.py`)
- **detect_market_regime**: Detects overall market regime (Bull/Bear/Sideways)
- **get_market_regime_features**: Detailed market regime features
- Based on SPY price action, moving averages, volatility

#### Model Registry (`src/models/model_registry.py`)
- **ModelRegistry**: Singleton for managing and caching TwinManager
- Ensures only one instance is loaded
- Provides status tracking and lazy loading

#### Legacy Models
- **tft_gnn.py**: Original TFT-GNN hybrid (kept for backward compatibility)
- **ensemble.py**: LightGBM ranker and meta-ensemble
- **arima_garch.py**: ARIMA/GARCH baseline for volatility
- **patterns.py**: Chart pattern detection

### 4. LLM Agents (`src/agents/`)

- **text_summarizer.py**: Processes news/text into structured features
- **policy_agent.py**: Applies trading rules and curates final trade list
- **explainer.py**: Generates human-readable rationale for recommendations
  - **explain_trade_with_options()**: Options-aware explanations with PCR, gamma, IV context
- **pattern_agent.py**: Validates technical patterns using LLM + Rules
- **related_agent.py**: Finds correlated/sector peers

### 5. Pipeline Orchestration (`src/pipeline/`)

#### Orchestrator (`src/pipeline/orchestrator.py`)
- **PipelineOrchestrator**: Complete daily EOD pipeline
  - Step 1: Data Ingestion (market, news, fundamentals, macro)
  - Step 2: Load data from database
  - Step 3: Feature Engineering
  - Step 3.1: Market Regime Detection
  - Step 3.5: Build and cache correlation graph to parquet
  - Step 3.6: Options Features (if enabled) - computes 40 features per stock
  - Step 4: Text Features (LLM Agents)
  - Step 5: Model Inference (TwinManager for pilot stocks with options, fallback for others)
  - Step 6: Ensemble & Ranking (RL agent uses options signals if enabled)
  - Step 7: Agent Curation (with options-aware explanations)
  - Step 8: Save to Database
  - Step 9: Record paper trades

#### Batch Preparation (`src/pipeline/batch_preparation.py`)
- **prepare_batch_for_twin**: Converts DataFrame to TFT-compatible batch format
- **prepare_batches_for_multiple_tickers**: Batch preparation for multiple tickers
- Handles feature normalization, time indexing, and tensor conversion

### 6. API (`src/api/`)

- **main.py**: FastAPI application with endpoints:
  - `/recommendations/latest` - Get today's recommendations
  - `/recommendations/{date}` - Get recommendations for a date
  - `/briefs/latest` - Get daily brief with market context
  - `/explain/{ticker}` - Get detailed explanation
  - `/performance/backtest` - Historical backtest metrics
  - `/performance/paper` - Paper trading performance
  - `/models/twins` - List all available digital twins
  - `/models/twins/{ticker}` - Get detailed twin information
  - `/predictions/{ticker}/regime` - Get current regime for a ticker

### 7. UI (`src/ui/`)

- React dashboard for displaying recommendations
- Shows daily brief, trade list, and rationale
- Displays twin information and regime classifications

### 8. Training (`src/training/`)

#### Foundation Training (`src/training/train_foundation.py`)
- **FoundationTrainingModule**: PyTorch Lightning module for foundation pre-training
- Uses synthetic data for demo/testing
- Supports real data training (when implemented)
- MLflow logging and checkpointing

#### Twin Fine-Tuning (`src/training/train_twins.py`)
- **TwinFineTuner**: Fine-tunes individual digital twins
- Uses last 6 months of stock-specific data
- Saves checkpoints per ticker

#### Weekly Retraining (`src/training/weekly_retrain.py`)
- **weekly_twin_flow**: Prefect flow for weekly retraining
- Updates stock characteristics
- Fine-tunes all pilot twins
- MLflow tracking

#### Dataset Preparation (`src/training/dataset.py`)
- **create_tft_dataset**: Creates PyTorch Forecasting TimeSeriesDataSet
- Handles static, known, and unknown time-varying features
- Proper target calculation and normalization

#### Loss Functions (`src/training/loss.py`)
- **FoundationLoss**: Loss for foundation pre-training (return MSE + probability BCE)
- **TwinLoss**: Loss for twin fine-tuning (return + probability + regime + quantiles)
- **QuantileLoss**: Pinball loss for quantile regression

#### Synthetic Data (`src/training/synthetic_data.py`)
- **generate_synthetic_data**: Generates realistic OHLCV data
- **generate_synthetic_features**: Generates synthetic features
- Used for testing and demo purposes

### 9. Evaluation (`src/evaluation/`)

- **backtest.py**: Backtesting framework with portfolio tracking
- **paper_trading.py**: Paper trading tracker
  - Trade recording and position updates
  - Performance reporting

### 10. Utilities (`src/utils/`)

- **tickers.py**: Ticker utilities with complete S&P 500 sector mapping
- **config.py**: Configuration loader with environment variable support
- **mlflow_logger.py**: MLflow integration for experiment tracking
- **colab_utils.py**: Colab environment detection and Google Drive integration
  - `is_colab()`: Detects if running in Google Colab
  - `mount_google_drive()`: Mounts Google Drive in Colab
  - `get_data_paths()`: Returns appropriate paths based on environment
  - `setup_colab_environment()`: Sets up Colab environment with Drive mounting

## Data Flow

### Daily Pipeline Flow

1. **Data Ingestion** (Prefect)
   - Market data (Polygon) → Local filesystem (`data/raw/prices/`) + TimescaleDB
   - News/Sentiment (Finnhub) → Local filesystem (`data/raw/news/`)
   - Fundamentals (Finnhub) → Local filesystem (`data/raw/fundamentals/`)
   - Macro data (FRED, VIX) → Local filesystem (`data/raw/macro/`)

2. **Feature Engineering**
   - Technical indicators → Feature DataFrame
   - Options features (if enabled) → 40 features per stock → Options DataFrame
   - Cross-sectional features → Feature DataFrame
   - Macro features → Feature DataFrame
   - Market regime detection → Feature DataFrame

3. **Correlation Graph**
   - Compute correlation matrix from prices
   - Build PyTorch Geometric graph
   - Cache to parquet files in `data/graphs/`
   - Retrieve from parquet cache for inference (or recompute if missing)

4. **Text Processing** (LLM Agents)
   - News summarization → Structured text features

5. **Model Inference**
   - **Pilot Stocks**: Use TwinManager → Digital Twin predictions (with options encoder if enabled)
   - **Other Stocks**: Fallback to LightGBM/ARIMA
   - Load graph from parquet cache → GNN processing
   - Options features passed to twins for gamma adjustment and PCR sentiment gating

6. **Portfolio Construction**
   - **RL Portfolio Manager** (if enabled): Uses PPO agent to select stocks and position sizes
     - Builds RL state from twin predictions, features, portfolio, macro context, and options signals
     - Options market encoder aggregates options signals across stocks
     - Agent outputs attention scores and position sizes (options-informed)
     - Falls back to priority scoring if RL agent unavailable
   - **Priority Scoring** (fallback): Calculate priority scores from predictions

7. **Agent Curation** (PolicyAgent)
   - Apply trading rules
   - Curate final trade list (10-15 trades)

8. **Save to Database**
   - Recommendations → TimescaleDB
   - Twin predictions → TimescaleDB (twin_predictions table)

9. **Paper Trading**
   - Record recommendations
   - Track positions and performance

10. **API** → React UI

### Training Flow

#### Local Training (MacBook)
1. **Foundation Pre-Training**
   - Load all S&P 500 stock data (3+ years) from TimescaleDB
   - Create TimeSeriesDataSet
   - Train foundation model on universal patterns
   - Save foundation checkpoint to `models/foundation/`

2. **Initial Twin Fine-Tuning**
   - For each pilot stock:
     - Load foundation model (frozen)
     - Create StockDigitalTwin
     - Fine-tune on last 6 months of stock data
     - Save twin checkpoint to `models/twins/{TICKER}/`

3. **RL Portfolio Agent Training**
   - Load historical predictions, prices, features, and macro data
   - Train PPO agent using TradingEnvironment
   - Each episode = 60 trading days
   - Save best model checkpoint to `models/rl_portfolio/`

4. **Weekly Retraining**
   - Update stock characteristics
   - Fine-tune each twin on recent data
   - Optionally retrain RL portfolio agent on recent data
   - Save updated checkpoints

#### Colab Training Workflow
1. **Data Export** (on MacBook)
   - Run `scripts/export_training_data.py` to export training data from TimescaleDB
   - Exports to `data/training/` as Parquet files
   - Sync to Google Drive using `scripts/sync_to_drive.py`

2. **Training in Colab**
   - Mount Google Drive in Colab notebook
   - Training scripts auto-detect Colab environment
   - Load data from Google Drive paths
   - Save models to Google Drive during training

3. **Model Sync** (on MacBook)
   - Run `scripts/sync_from_drive.py` to download trained models
   - Models synced to local `models/` directory
   - Ready for inference on local MacBook

## Storage Schema

### TimescaleDB Tables

- **prices**: OHLCV data (hypertable)
- **options_prices**: Raw options chain data (Greeks, OI, IV) per contract (hypertable)
- **options_features**: Processed options features (40 features per stock per day) (hypertable)
- **stock_characteristics**: Stock metadata (beta, liquidity regime, etc.)
- **twin_predictions**: Per-stock twin predictions (hypertable)
- **recommendations**: Final trade recommendations
- **daily_briefs**: Daily market briefs

### Graph Storage Format

- **Parquet Files**: `data/graphs/correlations/YYYY-MM-DD.parquet`
  - Columns: `ticker1`, `ticker2`, `correlation`, `abs_correlation`
- **Metadata**: `data/graphs/metadata/YYYY-MM-DD.json`
  - Stores: `lookback_days`, `threshold`, `num_edges`, `num_nodes`, `computation_time`
- **Computed Daily**: Graphs are built from price data and cached to parquet
- **No Database**: Pure file-based storage using PyTorch Geometric and pandas
- **Relationships**: 
  - `CORRELATES_WITH` (stock → stock, weighted by correlation)
  - `BELONGS_TO` (stock → sector)
  - `INFLUENCED_BY` (sector → macro)

### Local Filesystem Structure

```
data/
├── raw/
│   ├── prices/{date}/        # Raw price data (Parquet)
│   ├── options/{date}/      # Raw options chain data (JSON/Parquet)
│   ├── news/{date}/          # News data (JSON)
│   ├── fundamentals/{date}/  # Fundamentals (Parquet)
│   └── macro/{date}/         # Macro data (JSON)
├── processed/
│   └── features/             # Processed features
│       └── options/          # Processed options features
├── feast/
│   └── registry.db           # Feast feature store registry
└── training/                 # Training data exports for Colab
    ├── prices_*.parquet
    ├── stock_characteristics.parquet
    └── metadata.json

models/
├── foundation/               # Foundation model checkpoints
└── twins/                    # Digital twin checkpoints
    └── {TICKER}/
        └── twin_latest.pt
```

## Configuration

See `config/config.example.yaml` for:
- API keys (Polygon, Finnhub, OpenAI, FRED)
- Storage configuration:
  - Local filesystem paths (`storage.local`)
  - Google Drive paths for Colab (`storage.google_drive`)
  - TimescaleDB, Redis (all local via Docker)
  - Graph storage: Parquet files (no database needed)
  - Feast registry (local path)
- Model configuration (foundation, twins, legacy models)
- Options configuration:
  - Enable/disable options data (`data_sources.options.enabled`)
  - Options provider (`data_sources.options.provider`)
  - Ticker subset for options (`data_sources.options.tickers`)
- RL Portfolio configuration:
  - Enable/disable RL agent (`rl_portfolio.enabled`)
  - Enable/disable options signals (`rl_portfolio.options_enabled`)
  - Checkpoint path (`rl_portfolio.checkpoint_path`)
  - PPO hyperparameters (`rl_portfolio.ppo`)
- Twins configuration:
  - Enable/disable options encoder (`models.twins.options_enabled`)
  - Options embedding dimension (`models.twins.options_embedding_dim`)
- Training configuration:
  - Environment detection (`training.environment`: auto/local/colab)
  - Colab-specific paths (`training.colab`)
- LLM agent configuration
- Trading rules

## Architecture: Local MacBook + Colab Training

The system uses a hybrid architecture:

- **Local MacBook**:
  - Databases: TimescaleDB, Redis (Docker Compose)
  - Graph Storage: Parquet files in `data/graphs/`
  - Storage: Local filesystem (`data/`)
  - Inference: FastAPI server running locally
  - Data ingestion: Daily pipeline runs locally

- **Google Colab**:
  - Training: Foundation model and digital twin training
  - Data: Loaded from Google Drive (synced from MacBook)
  - Models: Saved to Google Drive (synced back to MacBook)

- **Google Drive**:
  - Data synchronization between MacBook and Colab
  - Training data exports
  - Trained model checkpoints

See [LOCAL_SETUP.md](./LOCAL_SETUP.md) for local setup and [COLAB_TRAINING.md](./COLAB_TRAINING.md) for Colab training workflow.

## Key Design Decisions

1. **Stock Digital Twins**: Per-stock specialized models vs. monolithic model
2. **Hierarchical Learning**: Foundation model → Stock-specific adaptation
3. **LoRA Adaptation**: Minimal parameters for stock-specific fine-tuning
4. **Regime-Aware**: Both stock-level and market-level regime detection
5. **RL Portfolio Optimization**: PPO-based agent learns optimal portfolio construction vs. heuristic priority scoring
6. **Graph Persistence**: Parquet files for correlation graph storage and retrieval (computed daily from price data)
7. **Macro Integration**: FRED and VIX data for macro context
8. **Model Registry**: Singleton pattern for efficient model serving
9. **Local-First Architecture**: All databases and storage run locally on MacBook
10. **Hybrid Training**: Local inference with Colab training via Google Drive sync
11. **Environment Auto-Detection**: Training scripts automatically detect Colab vs local environment
12. **Options Intelligence Layer**: Options data provides forward-looking market microstructure signals (PCR extremes, gamma zones, IV regimes, OI trends) integrated into twins and RL agent
