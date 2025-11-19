# Stock Digital Twin Architecture

## Overview

The Stock Digital Twin system implements a hierarchical meta-learning architecture where each stock receives its own specialized AI twin that understands its unique behavior patterns, regime dynamics, and risk characteristics.

## Architecture Philosophy

### Why Digital Twins?

Traditional monolithic models assume all stocks behave similarly, but in reality:
- **AAPL (β=0.9)**: Trends with tech sector, sensitive to product cycles, high liquidity
- **TSLA (β=2.2)**: Extreme volatility, sentiment-driven, erratic regime shifts
- **JNJ (β=0.6)**: Mean-reverting, dividend-focused, low volatility, defensive

Each stock has unique behavioral "fingerprints" that require specialized models.

### Three-Layer Intelligence Stack

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: LLM Agents (Context & Policy)                      │
│ - Text → Structured Features                                │
│ - Trade Curation & Risk Rules                               │
│ - Explanation Generation                                     │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │  Context + Constraints
                          │
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Digital Twins (50 Stock-Specific Models)          │
│ - Per-Stock Adaptation Layers                               │
│ - Regime Detection                                           │
│ - Idiosyncratic Alpha Capture                               │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │  Fine-Tuned Specialization
                          │
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Foundation Model (Universal Market Intelligence)   │
│ - TFT: Temporal patterns across all stocks                  │
│ - GNN: Cross-stock relationships & sector dynamics          │
│ - Shared embeddings: Sector, liquidity, macro context       │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Foundation Model (`src/models/foundation.py`)

**Purpose**: Universal market intelligence engine trained on all 500 stocks.

**Architecture**:
- **TFT Encoder**: 256-dim hidden, 60-day lookback, 5-day forward
- **GAT Layers**: 2 layers, 8 heads, edge weights from Neo4j correlations
- **Shared Embeddings**: Sector (32-dim), Liquidity (16-dim), Market Regime (16-dim)
- **Foundation Backbone**: 448-dim input → 256 → 128-dim output

**Output**: 128-dim embeddings that capture universal patterns transferable to any stock.

**Training**: One-time pre-training on all 500 stocks × 3 years of data.

### 2. Digital Twins (`src/models/digital_twin.py`)

**Purpose**: Stock-specific adaptation of foundation knowledge.

**Architecture**:
1. **Foundation Embeddings** (frozen, 128-dim)
2. **LoRA Adaptation**: Down (128→16) + Up (16→128), near-zero init
3. **Options Encoder** (if enabled): 40 options features → 64 → 32-dim embeddings
   - **Gamma Adjustment Head**: Max pain corrections for return predictions
   - **PCR Sentiment Gate**: Contrarian signal modulation for probability
4. **Stock Embedding**: Learnable 64-dim vector
5. **Regime Detector**: 4-class classifier (Trending/MeanReverting/Choppy/Volatile)
   - Enhanced with options embeddings if enabled
6. **Regime Embeddings**: 4 learned 32-dim vectors
7. **Correction Layers**: Combines adapted + stock + regime + options → 64-dim
8. **Prediction Heads**:
   - Expected return (5-day) - adjusted by gamma signals
   - Hit probability - gated by PCR sentiment extremes
   - Volatility forecast
   - Quantiles (q10, q50, q90)

**Parameter Efficiency**:
- Foundation: ~5M parameters (shared, frozen)
- Each Twin (without options): ~50K parameters (adaptation + heads)
- Each Twin (with options): ~75K parameters (adaptation + options encoder + heads)
- Total (50 twins, options enabled): 5M + (50 × 75K) = 8.75M parameters

**Training**: Weekly fine-tuning on last 6 months of stock-specific data.

### 3. Regime Detection (`src/models/regime.py`)

**Regimes**:
- **Trending (0)**: Strong directional move, momentum strategy
- **MeanReverting (1)**: Oscillating around mean, counter-trend strategy
- **Choppy (2)**: No clear pattern, avoid or tight ranges
- **Volatile (3)**: High volatility, wider stops, smaller positions

**Detection**: Based on trend strength, RSI extremes, volatility, beta, Hurst exponent.

### 4. Twin Manager (`src/models/twin_manager.py`)

**Purpose**: Central orchestrator for managing 50 pilot twins.

**Features**:
- Loads foundation model (frozen)
- Initializes twins for pilot stocks
- Caches twins in memory
- Batch predictions
- Checkpoint management

## Data Flow

1. **Data Ingestion** → S3 + TimescaleDB
2. **Feature Engineering** → Feature store (Feast)
3. **Correlation Graph** → Computed from prices → Persisted to Neo4j
4. **Text Processing** (LLM Agents) → Structured text features
5. **Options Feature Engineering** (if enabled):
   - Load raw options data → Extract 40 features per stock
   - Save to TimescaleDB (options_features table)
6. **Model Inference**:
   - Load graph from Neo4j
   - For pilot stocks: Use TwinManager → Digital Twin (with options encoder) → Predictions
   - Options features passed to twins for gamma adjustment and PCR gating
   - For others: Fallback to LightGBM
7. **Portfolio Construction**:
   - **RL Portfolio Manager** (if enabled): PPO agent selects stocks and position sizes from twin predictions + options signals
   - **Fallback**: Priority scoring (expected_return × hit_prob × (1 - volatility))
8. **Agent Curation** (PolicyAgent) → Final recommendations (with options-aware explanations)
9. **Paper Trading** → Trade tracking

## Training Pipeline

### Phase 1: Foundation Pre-Training
- Train on all 500 stocks × 3 years
- Learn universal market patterns
- Save foundation checkpoint

### Phase 2: Initial Twin Fine-Tuning
- For each pilot stock:
  - Load foundation (frozen)
  - Fine-tune adaptation layers + heads
  - Save twin checkpoint

### Phase 3: Weekly Retraining
- Every Sunday 2 AM:
  - Update stock characteristics
  - Fine-tune twins on last 6 months
  - Deploy for Monday trading

## Pilot Stocks (50 stocks across 11 sectors)

- **Technology (7)**: AAPL, MSFT, NVDA, AMD, INTC, QCOM, AVGO
- **Financials (7)**: JPM, BAC, WFC, GS, MS, C, BLK
- **Healthcare (6)**: JNJ, UNH, PFE, ABBV, TMO, MRK
- **Consumer Discretionary (6)**: AMZN, TSLA, HD, MCD, NKE, SBUX
- **Consumer Staples (5)**: PG, KO, PEP, WMT, COST
- **Energy (5)**: XOM, CVX, COP, SLB, EOG
- **Communication Services (5)**: GOOGL, META, NFLX, DIS, CMCSA
- **Industrials (5)**: BA, CAT, UNP, HON, LMT
- **Materials (2)**: LIN, APD
- **Utilities (1)**: NEE
- **Real Estate (1)**: AMT

## Performance Targets

- **Hit rate**: 65%+ (vs 55% baseline)
- **Regime accuracy**: 75%+
- **Inference time**: <100ms per stock
- **Weekly training time**: 4-8 hours (50 stocks × 5-10 min each)

## Configuration

See `config/config.example.yaml` for:
- Foundation model configuration
- Twin configuration (50 pilot tickers)
- Training hyperparameters
- Checkpoint paths

## API Endpoints

- `GET /models/twins` - List all available twins
- `GET /models/twins/{ticker}` - Get twin info
- `GET /predictions/{ticker}/regime` - Get current regime

## References

- Complete Architecture Document: `docs/Stock Twin Complete Architecture & Design Document.pdf`
- Training Guide: `docs/TWIN_TRAINING.md`
- Regime Detection: `docs/REGIME_DETECTION.md`


