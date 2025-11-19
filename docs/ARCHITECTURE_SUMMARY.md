# Architecture Summary: v1-v4 Evolution

This document summarizes the key requirements and evolution across architecture versions v1-v4, serving as a reference for code audits and implementation verification.

## Architecture Evolution

### v1: Base Architecture (TFT-GNN Hybrid)
**Core Components:**
- Temporal Fusion Transformer (TFT) for temporal patterns
- Graph Neural Network (GNN) for cross-stock relationships
- Ensemble of TFT-GNN, LightGBM ranker, ARIMA/GARCH
- LLM agents for text processing and policy enforcement
- Technical analysis and chart pattern detection

**Key Metrics:**
- Target: 65% win rate with 2:1 reward/risk ratio
- Processes 500+ stocks daily
- 5-day prediction horizon

**Data Flow:**
- Market data (Polygon) → TimescaleDB
- News/Sentiment (Finnhub) → Text features
- Features → TFT-GNN → Ensemble → Priority Scoring → Recommendations

### v2: Digital Twins Architecture
**Core Innovation:**
- Foundation Model: Universal market intelligence (TFT + GNN) trained on all 500 stocks
- 500 Digital Twins: Stock-specific models fine-tuned from foundation
- Hierarchical Learning: Transfer learning from universal → stock-specific patterns

**Key Metrics:**
- Return prediction accuracy: +29% vs. single model
- Hit probability calibration: +35% improvement
- Regime detection: 78% accuracy (vs. 62% baseline)

**Architecture:**
```
Layer 1: Foundation Model (Universal)
  ↓
Layer 2: Digital Twins (500 Stock-Specific)
  ↓
Layer 3: LLM Agents (Context & Policy)
```

**Training:**
- Foundation: One-time pre-training on all stocks × 3 years
- Twins: Weekly fine-tuning on last 6 months per stock

### v3: RL Portfolio Manager
**Core Innovation:**
- Replaces hand-crafted priority scoring with learned PPO policy
- Optimizes portfolio-level objectives (Sharpe ratio, drawdown)
- Discovers non-linear signal interactions
- Adapts to market regimes automatically

**State Space:**
- Twin predictions (500 stocks × 7 outputs)
- Per-stock features (technical, sentiment, patterns)
- Portfolio state (positions, sector exposure, cash)
- Macro context (VIX, SPY returns, treasury yields)
- Correlation graph (GNN input)

**Action Space:**
- Stock selection: Multi-label binary (select top N stocks)
- Position sizing: Beta distribution per selected stock

**Reward Function:**
- Portfolio return (scaled 100x)
- Drawdown penalty
- Transaction costs
- Diversification bonus
- Sharpe adjustment
- Constraint penalties

**Expected Improvements:**
- Sharpe Ratio: 1.8 → 2.3 (+28%)
- Max Drawdown: -12% → -8% (-33%)
- Win Rate: 65% → 70% (+5pp)

### v4: Options Intelligence Layer
**Core Innovation:**
- 40 options features per stock per day
- Options encoder in Digital Twins (40 → 32-dim embeddings)
- Gamma adjustment & PCR sentiment gates
- Options-informed RL portfolio manager
- Options-aware explanations

**40 Options Features:**
1. Volume & Open Interest (8): call/put OI, volume, OI change, volume z-score
2. Put-Call Ratios (6): PCR OI/volume, z-scores, extreme flags, trend
3. Gamma Exposure (7): max pain strike, gamma concentration, flip levels
4. Implied Volatility (7): ATM IV, skew, percentile, rank, term structure
5. Net Greeks (5): net delta, gamma, vega, theta, delta strength
6. Term Structure (4): front/next month OI, roll ratio, curve slope
7. Composite Signals (3): trend confirmation, sentiment extreme, gamma zone

**Digital Twin Integration:**
- Options Encoder: 40 features → 32-dim embeddings
- Gamma Adjustment Head: Adjusts expected return based on max pain distance
- PCR Sentiment Gate: Modulates hit probability based on sentiment extremes

**RL Integration:**
- Options Market Encoder: Aggregates options signals across stocks
- Stock Encoder Enhancement: Includes 9 key options features per stock
- Policy Network: Options-informed stock selection and position sizing

**Expected Performance (v4 vs v3):**
- Annual Return: 18% → 24% (+33%)
- Sharpe Ratio: 1.8 → 2.5 (+39%)
- Win Rate: 65% → 72% (+11%)
- Max Drawdown: -12% → -8% (-33%)

## Four-Layer Stack (v4 Target)

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Foundation Model (Universal Market Intelligence)   │
│ - TFT: Temporal patterns across all stocks                  │
│ - GNN: Cross-stock relationships & sector dynamics          │
│ - Output: 128-dim embeddings                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Digital Twins (500 Stock-Specific Predictors)      │
│ - Foundation Encoder (frozen)                               │
│ - Options Encoder (40 → 32 dim) [if enabled]               │
│ - Gamma Adjustment Head [if enabled]                       │
│ - PCR Sentiment Gate [if enabled]                           │
│ - Output: expected_return, hit_prob, volatility, quantiles   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Options Intelligence (Market Microstructure)       │
│ - PCR Extremes (Sentiment)                                  │
│ - Gamma Zones (Support/Resist)                              │
│ - IV Regime (Vol Context)                                   │
│ - Net Positioning (Smart Money)                             │
│ - OI Confirmation (Trend Validation)                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: RL Portfolio Manager (Learned Policy)              │
│ - State Encoder (includes options market signals)           │
│ - Policy Network (options-informed biases)                  │
│ - Value Network                                             │
│ - Output: Stock selection + position sizes                  │
└─────────────────────────────────────────────────────────────┘
```

## Success Metrics & Constraints

### Performance Targets (v4)
- **Annual Return**: 24% (vs. 18% without options)
- **Sharpe Ratio**: 2.5 (vs. 1.8 without options)
- **Win Rate**: 72% (vs. 65% without options)
- **Max Drawdown**: -8% (vs. -12% without options)

### Operational Constraints
- **Max Positions**: 15 concurrent positions
- **Max Sector Exposure**: 25% per sector
- **Max Position Size**: 10% per position
- **Min Probability**: 60% for trade entry
- **Transaction Cost**: 0.1% per trade
- **Prediction Horizon**: 5 days

### Data Requirements
- **Foundation Training**: 3+ years of data, 100-500 stocks
- **Twin Fine-Tuning**: 6 months per stock
- **RL Training**: 1+ year of historical predictions
- **Options Data**: Polygon Starter plan required (~$249/month)

## Key Design Principles

1. **LLMs as Context Processors**: Use LLMs to structure text and enforce rules, NOT predict prices
2. **Multi-Scale Signal Fusion**: Combine temporal (TFT), relational (GNN), statistical (ARIMA), and textual (LLM) signals
3. **Explainability by Design**: Every recommendation includes human-readable rationale with options context
4. **Production-Ready**: Cloud-native, scalable, monitored, with proper CI/CD
5. **Digital Twins Capture Idiosyncratic Risk**: Each stock's unique behavioral "fingerprint"
6. **RL Learns Optimal Policy**: Replaces arbitrary hand-tuned weights with learned relationships
7. **Options Reveal Forward-Looking Signals**: Smart money positioning and market microstructure

## Data Flow (Complete v4 Pipeline)

1. **Data Ingestion** (5:05 PM)
   - Equity data (OHLCV, volume)
   - Options data (OI, Greeks, IV) [if enabled]
   - News/text data
   - Fundamentals
   - Macro data (FRED, VIX)

2. **Feature Engineering** (5:10 PM)
   - Technical indicators (25 features)
   - Options features (40 features) [if enabled]
   - Text processing via LLM (35 features)
   - Cross-sectional + patterns (23 features)

3. **Foundation Model Inference** (5:15 PM)
   - Universal embeddings for all 500 stocks

4. **Digital Twin Inference** (5:18 PM)
   - 500 twins predict return/prob/vol/regime
   - Options encoder processes 40 features → 32-dim embeddings [if enabled]
   - Gamma adjustment applied to returns [if enabled]
   - PCR sentiment gate applied to probabilities [if enabled]

5. **RL Portfolio Manager** (5:21 PM)
   - State: twins + options + portfolio + macro
   - Action: select 15 stocks + position sizes
   - Options-informed biases applied [if enabled]

6. **LLM Explainer** (5:23 PM)
   - Generate human-readable rationale
   - Includes options context (PCR, gamma, IV) [if enabled]

7. **Save & Serve** (5:25 PM)
   - Recommendations → TimescaleDB
   - API endpoints → React Dashboard

## Configuration Flags

### Options Layer
- `data_sources.options.enabled`: Enable/disable options data ingestion
- `models.twins.options_enabled`: Enable options encoder in twins
- `rl_portfolio.options_enabled`: Enable options signals in RL agent

### RL Portfolio
- `rl_portfolio.enabled`: Use RL agent instead of priority scoring
- `rl_portfolio.checkpoint_path`: Path to trained RL agent checkpoint

### Digital Twins
- `models.twins.pilot_tickers`: List of tickers with active twins
- `models.foundation.checkpoint_path`: Path to foundation model checkpoint

## Critical Integration Points

1. **Options Data → Features**: `src/features/options.py` → `src/pipeline/orchestrator.py::_compute_options_features()`
2. **Options Features → Twins**: Options features passed to `StockDigitalTwin.forward()` via `options_features` tensor
3. **Twin Predictions → RL**: Twin outputs passed to `RLStateBuilder.build_state()` → `PortfolioRLAgent`
4. **RL State → Agent**: Complete state (twin predictions + features + portfolio + macro + options + graph) → RL agent
5. **RL Action → Ranking**: RL attention scores replace priority scores in `_rank_predictions()`

## Known Gaps & Future Work

1. **Feast Feature Store**: Code exists but not integrated into pipeline
2. **React Dashboard**: Backend exists, frontend missing
3. **Heterogeneous Graph**: Code exists but not used (homogeneous graph used instead)
4. **Supply Chain Knowledge Graph**: Basic correlation-based peer finding only

## References

- v1: `docs/Complete Architecture & Design Document v1.pdf`
- v2: `docs/Stock Twin Complete Architecture & Design Document v2.pdf`
- v3: `docs/RL Portfolio complete architecture and design document v3 .pdf`
- v4: `docs/Options Complete Architecture & Design Document v4.pdf`
- Implementation Status: `docs/IMPLEMENTATION_STATUS.md`

