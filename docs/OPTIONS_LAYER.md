# Options Intelligence Layer

## Overview

The Options Intelligence Layer integrates options market data and signals into the swing trading system, providing forward-looking market microstructure information that equity data alone cannot reveal. This layer is based on the "Options Complete Architecture & Design Document v4.pdf" specification.

## Why Options Data?

Options reveal what equity data cannot:

| Equity Data | What It Shows | Limitation |
|------------|---------------|------------|
| Price | Past transactions | Backward-looking |
| Volume | Activity level | Doesn't show direction/intent |
| Technical Indicators | Pattern recognition | Derived from price (no new info) |

| Options Data | What It Shows | Edge |
|-------------|---------------|------|
| Open Interest | Outstanding positions | Shows conviction (not just trades) |
| Put/Call Ratio | Market sentiment | Contrarian signals at extremes |
| Gamma Exposure | Price magnetism | Reveals support/resistance zones |
| Implied Volatility | Expected volatility | Forward-looking risk assessment |
| Options Flow | Smart money activity | See institutional positioning |

## Architecture

The options layer integrates across all system layers:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Foundation Model (Universal Patterns)            │
│ - Uses equity features only                                │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Digital Twins (Stock-Specific Predictors)        │
│ - Foundation Encoder (frozen)                              │
│ - Options Encoder (40 → 32 dim) ← NEW                     │
│ - Fusion Layer (foundation + options)                      │
│ - Gamma Adjustment Head ← NEW                               │
│ - PCR Sentiment Gate ← NEW                                 │
│ - Prediction Heads (options-informed)                     │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Options Intelligence (Market Microstructure)      │
│ - Options Signals Aggregator                               │
│   ├─ PCR Extremes (Sentiment)                              │
│   ├─ Gamma Zones (Support/Resist)                          │
│   ├─ IV Regime (Vol Context)                               │
│   ├─ Net Positioning (Smart Money)                         │
│   └─ OI Confirmation (Trend Valid.)                        │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: RL Portfolio Manager (Learned Policy)            │
│ - State Encoder (includes options market signals) ← NEW    │
│ - Policy Network (options-informed biases) ← NEW           │
│ - Value Network                                            │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Acquisition (`src/data/sources.py`)

**PolygonOptionsClient**:
- Fetches full options chains for stocks
- Retrieves Greeks (delta, gamma, vega, theta, rho)
- Gets open interest, volume, and implied volatility
- Supports historical options data for backtesting

**Integration**:
- Added to `daily_data_ingestion_flow` in `src/data/ingestion.py`
- Fetches options data for configured tickers (default: top 100 liquid stocks)
- Saves raw options data to `data/raw/options/{date}/`
- Persists to TimescaleDB `options_prices` table

### 2. Options Feature Engineering (`src/features/options.py`)

**OptionsFeatureExtractor** extracts 40 features per stock per day:

#### Volume & Open Interest (8 features)
- Call/Put OI and volume
- Total OI and volume
- OI change percentage
- Volume z-score (vs 20-day average)

#### Put-Call Ratios (6 features)
- PCR OI and PCR volume
- PCR z-score (vs 60-day average)
- Extreme flags (bullish < 0.7, bearish > 1.0)
- PCR trend (change vs yesterday)

#### Gamma Exposure (7 features)
- Max pain strike (highest gamma concentration)
- Max pain distance (% from current price)
- Total gamma exposure (billions)
- Gamma sign (positive/negative)
- Gamma concentration (distribution peakedness)
- Gamma flip strike (where gamma changes sign)
- Gamma flip distance

#### Implied Volatility (7 features)
- ATM call/put IV
- IV skew (put IV - call IV)
- Put/call IV ratio
- IV percentile (vs 1-year history)
- IV rank (normalized 0-1)
- IV change percentage

#### Net Greeks (5 features)
- Net delta (directional bias)
- Net gamma (convexity)
- Net vega (vol exposure)
- Net theta (time decay)
- Net delta absolute (bias strength)

#### Term Structure (4 features)
- Front month OI
- Next month OI
- Roll ratio (front/next)
- Term curve slope (backwardation/contango)

#### Composite Signals (3 features)
- Trend signal: OI + price confirmation (-1 to +1)
- Sentiment signal: PCR extreme contrarian (-1 to +1)
- Gamma signal: Near gamma zone (0/1)

### 3. Digital Twin Integration (`src/models/digital_twin.py`)

**Options Encoder**:
- Input: 40 options features
- Architecture: Linear(40→64) → LayerNorm → ReLU → Linear(64→32) → LayerNorm → ReLU
- Output: 32-dim options embeddings

**Gamma Adjustment Head**:
- Uses options embeddings to predict max pain corrections
- Output: Adjustment in [-1, 1] range
- Applied to expected return: `return = return + gamma_adj * 0.01`

**PCR Sentiment Gate**:
- Uses options embeddings to detect sentiment extremes
- Output: Gate value in [0, 1]
- Applied to hit probability: `prob = prob * (0.5 + 0.5 * pcr_gate)`

**Fusion**:
- Options embeddings concatenated with foundation + stock + regime embeddings
- Used in regime detection and correction layers
- Enhances stock-specific predictions with options intelligence

### 4. RL Portfolio Agent Integration (`src/agents/portfolio_rl_agent.py`)

**Options Market Encoder**:
- Aggregates options signals across all stocks
- Input: 9 key options features per stock
- Output: 16-dim market-level options representation
- Added to global state fusion

**Stock Encoder Enhancement**:
- Per-stock encoder now includes 9 options features:
  - trend_signal, sentiment_signal, gamma_signal
  - pcr_zscore, pcr_extreme_bullish, pcr_extreme_bearish
  - max_pain_distance_pct, iv_percentile, net_delta
- Options features concatenated with twin predictions and technical features

**Policy Network**:
- Stock selection and position sizing now informed by options signals
- Agent learns to weight options signals appropriately through training

### 5. Monitoring & Evaluation (`src/evaluation/options_metrics.py`)

**OptionsMetricsTracker**:
- Tracks options prediction accuracy:
  - PCR extreme accuracy (contrarian signal effectiveness)
  - Gamma zone accuracy (price magnetism validation)
  - OI trend confirmation accuracy
- Evaluates historical performance of options signals

**Options Monitoring Dashboard**:
- Real-time alerts for:
  - Extreme PCR readings (contrarian opportunities)
  - High gamma concentration (price magnetism)
  - IV spikes (volatility regime change)
  - Unusual OI changes (smart money flows)

### 6. Explainability (`src/agents/explainer.py`)

**explain_trade_with_options()**:
- Generates human-readable explanations incorporating:
  - Twin predictions (expected return, probability, regime)
  - Options signals (PCR extremes, gamma zones, IV regime)
  - How options confirm or contradict equity signals
  - Risk factors from options (max pain, IV spikes)

## Configuration

Enable options layer in `config/config.example.yaml`:

```yaml
data_sources:
  options:
    enabled: true  # Enable options data ingestion
    provider: "polygon"
    tickers: []  # Empty = use all tickers, or specify subset
    max_contracts_per_ticker: 1000

models:
  twins:
    options_enabled: true  # Enable options encoder in twins
    options_embedding_dim: 32

rl_portfolio:
  options_enabled: true  # Enable options signals in RL agent
```

## Data Flow

### Daily Pipeline

1. **5:05 PM - Data Ingestion**
   - Equity data (OHLCV, volume)
   - **Options data (OI, Greeks, PCR, IV)** ← NEW
   - News/text data

2. **5:10 PM - Feature Engineering**
   - Technical indicators (25 features)
   - **Options features (40 features)** ← NEW
   - Text processing via LLM (35 features)
   - Cross-sectional + patterns (23 features)

3. **5:15 PM - Foundation Model Inference**
   - Universal embeddings for all 500 stocks

4. **5:18 PM - Digital Twin Inference**
   - 500 twins predict return/prob/vol/regime
   - **Options encoder processes 40 features → 32-dim embeddings** ← NEW
   - **Gamma adjustment applied to returns** ← NEW
   - **PCR sentiment gate applied to probabilities** ← NEW

5. **5:21 PM - RL Portfolio Manager**
   - State: twins + **options** + portfolio + macro
   - Action: select 15 stocks + position sizes
   - **Options-informed biases applied** ← NEW

6. **5:23 PM - LLM Explainer**
   - Generate human-readable rationale
   - **Includes options context (PCR, gamma, IV)** ← NEW

## Options Signal Interpretation

### PCR Signals
- **PCR < 0.7**: Extreme bullish → Contrarian bearish (reduce longs, consider shorts)
- **PCR > 1.0**: Extreme bearish → Contrarian bullish (reduce shorts, consider longs)
- **PCR 0.7-1.0**: Neutral (no extreme)

### Gamma Signals
- **Near max pain (< 3%)**: Price magnetism → Set target near max pain
- **High gamma concentration (> 0.7)**: Strong support/resistance
- **Positive net gamma**: Market makers short gamma → Stabilizing
- **Negative net gamma**: Market makers long gamma → Volatility amplification

### IV Signals
- **IV percentile > 80%**: Expensive options → Expect IV contraction (sell premium)
- **IV percentile < 20%**: Cheap options → Expect IV expansion (buy premium)
- **High IV skew**: Downside protection expensive → Bearish bias

### OI Signals
- **OI ↑ + Price ↑**: Bullish confirmation
- **OI ↑ + Price ↓**: Bearish confirmation
- **OI ↓**: Position unwinding (trend exhaustion)

## Storage Schema

### TimescaleDB Tables

**options_prices** (hypertable):
- Raw options chain data per contract
- Columns: time, ticker, strike_price, expiration, option_type, bid, ask, last_price, delta, gamma, vega, theta, rho, volume, open_interest, implied_volatility

**options_features** (hypertable):
- Processed 40 features per stock per day
- All 40 features stored with time and ticker

## Performance Expectations

Based on v4 architecture document:

- **24% annual return** (vs. 18% without options)
- **2.5 Sharpe ratio** (vs. 1.8 without options)
- **72% win rate** (vs. 65% without options)
- **-8% max drawdown** (vs. -12% without options)

## Cost Considerations

- **Polygon Starter Plan**: Required for options data
- **API Costs**: ~$249/month for options data access
- **Storage**: Additional ~50GB for options data (raw + features)
- **Compute**: Minimal overhead (~5% increase in inference time)

## Future Enhancements

1. **Options Flow Analysis**: Track large block trades and unusual activity
2. **Volatility Surface Modeling**: Full IV surface analysis beyond ATM
3. **Options Strategy Generation**: Suggest optimal options strategies (calls, puts, spreads)
4. **Options Backtesting**: Historical options strategy performance
5. **Real-Time Options Alerts**: WebSocket-based live options monitoring

