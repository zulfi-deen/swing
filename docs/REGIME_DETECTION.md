# Regime Detection Guide

## Overview

Regime detection identifies the current market behavior pattern for each stock, enabling regime-aware trading strategies.

## Regimes

### 1. Trending (ID: 0)

**Characteristics**:
- Strong directional move
- High trend strength (>1.5)
- High beta (>1.2)

**Trading Strategy**:
- Momentum with trailing stops
- Stop loss multiplier: 1.5x
- Position size: 100%
- Hold period: 5 days

### 2. MeanReverting (ID: 1)

**Characteristics**:
- Oscillating around mean
- Extreme RSI (>30 from 50)
- High mean-reversion strength (>0.6)

**Trading Strategy**:
- Counter-trend mean reversion
- Stop loss multiplier: 1.0x
- Position size: 80%
- Hold period: 3 days

### 3. Choppy (ID: 2)

**Characteristics**:
- No clear pattern
- Low volatility
- Random walk behavior

**Trading Strategy**:
- Avoid or tight ranges
- Stop loss multiplier: 1.0x
- Position size: 50%
- Hold period: 2 days

### 4. Volatile (ID: 3)

**Characteristics**:
- High volatility (>2x ATR)
- Erratic price movements
- High uncertainty

**Trading Strategy**:
- Wider stops, smaller positions
- Stop loss multiplier: 2.0x
- Position size: 60%
- Hold period: 3 days

## Detection Logic

The regime is detected using a combination of:

1. **Trend Strength**: `abs(mean_return) / volatility`
2. **RSI Extremes**: Distance from 50
3. **Volatility**: Relative to ATR or historical average
4. **Beta**: Stock sensitivity to market
5. **Hurst Exponent**: Mean-reversion tendency

### Rule-Based Detection

```python
from src.models.regime import detect_regime_features

regime_id = detect_regime_features(stock_data, stock_characteristics)
```

### Neural Network Detection

The digital twin's regime detector uses a neural network:

```python
regime_logits = twin.regime_detector(embeddings)
regime_probs = torch.softmax(regime_logits, dim=-1)
current_regime = torch.argmax(regime_probs, dim=-1)
```

## Usage

### Get Regime for a Ticker

```python
from src.models.regime import get_regime_name, get_regime_trading_strategy

regime_id = detect_regime_features(stock_data, stock_chars)
regime_name = get_regime_name(regime_id)
strategy = get_regime_trading_strategy(regime_id)
```

### API Endpoint

```bash
GET /predictions/{ticker}/regime?date=2024-01-15
```

Returns:
```json
{
  "ticker": "AAPL",
  "date": "2024-01-15",
  "regime_id": 0,
  "regime_name": "Trending",
  "trading_strategy": {
    "strategy": "momentum",
    "stop_loss_multiplier": 1.5,
    "position_size_multiplier": 1.0,
    "hold_period_days": 5
  }
}
```

## Regime Transitions

Regimes can change over time:
- **Trending → MeanReverting**: When trend exhausts and RSI becomes extreme
- **MeanReverting → Choppy**: When mean reversion weakens
- **Choppy → Volatile**: When volatility spikes
- **Volatile → Trending**: When volatility consolidates into trend

The digital twin tracks regime probabilities, allowing for smooth transitions.

## Best Practices

1. **Monitor Regime Changes**: Track regime transitions over time
2. **Regime-Aware Position Sizing**: Adjust position size based on regime
3. **Stop Loss Adaptation**: Use regime-specific stop loss multipliers
4. **Avoid Choppy Regimes**: Reduce exposure during choppy periods
5. **Volatile Regime Caution**: Use smaller positions and wider stops


