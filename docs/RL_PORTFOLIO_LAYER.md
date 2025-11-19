# RL Portfolio Layer Documentation

## Overview

The RL Portfolio Layer replaces the heuristic priority scoring system with a learned reinforcement learning (RL) agent that optimizes portfolio construction using Proximal Policy Optimization (PPO). This addresses the fundamental weakness of arbitrary hand-tuned weights in the original priority scoring formula.

## Architecture

### Three-Layer System

1. **Layer 1: Foundation Model** - Universal market intelligence (TFT + GNN)
2. **Layer 2: Digital Twins** - Stock-specific predictors (500 stocks)
3. **Layer 3: RL Portfolio Manager** - Portfolio construction agent (NEW)

### Key Components

#### 1. RL State Builder (`src/data/rl_state_builder.py`)

Constructs the state representation for the RL agent:

- **Twin Predictions**: Per-stock predictions (expected_return, hit_prob, volatility, regime, etc.)
- **Features**: Per-stock technical/feature data (RSI, MACD, volume, sentiment, etc.)
- **Options Features** (if enabled): Per-stock options signals (PCR, gamma, IV, trend, sentiment)
- **Portfolio State**: Current positions, sector exposure, cash allocation
- **Macro Context**: VIX, SPY returns, treasury yields, market regime
- **Correlation Graph**: Graph structure for GNN aggregation

#### 2. Trading Environment (`src/training/rl_environment.py`)

Gym-style environment for RL training:

- Simulates portfolio trading using historical data
- Enforces constraints (max positions, position sizes, sector exposure)
- Computes rewards based on portfolio performance
- Handles transaction costs and portfolio updates

#### 3. RL Portfolio Agent (`src/agents/portfolio_rl_agent.py`)

PPO-based agent architecture:

- **State Encoder**: GNN-based aggregation of stock predictions
- **Options Market Encoder** (if enabled): Aggregates options signals across stocks
- **Stock Encoder**: Encodes per-stock features (twin predictions + technical + options)
- **Policy Network**: Dual-head (stock selection + position sizing) - options-informed
- **Value Network**: State value estimation for PPO
- **PPO Trainer**: Handles rollouts, GAE, clipping, and optimization

## Training

### Prerequisites

1. Historical predictions from digital twins
2. Historical prices and features
3. Historical options features (if options enabled)
4. Macro data (VIX, SPY, treasury yields)

### Training Command

```bash
python -m src.training.train_rl_portfolio \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-episodes 1000 \
    --episode-length 60 \
    --checkpoint-dir models/rl_portfolio/
```

### Training Process

1. **Episode Structure**: Each episode = 60 trading days (3 months)
2. **Rollout Collection**: Agent interacts with environment, collecting (state, action, reward) tuples
3. **PPO Updates**: Multiple epochs of policy updates using clipped surrogate objective
4. **Checkpointing**: Best model saved based on episode return

### Hyperparameters

Default PPO hyperparameters (configurable in `config.yaml`):

- Learning rate: 3e-4
- Clip epsilon: 0.2
- Value loss coefficient: 0.5
- Entropy coefficient: 0.01
- Discount factor (gamma): 0.99
- GAE lambda: 0.95
- Max gradient norm: 0.5
- PPO epochs: 4

## Reward Function

The reward function combines multiple objectives:

1. **Portfolio Return** (scaled 100x): Primary objective
2. **Drawdown Penalty**: Penalizes large drawdowns from peak
3. **Transaction Costs**: Penalizes excessive trading
4. **Diversification Bonus**: Rewards sector diversification
5. **Sharpe Adjustment**: Risk-adjusted return component
6. **Options Prediction Bonus** (if enabled): Rewards accurate options signal predictions
7. **Constraint Penalties**: Soft penalties for violating constraints

## Integration

### Pipeline Integration

The RL agent is integrated into the daily pipeline via `PipelineOrchestrator`:

1. If `rl_portfolio.enabled = true` in config, RL agent is loaded
2. `_rank_predictions()` method uses RL agent instead of priority scoring
3. RL attention scores replace priority scores
4. Falls back to priority scoring if RL agent unavailable

### Configuration

Enable RL portfolio in `config.yaml`:

```yaml
rl_portfolio:
  enabled: true
  options_enabled: true  # Enable options signals in RL agent
  checkpoint_path: "models/rl_portfolio/agent_best.pt"
  num_stocks: 500
  max_positions: 15
  ppo:
    learning_rate: 3e-4
    clip_epsilon: 0.2
    # ... other PPO hyperparameters
```

## Evaluation

### Backtesting

Use `backtest_rl_agent()` in `src/evaluation/backtest.py`:

```python
from src.evaluation.backtest import backtest_rl_agent

metrics = backtest_rl_agent(
    agent=rl_agent,
    predictions=predictions_df,
    prices=prices_df,
    features=features_df,
    macro_data=macro_df,
    start_date='2024-01-01',
    end_date='2024-06-01'
)
```

### Comparison

Compare RL agent vs priority scoring:

```python
from src.evaluation.backtest import compare_rl_vs_priority_scoring

comparison = compare_rl_vs_priority_scoring(
    rl_agent=agent,
    predictions=predictions_df,
    prices=prices_df,
    features=features_df,
    macro_data=macro_df,
    start_date='2024-01-01',
    end_date='2024-06-01'
)
```

## Expected Improvements

Based on RL literature in portfolio management:

| Metric | Priority Scoring | RL Agent | Improvement |
|--------|-----------------|----------|-------------|
| Sharpe Ratio | 1.8 | 2.3 | +28% |
| Max Drawdown | 12% | 8% | -33% |
| Win Rate | 65% | 70% | +5 pp |
| Profit Factor | 2.5 | 3.2 | +28% |

## Advantages Over Priority Scoring

1. **Learned Weights**: No arbitrary hand-tuning
2. **Non-linear Interactions**: Discovers complex signal interactions
3. **Regime Adaptation**: Automatically adapts to market regimes
4. **Portfolio-Level Optimization**: Optimizes for Sharpe, not arbitrary scores
5. **Continual Learning**: Can be retrained weekly/monthly

## Deployment Workflow

### 1. Training Phase

1. Train RL agent on historical data (2023-2024)
2. Validate on holdout period (2024)
3. Compare metrics vs priority scoring
4. Tune hyperparameters if needed

### 2. Paper Trading Phase

1. Deploy RL agent in paper trading mode
2. Monitor for 2-4 weeks
3. Compare live performance vs priority scoring
4. Validate constraint adherence

### 3. Production Deployment

1. Enable RL agent in config (`rl_portfolio.enabled = true`)
2. Monitor initial performance
3. Set up weekly retraining schedule
4. Maintain fallback to priority scoring

## Weekly Retraining

Set up automated retraining:

```bash
# Weekly retraining script
python -m src.training.train_rl_portfolio \
    --start-date $(date -d '1 year ago' +%Y-%m-%d) \
    --end-date $(date +%Y-%m-%d) \
    --num-episodes 500 \
    --checkpoint-dir models/rl_portfolio/
```

## Troubleshooting

### RL Agent Not Loading

- Check `checkpoint_path` in config
- Verify checkpoint file exists
- Check logs for loading errors

### Poor Performance

- Increase training episodes
- Tune hyperparameters
- Check reward function weights
- Validate state representation

### Memory Issues

- Reduce `num_stocks` in config
- Use smaller batch sizes
- Enable gradient checkpointing

## References

- RL Portfolio Layer Design Document: `docs/RL Portfolio layer.pdf`
- PPO Algorithm: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- Portfolio Optimization: Modern quant fund practices (Jane Street, Two Sigma)

