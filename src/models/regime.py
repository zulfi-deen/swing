"""Stock-level regime detection helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

REGIME_TRENDING = 0
REGIME_MEAN_REVERTING = 1
REGIME_CHOPPY = 2
REGIME_VOLATILE = 3

REGIME_NAMES = {
    REGIME_TRENDING: "Trending",
    REGIME_MEAN_REVERTING: "MeanReverting",
    REGIME_CHOPPY: "Choppy",
    REGIME_VOLATILE: "Volatile",
}

REGIME_STRATEGIES = {
    REGIME_TRENDING: {
        "strategy": "Follow momentum with pyramiding entries",
        "stop_loss_multiplier": 1.0,
        "position_size_multiplier": 1.0,
    },
    REGIME_MEAN_REVERTING: {
        "strategy": "Fade extremes and use tighter stops",
        "stop_loss_multiplier": 0.8,
        "position_size_multiplier": 0.75,
    },
    REGIME_CHOPPY: {
        "strategy": "Reduce exposure, focus on range edges",
        "stop_loss_multiplier": 0.6,
        "position_size_multiplier": 0.5,
    },
    REGIME_VOLATILE: {
        "strategy": "Trade smaller with wider stops",
        "stop_loss_multiplier": 1.4,
        "position_size_multiplier": 0.4,
    },
}


def detect_regime_features(stock_data: pd.DataFrame, stock_chars: Dict) -> int:
    """Classify the short-term behaviour of a stock."""

    if stock_data is None or stock_data.empty:
        return REGIME_CHOPPY

    closes = stock_data["close"].values
    if len(closes) < 5:
        return REGIME_CHOPPY

    returns = np.diff(closes) / closes[:-1]
    volatility = np.std(returns)
    trend = closes[-1] - closes[0]

    if volatility > 0.04:
        return REGIME_VOLATILE
    if abs(trend) / closes[0] > 0.05:
        return REGIME_TRENDING if trend > 0 else REGIME_MEAN_REVERTING
    if volatility < 0.01:
        return REGIME_MEAN_REVERTING
    return REGIME_CHOPPY


def get_regime_name(regime_id: int) -> str:
    return REGIME_NAMES.get(regime_id, "Unknown")


def get_regime_trading_strategy(regime_id: int) -> Dict:
    return REGIME_STRATEGIES.get(regime_id, REGIME_STRATEGIES[REGIME_CHOPPY])

