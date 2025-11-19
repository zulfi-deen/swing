"""Market-level (SPY) regime detection utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

REGIME_BEAR = 0
REGIME_SIDEWAYS = 1
REGIME_BULL = 2


def detect_market_regime(spy_prices: pd.DataFrame) -> int:
    if spy_prices is None or spy_prices.empty:
        return REGIME_SIDEWAYS

    closes = spy_prices["close"].values
    if len(closes) < 20:
        return REGIME_SIDEWAYS

    trend = closes[-1] - closes[0]
    pct_change = trend / closes[0]
    volatility = np.std(np.diff(closes) / closes[:-1])

    if pct_change > 0.03 and volatility < 0.02:
        return REGIME_BULL
    if pct_change < -0.03:
        return REGIME_BEAR
    return REGIME_SIDEWAYS


def get_market_regime_name(regime_id: int) -> str:
    return {REGIME_BEAR: "Bear", REGIME_SIDEWAYS: "Sideways", REGIME_BULL: "Bull"}.get(
        regime_id, "Unknown"
    )


def get_market_regime_features(spy_prices: pd.DataFrame) -> Dict:
    regime = detect_market_regime(spy_prices)
    closes = spy_prices["close"].values if spy_prices is not None else []
    volatility = float(np.std(np.diff(closes) / closes[:-1])) if len(closes) > 1 else 0.0
    trend_strength = float((closes[-1] - closes[0]) / closes[0]) if len(closes) > 1 else 0.0
    return {
        "regime": regime,
        "regime_name": get_market_regime_name(regime),
        "volatility": volatility,
        "trend_strength": trend_strength,
    }

