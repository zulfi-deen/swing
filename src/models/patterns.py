"""Chart pattern detection helpers."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def detect_chart_patterns(price_history: pd.DataFrame) -> Dict:
    """Detect a couple of simple price action patterns."""

    if price_history is None or price_history.empty:
        return {"pattern": "none", "confidence": 0.0}

    closes = price_history["close"].values
    if len(closes) < 5:
        return {"pattern": "none", "confidence": 0.0}

    diff = closes[-1] - closes[-5]
    pct_change = diff / closes[-5]

    if pct_change > 0.03:
        return {"pattern": "bullish_breakout", "confidence": min(1.0, pct_change * 10)}
    if pct_change < -0.03:
        return {"pattern": "bearish_breakdown", "confidence": min(1.0, abs(pct_change) * 10)}
    return {"pattern": "range", "confidence": 0.2}

