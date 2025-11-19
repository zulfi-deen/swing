"""Lightweight ARIMA/GARCH placeholder utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np


def fit_arima_garch(ticker: str, returns) -> Dict:
    """Return simple volatility statistics for the orchestrator ensemble."""

    if returns is None or len(returns) == 0:
        return {}

    arr = np.asarray(returns)
    mean_return = float(np.nanmean(arr))
    volatility = float(np.nanstd(arr) * np.sqrt(5))

    return {
        "ticker": ticker,
        "return": mean_return,
        "volatility": max(volatility, 1e-4),
    }

