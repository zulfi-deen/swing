"""Simplified data ingestion flow used by the orchestrator."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from src.data.storage import save_to_local
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def daily_data_ingestion_flow(
    date: str,
    tickers: List[str],
    config: Optional[Dict] = None,
) -> Dict:
    """Placeholder ingestion flow.

    The production project relies on Prefect and multiple external APIs.  For the
    open repository we emulate the behaviour by writing a skeleton dataset to the
    local storage directory so the downstream stages can read deterministic data
    during tests.
    """

    cfg = config or load_config()
    logger.info("Running simplified ingestion flow for %s (%d tickers)", date, len(tickers))

    prices = pd.DataFrame(
        {
            "time": pd.date_range(end=date, periods=5).repeat(len(tickers)),
            "ticker": sorted(tickers) * 5,
            "close": 100.0,
            "volume": 1_000_000,
        }
    )
    save_to_local(prices, f"raw/prices/{date}/prices", cfg)

    news_payload = {ticker: [] for ticker in tickers}
    save_to_local(news_payload, f"raw/news/{date}/news", cfg)

    logger.info("Ingestion flow completed")
    return {
        "date": date,
        "tickers": len(tickers),
        "prices_rows": len(prices),
    }

