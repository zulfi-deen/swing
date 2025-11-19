"""Stubs for external data sources (Polygon, Finnhub, FRED, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class FREDClient:
    api_key: Optional[str] = None

    def get_indicator(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq="W")
        return pd.DataFrame({"date": dates, "value": 0.0, "series": series_id})


@dataclass
class VIXClient:
    def get_latest(self) -> Dict:
        return {"value": 20.0}

