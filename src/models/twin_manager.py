"""TwinManager orchestration utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.models.digital_twin import StockDigitalTwin
from src.utils.config import load_config

logger = logging.getLogger(__name__)


class TwinManager:
    """Simple in-memory twin registry used by the orchestrator and API."""

    def __init__(
        self,
        foundation_model,
        pilot_tickers: List[str],
        config: Optional[Dict] = None,
    ):
        self.foundation = foundation_model
        self.pilot_tickers = pilot_tickers or []
        self.config = config or load_config()
        self.twins: Dict[str, StockDigitalTwin] = {}
        self.checkpoint_dir = (
            Path(
                self.config.get("models", {})
                .get("twins", {})
                .get("checkpoint_dir", "models/twins")
            )
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _load_stock_characteristics(self, ticker: str) -> Dict:
        try:
            from src.data.storage import get_stock_characteristics
        except Exception:  # pragma: no cover - circular imports safety
            return {}
        return get_stock_characteristics(ticker) or {}

    def _load_from_checkpoint(self, ticker: str) -> Optional[StockDigitalTwin]:
        checkpoint_path = self.checkpoint_dir / ticker / "twin_latest.pt"
        if checkpoint_path.exists():
            twin = StockDigitalTwin(
                self.foundation,
                ticker,
                self._load_stock_characteristics(ticker),
                self.config,
            )
            twin.load_checkpoint(str(checkpoint_path))
            twin.eval()
            return twin
        return None

    def get_twin(self, ticker: str) -> Optional[StockDigitalTwin]:
        if ticker not in self.pilot_tickers:
            return None

        if ticker not in self.twins:
            twin = self._load_from_checkpoint(ticker)
            if twin is None:
                stock_chars = self._load_stock_characteristics(ticker)
                twin = StockDigitalTwin(self.foundation, ticker, stock_chars, self.config)
            self.twins[ticker] = twin
        return self.twins[ticker]

    def list_twins(self) -> List[Dict]:
        details = []
        for ticker in self.pilot_tickers:
            checkpoint_exists = (self.checkpoint_dir / ticker / "twin_latest.pt").exists()
            details.append(
                {
                    "ticker": ticker,
                    "has_checkpoint": checkpoint_exists,
                }
            )
        return details

    def save_all(self) -> None:
        for ticker, twin in self.twins.items():
            out_dir = self.checkpoint_dir / ticker
            out_dir.mkdir(parents=True, exist_ok=True)
            twin.save_checkpoint(str(out_dir / "twin_latest.pt"))

    def to_json(self) -> str:
        return json.dumps(self.list_twins())

