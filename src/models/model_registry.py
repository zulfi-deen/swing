"""Singleton registry used by the FastAPI service."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from src.models.foundation import load_foundation_model
from src.models.twin_manager import TwinManager
from src.utils.config import load_config

logger = logging.getLogger(__name__)


class ModelRegistry:
    _twin_manager: Optional[TwinManager] = None
    _status: str = "uninitialized"

    @classmethod
    def get_twin_manager(cls, config: Optional[Dict] = None) -> Optional[TwinManager]:
        if cls._twin_manager is not None:
            return cls._twin_manager

        cfg = config or load_config()
        models_cfg = cfg.get("models", {})
        foundation_cfg = models_cfg.get("foundation", {})
        twins_cfg = models_cfg.get("twins", {})

        checkpoint_path = foundation_cfg.get("checkpoint_path")
        pilot_tickers = twins_cfg.get("pilot_tickers", [])

        if not pilot_tickers:
            cls._status = "missing_pilot_tickers"
            logger.warning("Pilot tickers not configured; TwinManager unavailable")
            return None

        foundation = load_foundation_model(checkpoint_path, foundation_cfg)
        foundation.freeze()

        cls._twin_manager = TwinManager(foundation, pilot_tickers, cfg)
        cls._status = "ready"
        return cls._twin_manager

    @classmethod
    def reset(cls) -> None:
        cls._twin_manager = None
        cls._status = "uninitialized"

    @classmethod
    def get_status(cls) -> str:
        return cls._status

