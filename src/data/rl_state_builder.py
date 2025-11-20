"""RL state builder utilities."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:  # pragma: no cover - optional dependency
    from torch_geometric.data import Data as GraphData

    TORCH_GEOMETRIC_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    GraphData = object
    TORCH_GEOMETRIC_AVAILABLE = False

from src.utils.config import load_config


class RLStateBuilder:
    """Converts model outputs into a single state dictionary for the RL agent."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or load_config()
        rl_cfg = self.config.get("rl_portfolio", {})
        self.options_enabled = rl_cfg.get("options_enabled", False)

    def build_state(
        self,
        twin_predictions: Dict[str, Dict],
        features_df: Optional[pd.DataFrame],
        portfolio_state: Optional[Dict],
        macro_context: Optional[Dict],
        prices_df: Optional[pd.DataFrame] = None,
        date: Optional[str] = None,
        config: Optional[Dict] = None,
        options_features: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        cfg = config or self.config
        tickers = sorted(list(twin_predictions.keys()))

        feature_map = {}
        if isinstance(features_df, pd.DataFrame) and not features_df.empty:
            for _, row in features_df.iterrows():
                ticker = row.get("ticker")
                if ticker is None:
                    continue
                feature_map[ticker] = row.drop(labels=["ticker"], errors="ignore").to_dict()

        if options_features is None:
            options_features = {}
        if not self.options_enabled:
            options_features = {}

        graph, ticker_to_idx = self._build_graph(tickers, prices_df)

        return {
            "tickers": tickers,
            "twin_predictions": twin_predictions,
            "features": feature_map,
            "portfolio": portfolio_state or {},
            "macro": macro_context or {},
            "correlation_graph": graph,
            "ticker_to_idx": ticker_to_idx,
            "options_features": options_features,
            "date": date,
            "config": cfg,
        }

    def _build_graph(
        self, tickers: Dict, prices_df: Optional[pd.DataFrame]
    ) -> Tuple[Optional[GraphData], Dict[str, int]]:
        ticker_to_idx = {ticker: idx for idx, ticker in enumerate(tickers)}
        if not tickers or not TORCH_GEOMETRIC_AVAILABLE:
            return None, ticker_to_idx

        if prices_df is None or prices_df.empty:
            # Create a fully connected graph with equal weights
            edge_index = torch.combinations(torch.arange(len(tickers)), r=2).t().contiguous()
            data = GraphData(edge_index=edge_index)
            return data, ticker_to_idx

        pivot = prices_df.pivot_table(index="time", columns="ticker", values="close")
        correlation = pivot.corr().fillna(0.0).values
        edges = np.where(np.abs(correlation) > 0.3)
        if edges[0].size == 0:
            return None, ticker_to_idx
        edge_index = torch.tensor(np.vstack(edges), dtype=torch.long)
        data = GraphData(edge_index=edge_index)
        return data, ticker_to_idx

