"""Stock-specific digital twin implementation."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.regime import detect_regime_features


class StockDigitalTwin(nn.Module):
    """Lightweight stock-specific adapter that sits on top of the foundation."""

    def __init__(
        self,
        foundation_model: nn.Module,
        ticker: str,
        stock_characteristics: Optional[Dict] = None,
        config: Optional[Dict] = None,
    ):
        super().__init__()

        self.foundation = foundation_model
        self.ticker = ticker
        self.stock_characteristics = stock_characteristics or {}
        self.config = config or {}

        twin_cfg = self.config.get("models", {}).get("twins", {})
        self.adapter_rank = int(twin_cfg.get("adapter_rank", 16))
        self.stock_embedding_dim = int(twin_cfg.get("stock_embedding_dim", 64))
        self.regime_embedding_dim = int(twin_cfg.get("regime_embedding_dim", 32))
        self.options_enabled = twin_cfg.get("options_enabled", False)
        self.options_embedding_dim = int(twin_cfg.get("options_embedding_dim", 32))

        self.embedding_dim = getattr(self.foundation, "embedding_dim", 128)

        self.down_project = nn.Linear(self.embedding_dim, self.adapter_rank)
        self.up_project = nn.Linear(self.adapter_rank, self.embedding_dim)
        self.stock_embedding = nn.Parameter(
            torch.zeros(self.stock_embedding_dim), requires_grad=True
        )
        self.regime_embeddings = nn.Embedding(4, self.regime_embedding_dim)

        fusion_input_dim = (
            self.embedding_dim + self.stock_embedding_dim + self.regime_embedding_dim
        )

        if self.options_enabled:
            self.options_encoder = nn.Sequential(
                nn.Linear(40, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, self.options_embedding_dim),
                nn.LayerNorm(self.options_embedding_dim),
                nn.ReLU(),
            )
            fusion_input_dim += self.options_embedding_dim
        else:
            self.options_encoder = None

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
        )

        self.return_head = nn.Linear(96, 1)
        self.hit_prob_head = nn.Linear(96, 1)
        self.vol_head = nn.Linear(96, 1)
        self.quantile_head = nn.Linear(96, 3)
        self.regime_head = nn.Linear(96, 4)

    def get_stock_characteristics(self) -> Dict:
        """Expose characteristics for diagnostics."""
        return {
            "ticker": self.ticker,
            **self.stock_characteristics,
        }

    def forward(
        self,
        batch: Dict,
        graph=None,
        options_features: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Forward pass returning the standard prediction dictionary."""

        features_tensor = None
        if isinstance(batch, dict):
            if "features" in batch:
                features_tensor = batch["features"]
            elif "encoder_cont" in batch:
                features_tensor = batch["encoder_cont"]
        else:
            features_tensor = batch

        if features_tensor is None:
            raise ValueError("Batch does not contain encoder features")

        foundation_embedding = self.foundation.encode(features_tensor)
        adapted = foundation_embedding + self.up_project(
            torch.relu(self.down_project(foundation_embedding))
        )

        stock_embed = self.stock_embedding.expand(adapted.size(0), -1)

        regime_index = torch.zeros(adapted.size(0), dtype=torch.long, device=adapted.device)
        regime_embed = self.regime_embeddings(regime_index)

        fusion_parts = [adapted, stock_embed, regime_embed]

        if self.options_enabled and options_features is not None:
            if options_features.dim() == 2:
                opts = options_features
            else:
                opts = options_features.view(options_features.size(0), -1)
            fusion_parts.append(self.options_encoder(opts))

        fused = self.fusion(torch.cat(fusion_parts, dim=-1))

        expected_return = self.return_head(fused).squeeze(-1)
        hit_prob = torch.sigmoid(self.hit_prob_head(fused)).squeeze(-1)
        volatility = torch.nn.functional.softplus(self.vol_head(fused)).squeeze(-1)
        quantiles = self.quantile_head(fused)
        regime_logits = self.regime_head(fused)

        return {
            "expected_return": expected_return,
            "hit_prob": hit_prob,
            "volatility": volatility,
            "quantiles": {
                "q10": quantiles[:, 0],
                "q50": quantiles[:, 1],
                "q90": quantiles[:, 2],
            },
            "regime": torch.argmax(regime_logits, dim=-1),
        }

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "ticker": self.ticker,
                "stock_characteristics": self.stock_characteristics,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str, map_location: str = "cpu") -> None:
        state = torch.load(path, map_location=map_location)
        state_dict = state.get("state_dict", state)
        self.load_state_dict(state_dict, strict=False)


def predict_regime_from_prices(prices_df, stock_chars: Optional[Dict] = None) -> int:
    """Helper used by orchestrator/tests."""
    if prices_df is None or len(prices_df) < 10:
        return 0
    return detect_regime_features(prices_df, stock_chars or {})

