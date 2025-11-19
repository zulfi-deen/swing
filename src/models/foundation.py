"""Foundation model implementation for the swing trading system.

The real project uses a Temporal Fusion Transformer (TFT) combined with a graph
encoder.  For the purposes of the open-source skeleton (and to keep the unit
tests self-contained), we provide a lightweight PyTorch module that mimics the
public API of the production model:

- exposes ``embedding_dim`` and ``tft_hidden_size`` attributes used throughout
  the code base and tests
- provides ``forward`` and ``encode`` helpers that return deterministic
  embeddings
- implements ``freeze`` / ``unfreeze`` utilities
- provides ``initialize_tft`` to keep parity with the training script
- offers ``load_foundation_model`` helper used by the pipeline and registry
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.utils.config import load_config


class StockTwinFoundation(nn.Module):
    """Simplified TFT + GNN hybrid backbone."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {}

        tft_config = self.config.get("tft", {})
        backbone_config = self.config.get("backbone", {})

        self.tft_hidden_size = int(tft_config.get("hidden_size", 256))
        hidden_dims = backbone_config.get("hidden_dims", [256, 128])
        dropout_cfg = backbone_config.get("dropout", [0.1] * len(hidden_dims))
        if isinstance(dropout_cfg, (int, float)):
            dropout_cfg = [dropout_cfg] * len(hidden_dims)

        self.embedding_dim = int(hidden_dims[-1])
        layers = []
        in_dim = self.tft_hidden_size
        for idx, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            dropout_rate = dropout_cfg[idx] if idx < len(dropout_cfg) else 0.0
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self._tft_initialized = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode features into a shared embedding."""
        if inputs.dim() == 3:
            # (batch, seq, feat) -> average over time
            inputs = inputs.mean(dim=1)
        return self.backbone(inputs)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Alias for ``forward``."""
        return self.forward(features)

    def initialize_tft(self, dataset) -> None:  # pragma: no cover - placeholder
        """Mark that TFT weights were initialised from a dataset."""
        _ = dataset  # Dataset is unused in the lightweight implementation
        self._tft_initialized = True

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True


def load_foundation_model(
    checkpoint_path: Optional[str],
    config: Optional[Dict] = None,
    map_location: Optional[str] = None,
) -> StockTwinFoundation:
    """Utility that mirrors the behaviour of the production loader."""

    resolved_config = config or load_config().get("models", {}).get("foundation", {})
    model = StockTwinFoundation(resolved_config)

    if checkpoint_path:
        path = Path(checkpoint_path)
        if path.exists():
            state = torch.load(path, map_location=map_location or "cpu")
            state_dict = state.get("state_dict", state)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:  # pragma: no cover - logging path
                print(
                    f"[foundation] Non-strict load (missing={missing}, unexpected={unexpected})"
                )

    return model

