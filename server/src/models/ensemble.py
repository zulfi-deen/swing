"""Ensemble utilities combining digital twins, classical models and LightGBM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd


def ensemble_predictions(
    twin_predictions: Dict,
    lgbm_ranks: Optional[float] = None,
    arima_garch: Optional[Dict] = None,
    patterns: Optional[Dict] = None,
    text_features: Optional[Dict] = None,
) -> Dict:
    """Blend predictions using light heuristic weights."""

    arima_garch = arima_garch or {}
    patterns = patterns or {}
    text_features = text_features or {}

    expected_return = twin_predictions.get("return", 0.0)
    hit_prob = twin_predictions.get("prob_hit_long", 0.5)
    volatility = twin_predictions.get("volatility", 0.02)

    if arima_garch:
        expected_return = 0.7 * expected_return + 0.3 * arima_garch.get("return", 0.0)
        volatility = 0.4 * volatility + 0.6 * arima_garch.get("volatility", volatility)

    if lgbm_ranks is not None:
        # Convert rank score to [0,1] weight
        lgbm_score = 1 / (1 + np.exp(-lgbm_ranks))
        expected_return = 0.8 * expected_return + 0.2 * lgbm_score

    pattern_boost = patterns.get("confidence", 0.0)
    text_boost = text_features.get("sentiment_score", 0.0)
    expected_return += 0.01 * (pattern_boost + text_boost)

    return {
        "expected_return": float(expected_return),
        "hit_probability": float(hit_prob),
        "volatility": float(max(1e-4, volatility)),
    }


@dataclass
class LightGBMRanker:
    """Light wrapper around LightGBM to keep training scripts concise."""

    objective: str = "lambdarank"
    metric: str = "ndcg"
    learning_rate: float = 0.05
    num_leaves: int = 31
    n_estimators: int = 500
    max_depth: int = -1
    additional_params: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        import lightgbm as lgb

        self._lgb = lgb.LGBMRanker(
            objective=self.objective,
            metric=self.metric,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            **self.additional_params,
        )

    @property
    def model(self):
        return self._lgb

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups: Iterable[int],
        eval_set=None,
        eval_groups=None,
    ) -> None:
        self._lgb.fit(
            X_train,
            y_train,
            group=groups,
            eval_set=eval_set,
            eval_group=eval_groups,
            verbose=False,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._lgb.predict(X)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._lgb, path)

    def load(self, path: str) -> None:
        self._lgb = joblib.load(path)

    def get_feature_importance(self) -> pd.Series:
        importance = self._lgb.booster_.feature_importance()
        names = self._lgb.booster_.feature_name()
        return pd.Series(importance, index=names).sort_values(ascending=False)

