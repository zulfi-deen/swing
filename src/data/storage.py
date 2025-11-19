"""Shared storage helpers for local filesystem and TimescaleDB access."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
try:  # pragma: no cover - optional dependency
    from sqlalchemy import create_engine
    from sqlalchemy.engine import Engine

    SQLALCHEMY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    create_engine = None
    Engine = object
    SQLALCHEMY_AVAILABLE = False

from src.utils.config import load_config


def _resolve_config(config: Optional[Dict]) -> Dict:
    return config or load_config()


def _get_data_dir(config: Optional[Dict]) -> Path:
    cfg = _resolve_config(config)
    directory = (
        cfg.get("storage", {}).get("local", {}).get("data_dir", "data")
    )
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timescaledb_engine(config: Optional[Dict] = None) -> Engine:
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError(
            "sqlalchemy is required for TimescaleDB access but is not installed."
        )
    cfg = _resolve_config(config)
    ts_cfg = cfg.get("storage", {}).get("timescaledb", {})
    host = ts_cfg.get("host", "localhost")
    port = ts_cfg.get("port", 5432)
    database = ts_cfg.get("database", "swing_trading")
    user = ts_cfg.get("user", "postgres")
    password = ts_cfg.get("password", "postgres")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    return create_engine(url, future=True)


def save_to_timescaledb(
    df: pd.DataFrame,
    table: str,
    config: Optional[Dict] = None,
    if_exists: str = "append",
) -> None:
    engine = get_timescaledb_engine(config)
    df.to_sql(table, engine, if_exists=if_exists, index=False)


def _resolve_path(relative_path: str, config: Optional[Dict]) -> Path:
    data_dir = _get_data_dir(config)
    full_path = data_dir / f"{relative_path}"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    return full_path


def save_to_local(data, relative_path: str, config: Optional[Dict] = None) -> None:
    path = _resolve_path(relative_path, config)

    if isinstance(data, pd.DataFrame):
        path = path.with_suffix(".csv")
        data.to_csv(path, index=False)
        return

    # For dict/list types we default to JSON
    serialisable = data
    path = path.with_suffix(".json")
    with open(path, "w") as fp:
        json.dump(serialisable, fp, default=_json_serializer)


def _json_serializer(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def load_from_local(relative_path: str, config: Optional[Dict] = None):
    base = _resolve_path(relative_path, config)
    csv_path = base.with_suffix(".csv")
    json_path = base.with_suffix(".json")

    if csv_path.exists():
        return pd.read_csv(csv_path)
    if json_path.exists():
        with open(json_path) as fp:
            return json.load(fp)
    raise FileNotFoundError(f"No stored data found for path {relative_path}")


def _characteristics_path(config: Optional[Dict] = None) -> Path:
    return _resolve_path("metadata/stock_characteristics", config).with_suffix(".json")


def get_stock_characteristics(ticker: Optional[str] = None, config: Optional[Dict] = None):
    path = _characteristics_path(config)
    if not path.exists():
        return {} if ticker else {}
    with open(path) as fp:
        data = json.load(fp)
    if ticker:
        return data.get(ticker, {})
    return data


def update_stock_characteristics(
    ticker: str,
    characteristics: Dict,
    config: Optional[Dict] = None,
) -> None:
    path = _characteristics_path(config)
    existing = get_stock_characteristics(None, config)
    existing[ticker] = characteristics
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(existing, fp, indent=2)


def compute_stock_characteristics(prices_df: pd.DataFrame) -> Dict:
    if prices_df is None or prices_df.empty:
        return {}
    returns = prices_df["close"].pct_change().dropna()
    beta = float(returns.cov(returns) / returns.var()) if not returns.empty else 1.0
    return {
        "beta": beta,
        "mean_reversion_strength": float(1 / (1 + returns.std())),
        "liquidity_regime": 1,
    }

