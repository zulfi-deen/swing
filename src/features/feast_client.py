"""Feast Feature Store Client

Client for interacting with Feast feature store.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

try:
    from feast import FeatureStore
except ImportError:
    FeatureStore = None
    logging.warning("Feast not installed. Feature store functionality will be limited.")

logger = logging.getLogger(__name__)


class FeastFeatureClient:
    """Client for Feast feature store."""
    
    def __init__(self, repo_path: str = "feature_repo/"):
        """
        Initialize Feast feature store client.
        
        Args:
            repo_path: Path to Feast feature repository
        """
        if FeatureStore is None:
            raise ImportError("Feast is required. Install with: pip install feast")
        
        try:
            self.store = FeatureStore(repo_path=repo_path)
            logger.info(f"Initialized Feast feature store from {repo_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Feast feature store: {e}")
            raise
    
    def get_latest_features(
        self,
        tickers: List[str],
        feature_refs: List[str]
    ) -> pd.DataFrame:
        """
        Get latest features for tickers from online store.
        
        Args:
            tickers: List of ticker symbols
            feature_refs: List of feature references (e.g., ['technical_features:rsi_14'])
        
        Returns:
            DataFrame with features
        """
        
        entity_df = pd.DataFrame({
            "ticker": tickers,
            "event_timestamp": [datetime.now()] * len(tickers)
        })
        
        try:
            features = self.store.get_online_features(
                features=feature_refs,
                entity_rows=entity_df.to_dict('records')
            ).to_df()
            
            logger.info(f"Retrieved {len(features)} feature rows for {len(tickers)} tickers")
            return features
        except Exception as e:
            logger.error(f"Error retrieving features from Feast: {e}")
            return pd.DataFrame()
    
    def get_historical_features(
        self,
        tickers: List[str],
        feature_refs: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical features for tickers.
        
        Args:
            tickers: List of ticker symbols
            feature_refs: List of feature references
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with historical features
        """
        
        entity_df = pd.DataFrame({
            "ticker": tickers,
            "event_timestamp": [end_date] * len(tickers)
        })
        
        try:
            features = self.store.get_historical_features(
                features=feature_refs,
                entity_df=entity_df,
                full_feature_names=True
            ).to_df()
            
            logger.info(f"Retrieved historical features for {len(tickers)} tickers from {start_date} to {end_date}")
            return features
        except Exception as e:
            logger.error(f"Error retrieving historical features: {e}")
            return pd.DataFrame()
    
    def materialize_features(
        self,
        start_date: datetime,
        end_date: datetime
    ):
        """
        Materialize features to online store.
        
        Args:
            start_date: Start date for materialization
            end_date: End date for materialization
        """
        
        try:
            self.store.materialize(start_date, end_date)
            logger.info(f"Materialized features from {start_date} to {end_date}")
        except Exception as e:
            logger.error(f"Error materializing features: {e}")
            raise
    
    def list_features(self) -> List[str]:
        """List all available feature views."""
        try:
            feature_views = self.store.list_feature_views()
            feature_refs = []
            for fv in feature_views:
                for feature in fv.features:
                    feature_refs.append(f"{fv.name}:{feature.name}")
            return feature_refs
        except Exception as e:
            logger.error(f"Error listing features: {e}")
            return []


