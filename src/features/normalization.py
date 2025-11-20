"""Feature Normalization and Scaling

Normalizes features for model input to ensure consistent scales.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, List, Optional
import joblib
import logging
import os

logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """Normalizes features for model input."""
    
    def __init__(self):
        self.scalers: Dict[str, StandardScaler | RobustScaler] = {}
        self.feature_groups = {
            'price_based': ['close', 'open', 'high', 'low', 'vwap'],
            'indicators': ['rsi_14', 'macd', 'macd_signal', 'macd_hist', 'atr_14', 'adx', 'mfi', 'stoch_k', 'stoch_d'],
            'volume': ['volume', 'dollar_volume', 'avg_dollar_volume_20'],
            'returns': ['return_5d', 'return_20d', 'return_1d'],
            'sentiment': ['sentiment_score'],
            'cross_sectional': ['return_rank_5d', 'return_rank_20d', 'sector_relative_strength', 'correlation_to_spy']
        }
        self.fitted = False
    
    def fit(self, df: pd.DataFrame):
        """
        Fit scalers on training data.
        
        Args:
            df: Training DataFrame with features
        """
        logger.info("Fitting feature normalizers...")
        
        for group, features in self.feature_groups.items():
            available_features = [f for f in features if f in df.columns]
            if not available_features:
                continue
            
            # Use RobustScaler for returns (robust to outliers)
            # StandardScaler for others
            if group == 'returns':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            try:
                # Fit on available features
                scaler.fit(df[available_features].fillna(0))
                self.scalers[group] = scaler
                logger.debug(f"Fitted {group} scaler on {len(available_features)} features")
            except Exception as e:
                logger.warning(f"Failed to fit scaler for {group}: {e}")
        
        self.fitted = True
        logger.info(f"Fitted {len(self.scalers)} feature group scalers")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scalers.
        
        Args:
            df: DataFrame with features to transform
        
        Returns:
            DataFrame with normalized features
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        df_scaled = df.copy()
        
        for group, scaler in self.scalers.items():
            features = [f for f in self.feature_groups[group] if f in df.columns]
            if not features:
                continue
            
            try:
                # Fill NaN with 0 before scaling
                df_scaled[features] = scaler.transform(df[features].fillna(0))
            except Exception as e:
                logger.warning(f"Failed to transform {group} features: {e}")
                # Keep original values if transformation fails
                continue
        
        return df_scaled
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform(self, df: pd.DataFrame, feature_group: Optional[str] = None) -> pd.DataFrame:
        """
        Inverse transform features (for predictions).
        
        Args:
            df: DataFrame with normalized features
            feature_group: Optional group to inverse transform (if None, transforms all)
        
        Returns:
            DataFrame with denormalized features
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        df_denorm = df.copy()
        
        groups_to_transform = [feature_group] if feature_group else self.scalers.keys()
        
        for group in groups_to_transform:
            if group not in self.scalers:
                continue
            
            features = [f for f in self.feature_groups[group] if f in df.columns]
            if not features:
                continue
            
            try:
                scaler = self.scalers[group]
                if hasattr(scaler, 'inverse_transform'):
                    df_denorm[features] = scaler.inverse_transform(df[features])
            except Exception as e:
                logger.warning(f"Failed to inverse transform {group} features: {e}")
        
        return df_denorm
    
    def save(self, path: str):
        """Save fitted scalers to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        joblib.dump({
            'scalers': self.scalers,
            'feature_groups': self.feature_groups,
            'fitted': self.fitted
        }, path)
        logger.info(f"Saved normalizer to {path}")
    
    def load(self, path: str):
        """Load fitted scalers from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Normalizer file not found: {path}")
        
        data = joblib.load(path)
        self.scalers = data.get('scalers', {})
        self.feature_groups = data.get('feature_groups', self.feature_groups)
        self.fitted = data.get('fitted', False)
        logger.info(f"Loaded normalizer from {path}")
    
    def get_feature_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about features (for validation)."""
        stats = {}
        for group, features in self.feature_groups.items():
            available_features = [f for f in features if f in df.columns]
            if available_features:
                stats[group] = {
                    'count': len(available_features),
                    'features': available_features,
                    'has_nan': df[available_features].isnull().any().any(),
                    'has_inf': np.isinf(df[available_features].select_dtypes(include=[np.number])).any().any()
                }
        return stats


