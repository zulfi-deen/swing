"""Options prediction accuracy metrics and evaluation"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

from src.data.storage import get_timescaledb_engine

logger = logging.getLogger(__name__)


class OptionsMetricsTracker:
    """
    Track options prediction accuracy.
    """
    
    def __init__(self):
        self.db = get_timescaledb_engine()
    
    def evaluate_options_predictions(self, date_range: Tuple[str, str]) -> Dict:
        """
        Evaluate how well options signals predicted outcomes.
        
        Args:
            date_range: (start_date, end_date) tuple
        
        Returns:
            Dict with metrics
        """
        
        start_date, end_date = date_range
        
        # Query predictions + actuals
        query = f"""
        SELECT
            o.ticker,
            o.time as prediction_date,
            
            -- Options signals
            o.pcr_zscore,
            o.pcr_extreme_bullish,
            o.pcr_extreme_bearish,
            o.max_pain_distance_pct,
            o.gamma_signal,
            o.iv_percentile,
            o.trend_signal,
            
            -- Actual outcomes (5 days later)
            p.close as entry_price,
            f.close as exit_price,
            (f.close - p.close) / p.close as actual_return
            
        FROM options_features o
        JOIN prices p ON o.ticker = p.ticker AND o.time = p.time
        JOIN prices f ON o.ticker = f.ticker 
            AND f.time = o.time + INTERVAL '5 days'
        
        WHERE o.time >= '{start_date}'
          AND o.time <= '{end_date}'
        """
        
        try:
            df = pd.read_sql(query, self.db)
        except Exception as e:
            logger.error(f"Error querying options metrics: {e}")
            return {}
        
        if df.empty:
            logger.warning("No data for options evaluation")
            return {}
        
        metrics = {}
        
        # ===== Metric 1: PCR Extreme Accuracy =====
        
        pcr_extreme_bullish = df[df['pcr_extreme_bullish'] == True]
        pcr_extreme_bearish = df[df['pcr_extreme_bearish'] == True]
        
        # Contrarian signal: extreme bullish should lead to negative returns
        if len(pcr_extreme_bullish) > 0:
            pcr_bull_correct = (pcr_extreme_bullish['actual_return'] < 0).mean()
            metrics['pcr_extreme_bullish_accuracy'] = float(pcr_bull_correct)
        
        if len(pcr_extreme_bearish) > 0:
            pcr_bear_correct = (pcr_extreme_bearish['actual_return'] > 0).mean()
            metrics['pcr_extreme_bearish_accuracy'] = float(pcr_bear_correct)
        
        # ===== Metric 2: Gamma Zone Effectiveness =====
        
        # Did stocks near gamma zones move toward max pain?
        near_gamma = df[df['max_pain_distance_pct'].abs() < 0.05]
        
        if len(near_gamma) > 0:
            # Calculate if price moved toward max pain
            moved_toward_pain = (
                np.sign(near_gamma['actual_return']) == 
                np.sign(near_gamma['max_pain_distance_pct'])
            ).mean()
            
            metrics['gamma_zone_accuracy'] = float(moved_toward_pain)
        
        # ===== Metric 3: OI + Price Trend Confirmation =====
        
        strong_trend_signal = df[df['trend_signal'].abs() > 0.5]
        
        if len(strong_trend_signal) > 0:
            trend_correct = (
                np.sign(strong_trend_signal['trend_signal']) == 
                np.sign(strong_trend_signal['actual_return'])
            ).mean()
            
            metrics['oi_trend_confirmation_accuracy'] = float(trend_correct)
        
        # ===== Summary =====
        
        logger.info("\n===== Options Prediction Accuracy =====")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.2%}")
        
        return metrics
    
    def get_latest_options_features(self) -> Dict[str, Dict]:
        """
        Get latest options features for all tickers.
        
        Returns:
            Dict mapping ticker -> options features
        """
        
        query = """
        SELECT * FROM options_features
        WHERE time = (SELECT MAX(time) FROM options_features)
        """
        
        try:
            df = pd.read_sql(query, self.db)
            
            if df.empty:
                return {}
            
            features_dict = {}
            for _, row in df.iterrows():
                ticker = row['ticker']
                features_dict[ticker] = row.to_dict()
            
            return features_dict
        except Exception as e:
            logger.error(f"Error fetching latest options features: {e}")
            return {}


def options_monitoring_dashboard() -> list:
    """
    Real-time options market monitoring.
    
    Alerts:
    - Extreme PCR readings (contrarian opportunities)
    - High gamma concentration (price magnetism)
    - IV spikes (volatility regime change)
    - Unusual OI changes (smart money flows)
    
    Returns:
        List of alert dicts
    """
    
    tracker = OptionsMetricsTracker()
    current_options = tracker.get_latest_options_features()
    
    alerts = []
    
    for ticker, options in current_options.items():
        # Alert 1: Extreme PCR
        if options.get('pcr_extreme_bullish'):
            alerts.append({
                'ticker': ticker,
                'type': 'PCR_EXTREME_BULLISH',
                'message': f"{ticker} PCR = {options.get('pcr_oi', 0):.2f} (extreme bullish, contrarian bearish signal)",
                'severity': 'HIGH'
            })
        
        if options.get('pcr_extreme_bearish'):
            alerts.append({
                'ticker': ticker,
                'type': 'PCR_EXTREME_BEARISH',
                'message': f"{ticker} PCR = {options.get('pcr_oi', 0):.2f} (extreme bearish, contrarian bullish signal)",
                'severity': 'HIGH'
            })
        
        # Alert 2: High gamma concentration
        if options.get('gamma_concentration', 0) > 0.7:
            alerts.append({
                'ticker': ticker,
                'type': 'GAMMA_CONCENTRATION',
                'message': f"{ticker} high gamma at ${options.get('max_pain_strike', 0):.2f} (price magnetism expected)",
                'severity': 'MEDIUM'
            })
        
        # Alert 3: IV spike
        if options.get('iv_percentile', 0.5) > 0.9:
            alerts.append({
                'ticker': ticker,
                'type': 'IV_SPIKE',
                'message': f"{ticker} IV at {options.get('iv_percentile', 0):.0%} percentile (volatility regime change)",
                'severity': 'HIGH'
            })
        
        # Alert 4: Unusual OI change
        if abs(options.get('oi_change_pct', 0)) > 0.20:  # 20%+ OI change
            alerts.append({
                'ticker': ticker,
                'type': 'OI_SURGE',
                'message': f"{ticker} OI changed {options.get('oi_change_pct', 0):+.1%} (smart money flow)",
                'severity': 'MEDIUM'
            })
    
    return alerts

