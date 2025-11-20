"""Options feature engineering - 40 features per stock per day"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class OptionsFeatureExtractor:
    """
    Extract 40 options features from raw options data.
    
    Categories:
    1. Volume & Open Interest (8 features)
    2. Put-Call Ratios (6 features)
    3. Gamma Exposure (7 features)
    4. Implied Volatility (7 features)
    5. Net Greeks (5 features)
    6. Term Structure (4 features)
    7. Composite Signals (3 features)
    """
    
    def extract_all(self, ticker: str, date: str, options_data: Dict) -> Dict:
        """
        Master extraction function.
        
        Args:
            ticker: Stock ticker
            date: Date (YYYY-MM-DD)
            options_data: Dict with 'strikes' list and 'current_price', 'historical' context
        
        Returns:
            Dict with 40 options features
        """
        
        features = {}
        
        # Layer 1: Volume & OI
        features.update(self.extract_volume_oi(options_data))
        
        # Layer 2: PCR
        features.update(self.extract_pcr(options_data))
        
        # Layer 3: Gamma
        features.update(self.extract_gamma(options_data))
        
        # Layer 4: IV
        features.update(self.extract_iv(options_data))
        
        # Layer 5: Greeks
        features.update(self.extract_greeks(options_data))
        
        # Layer 6: Term structure
        features.update(self.extract_term_structure(options_data))
        
        # Layer 7: Composite signals
        features.update(self.extract_composite_signals(features, options_data))
        
        return features
    
    def extract_volume_oi(self, options_data: Dict) -> Dict:
        """
        Extract volume and open interest features (8 features).
        """
        
        strikes = options_data.get('strikes', [])
        
        if not strikes:
            return self._get_default_volume_oi()
        
        # Aggregate across all strikes
        call_oi = sum([s.get('call_oi', 0) for s in strikes])
        put_oi = sum([s.get('put_oi', 0) for s in strikes])
        total_oi = call_oi + put_oi
        
        call_volume = sum([s.get('call_volume', 0) for s in strikes])
        put_volume = sum([s.get('put_volume', 0) for s in strikes])
        total_volume = call_volume + put_volume
        
        # Historical comparison
        historical = options_data.get('historical', {})
        yesterday_oi = historical.get('total_oi_yesterday', total_oi)
        avg_volume_20d = historical.get('avg_volume_20d', total_volume)
        std_volume_20d = historical.get('std_volume_20d', total_volume * 0.5)
        
        # Change metrics
        oi_change_pct = (total_oi - yesterday_oi) / (yesterday_oi + 1e-8)
        volume_zscore = (total_volume - avg_volume_20d) / (std_volume_20d + 1e-8)
        
        # Call/Put split
        call_oi_pct = call_oi / (total_oi + 1e-8)
        put_oi_pct = put_oi / (total_oi + 1e-8)
        
        return {
            'call_oi': int(call_oi),
            'put_oi': int(put_oi),
            'total_oi': int(total_oi),
            'call_volume': int(call_volume),
            'put_volume': int(put_volume),
            'total_volume': int(total_volume),
            'oi_change_pct': float(oi_change_pct),
            'volume_zscore': float(volume_zscore),
        }
    
    def extract_pcr(self, options_data: Dict) -> Dict:
        """
        Extract put-call ratio features (6 features).
        """
        
        strikes = options_data.get('strikes', [])
        
        if not strikes:
            return self._get_default_pcr()
        
        call_oi = sum([s.get('call_oi', 0) for s in strikes])
        put_oi = sum([s.get('put_oi', 0) for s in strikes])
        
        call_volume = sum([s.get('call_volume', 0) for s in strikes])
        put_volume = sum([s.get('put_volume', 0) for s in strikes])
        
        # Basic ratios
        pcr_oi = put_oi / (call_oi + 1e-8)
        pcr_volume = put_volume / (call_volume + 1e-8)
        
        # Historical context
        historical = options_data.get('historical', {})
        historical_pcr = historical.get('pcr_oi_60d', [pcr_oi] * 60)
        
        if len(historical_pcr) > 0:
            pcr_mean = np.mean(historical_pcr)
            pcr_std = np.std(historical_pcr)
            pcr_zscore = (pcr_oi - pcr_mean) / (pcr_std + 1e-8)
        else:
            pcr_zscore = 0.0
        
        # Extreme flags
        pcr_extreme_bullish = 1 if pcr_oi < 0.7 else 0  # Too many calls
        pcr_extreme_bearish = 1 if pcr_oi > 1.0 else 0  # Too many puts
        
        # Trend
        pcr_yesterday = historical.get('pcr_oi_yesterday', pcr_oi)
        pcr_change = pcr_oi - pcr_yesterday
        
        return {
            'pcr_oi': float(pcr_oi),
            'pcr_volume': float(pcr_volume),
            'pcr_zscore': float(pcr_zscore),
            'pcr_extreme_bullish': bool(pcr_extreme_bullish),
            'pcr_extreme_bearish': bool(pcr_extreme_bearish),
            'pcr_change': float(pcr_change),
        }
    
    def extract_gamma(self, options_data: Dict) -> Dict:
        """
        Extract gamma exposure features (7 features).
        """
        
        strikes = options_data.get('strikes', [])
        current_price = options_data.get('current_price', 0.0)
        
        if not strikes or current_price == 0:
            return self._get_default_gamma()
        
        # Calculate gamma exposure by strike
        gamma_by_strike = []
        
        for strike_data in strikes:
            strike = strike_data.get('strike_price', 0)
            if strike == 0:
                continue
            
            # Call gamma exposure (positive)
            call_gamma = strike_data.get('call_gamma', 0.0)
            call_oi = strike_data.get('call_oi', 0)
            call_gex = call_gamma * call_oi * 100 * strike
            
            # Put gamma exposure (negative for market makers)
            put_gamma = strike_data.get('put_gamma', 0.0)
            put_oi = strike_data.get('put_oi', 0)
            put_gex = put_gamma * put_oi * 100 * strike * -1
            
            # Net gamma exposure
            net_gex = call_gex + put_gex
            
            gamma_by_strike.append({
                'strike': strike,
                'gamma': net_gex,
                'distance_pct': (strike - current_price) / current_price if current_price > 0 else 0.0
            })
        
        if not gamma_by_strike:
            return self._get_default_gamma()
        
        gamma_df = pd.DataFrame(gamma_by_strike)
        
        # Find max pain (strike with highest absolute gamma)
        max_gamma_idx = gamma_df['gamma'].abs().idxmax()
        max_pain_strike = gamma_df.loc[max_gamma_idx, 'strike']
        max_pain_distance = gamma_df.loc[max_gamma_idx, 'distance_pct']
        
        # Total gamma (sign indicates net positioning)
        total_gamma = gamma_df['gamma'].sum()
        gamma_sign = 1 if total_gamma > 0 else -1
        
        # Gamma concentration (how peaked is distribution?)
        gamma_abs_sum = gamma_df['gamma'].abs().sum()
        gamma_concentration = gamma_df['gamma'].abs().max() / (gamma_abs_sum + 1e-8)
        
        # Gamma flip level (where gamma changes sign)
        positive_gamma = gamma_df[gamma_df['gamma'] > 0]
        negative_gamma = gamma_df[gamma_df['gamma'] < 0]
        
        if len(positive_gamma) > 0 and len(negative_gamma) > 0:
            # Find strike closest to zero gamma
            gamma_flip_idx = gamma_df['gamma'].abs().idxmin()
            gamma_flip_strike = gamma_df.loc[gamma_flip_idx, 'strike']
            gamma_flip_distance = (gamma_flip_strike - current_price) / current_price if current_price > 0 else 0.0
        else:
            gamma_flip_strike = np.nan
            gamma_flip_distance = np.nan
        
        return {
            'max_pain_strike': float(max_pain_strike),
            'max_pain_distance_pct': float(max_pain_distance),
            'total_gamma': float(total_gamma / 1e9),  # Normalize to billions
            'gamma_sign': int(gamma_sign),
            'gamma_concentration': float(gamma_concentration),
            'gamma_flip_strike': float(gamma_flip_strike) if not np.isnan(gamma_flip_strike) else None,
            'gamma_flip_distance_pct': float(gamma_flip_distance) if not np.isnan(gamma_flip_distance) else None,
        }
    
    def extract_iv(self, options_data: Dict) -> Dict:
        """
        Extract implied volatility features (7 features).
        """
        
        strikes = options_data.get('strikes', [])
        current_price = options_data.get('current_price', 0.0)
        
        if not strikes or current_price == 0:
            return self._get_default_iv()
        
        # ATM IV (at-the-money)
        atm_strikes = self._find_atm_strikes(strikes, current_price)
        
        if not atm_strikes:
            return self._get_default_iv()
        
        atm_call_ivs = [s.get('call_iv', 0.0) for s in atm_strikes if s.get('call_iv', 0.0) > 0]
        atm_put_ivs = [s.get('put_iv', 0.0) for s in atm_strikes if s.get('put_iv', 0.0) > 0]
        
        atm_call_iv = np.mean(atm_call_ivs) if atm_call_ivs else 0.25
        atm_put_iv = np.mean(atm_put_ivs) if atm_put_ivs else 0.25
        
        # IV skew (put IV - call IV)
        iv_skew = atm_put_iv - atm_call_iv
        
        # Put/Call IV ratio
        put_call_iv_ratio = atm_put_iv / (atm_call_iv + 1e-8)
        
        # IV percentile (where is current IV vs. history?)
        historical = options_data.get('historical', {})
        historical_iv = historical.get('atm_iv_252d', [atm_call_iv] * 252)
        
        if len(historical_iv) > 0:
            iv_percentile = stats.percentileofscore(historical_iv, atm_call_iv) / 100.0
            
            # IV rank (normalized 0-1)
            iv_min = np.min(historical_iv)
            iv_max = np.max(historical_iv)
            iv_rank = (atm_call_iv - iv_min) / (iv_max - iv_min + 1e-8) if iv_max > iv_min else 0.5
        else:
            iv_percentile = 0.5
            iv_rank = 0.5
        
        # IV change (trend)
        iv_yesterday = historical.get('atm_iv_yesterday', atm_call_iv)
        iv_change_pct = (atm_call_iv - iv_yesterday) / (iv_yesterday + 1e-8)
        
        return {
            'atm_call_iv': float(atm_call_iv),
            'atm_put_iv': float(atm_put_iv),
            'iv_skew': float(iv_skew),
            'put_call_iv_ratio': float(put_call_iv_ratio),
            'iv_percentile': float(iv_percentile),
            'iv_rank': float(iv_rank),
            'iv_change_pct': float(iv_change_pct),
        }
    
    def extract_greeks(self, options_data: Dict) -> Dict:
        """
        Extract net Greek exposure (5 features).
        """
        
        strikes = options_data.get('strikes', [])
        
        if not strikes:
            return self._get_default_greeks()
        
        total_oi = sum([s.get('call_oi', 0) + s.get('put_oi', 0) for s in strikes])
        
        # Net Delta (directional exposure)
        net_delta = sum([
            s.get('call_delta', 0.0) * s.get('call_oi', 0) - 
            s.get('put_delta', 0.0) * s.get('put_oi', 0)
            for s in strikes
        ])
        
        # Net Gamma (convexity)
        net_gamma = sum([
            s.get('call_gamma', 0.0) * s.get('call_oi', 0) - 
            s.get('put_gamma', 0.0) * s.get('put_oi', 0)
            for s in strikes
        ])
        
        # Net Vega (vol exposure)
        net_vega = sum([
            s.get('call_vega', 0.0) * s.get('call_oi', 0) + 
            s.get('put_vega', 0.0) * s.get('put_oi', 0)
            for s in strikes
        ])
        
        # Net Theta (time decay)
        net_theta = sum([
            s.get('call_theta', 0.0) * s.get('call_oi', 0) + 
            s.get('put_theta', 0.0) * s.get('put_oi', 0)
            for s in strikes
        ])
        
        # Normalize by total OI
        return {
            'net_delta': float(net_delta / (total_oi + 1e-8)),
            'net_gamma': float(net_gamma / (total_oi + 1e-8)),
            'net_vega': float(net_vega / (total_oi + 1e-8)),
            'net_theta': float(net_theta / (total_oi + 1e-8)),
            'net_delta_abs': float(abs(net_delta) / (total_oi + 1e-8)),  # Strength of bias
        }
    
    def extract_term_structure(self, options_data: Dict) -> Dict:
        """
        Extract term structure features (4 features).
        """
        
        strikes = options_data.get('strikes', [])
        nearest_exp = options_data.get('nearest_expiration')
        second_exp = options_data.get('second_expiration')
        
        if not strikes or not nearest_exp:
            return self._get_default_term_structure()
        
        # Separate by expiration
        front_month_oi = sum([
            s.get('call_oi', 0) + s.get('put_oi', 0)
            for s in strikes
            if s.get('expiration') == nearest_exp
        ])
        
        next_month_oi = 0
        if second_exp:
            next_month_oi = sum([
                s.get('call_oi', 0) + s.get('put_oi', 0)
                for s in strikes
                if s.get('expiration') == second_exp
            ])
        
        # Roll ratio (front / next)
        roll_ratio = front_month_oi / (next_month_oi + 1e-8)
        
        # IV term structure
        front_month_ivs = [
            (s.get('call_iv', 0.0) + s.get('put_iv', 0.0)) / 2.0
            for s in strikes
            if s.get('expiration') == nearest_exp and (s.get('call_iv', 0.0) > 0 or s.get('put_iv', 0.0) > 0)
        ]
        
        next_month_ivs = []
        if second_exp:
            next_month_ivs = [
                (s.get('call_iv', 0.0) + s.get('put_iv', 0.0)) / 2.0
                for s in strikes
                if s.get('expiration') == second_exp and (s.get('call_iv', 0.0) > 0 or s.get('put_iv', 0.0) > 0)
            ]
        
        front_month_iv = np.mean(front_month_ivs) if front_month_ivs else 0.25
        next_month_iv = np.mean(next_month_ivs) if next_month_ivs else front_month_iv
        
        # Term curve slope (backwardation vs. contango)
        term_curve_slope = next_month_iv - front_month_iv
        
        return {
            'front_month_oi': int(front_month_oi),
            'next_month_oi': int(next_month_oi),
            'roll_ratio': float(roll_ratio),
            'term_curve_slope': float(term_curve_slope),
        }
    
    def extract_composite_signals(self, features: Dict, options_data: Dict) -> Dict:
        """
        Generate high-level composite signals (3 features).
        """
        
        recent_return = options_data.get('price_change_5d', 0.0)
        oi_change = features.get('oi_change_pct', 0.0)
        pcr_zscore = features.get('pcr_zscore', 0.0)
        gamma_distance = features.get('max_pain_distance_pct', 0.0)
        
        # Signal 1: OI + Price Trend Confirmation
        if oi_change > 0.05 and recent_return > 0.02:
            trend_signal = 1.0  # Bullish confirmation
        elif oi_change > 0.05 and recent_return < -0.02:
            trend_signal = -1.0  # Bearish confirmation
        elif oi_change < -0.05 and recent_return > 0.02:
            trend_signal = -0.5  # Warning: momentum fading
        else:
            trend_signal = 0.0  # Neutral
        
        # Signal 2: PCR Extreme (Mean Reversion)
        if pcr_zscore < -2:  # Extreme bullish (too many calls)
            sentiment_signal = -1.0  # Contrarian bearish
        elif pcr_zscore > 2:  # Extreme bearish (too many puts)
            sentiment_signal = 1.0  # Contrarian bullish
        else:
            sentiment_signal = 0.0  # Neutral
        
        # Signal 3: Gamma Zone Proximity
        if abs(gamma_distance) < 0.03:  # Within 3% of max pain
            gamma_signal = 1  # High gamma support/resistance
        else:
            gamma_signal = 0  # No gamma effect
        
        return {
            'trend_signal': float(trend_signal),
            'sentiment_signal': float(sentiment_signal),
            'gamma_signal': int(gamma_signal),
        }
    
    def _find_atm_strikes(self, strikes: List[Dict], current_price: float) -> List[Dict]:
        """Find strikes closest to current price (ATM)."""
        
        if not strikes or current_price == 0:
            return []
        
        # Find 3 strikes closest to current price
        strikes_with_distance = [
            (s, abs(s.get('strike_price', 0) - current_price))
            for s in strikes
        ]
        
        strikes_with_distance.sort(key=lambda x: x[1])
        return [s[0] for s in strikes_with_distance[:3]]
    
    def _get_default_volume_oi(self) -> Dict:
        return {
            'call_oi': 0, 'put_oi': 0, 'total_oi': 0,
            'call_volume': 0, 'put_volume': 0, 'total_volume': 0,
            'oi_change_pct': 0.0, 'volume_zscore': 0.0,
        }
    
    def _get_default_pcr(self) -> Dict:
        return {
            'pcr_oi': 1.0, 'pcr_volume': 1.0, 'pcr_zscore': 0.0,
            'pcr_extreme_bullish': False, 'pcr_extreme_bearish': False,
            'pcr_change': 0.0,
        }
    
    def _get_default_gamma(self) -> Dict:
        return {
            'max_pain_strike': 0.0, 'max_pain_distance_pct': 0.0,
            'total_gamma': 0.0, 'gamma_sign': 0,
            'gamma_concentration': 0.0,
            'gamma_flip_strike': None, 'gamma_flip_distance_pct': None,
        }
    
    def _get_default_iv(self) -> Dict:
        return {
            'atm_call_iv': 0.25, 'atm_put_iv': 0.25,
            'iv_skew': 0.0, 'put_call_iv_ratio': 1.0,
            'iv_percentile': 0.5, 'iv_rank': 0.5, 'iv_change_pct': 0.0,
        }
    
    def _get_default_greeks(self) -> Dict:
        return {
            'net_delta': 0.0, 'net_gamma': 0.0, 'net_vega': 0.0,
            'net_theta': 0.0, 'net_delta_abs': 0.0,
        }
    
    def _get_default_term_structure(self) -> Dict:
        return {
            'front_month_oi': 0, 'next_month_oi': 0,
            'roll_ratio': 1.0, 'term_curve_slope': 0.0,
        }


def get_default_options_features() -> Dict:
    """Return neutral options features when data unavailable."""
    extractor = OptionsFeatureExtractor()
    return {
        **extractor._get_default_volume_oi(),
        **extractor._get_default_pcr(),
        **extractor._get_default_gamma(),
        **extractor._get_default_iv(),
        **extractor._get_default_greeks(),
        **extractor._get_default_term_structure(),
        'trend_signal': 0.0, 'sentiment_signal': 0.0, 'gamma_signal': 0,
    }


def compute_options_features_batch(
    options_data_dict: Dict[str, Dict],
    prices_df: Optional[pd.DataFrame] = None,
    date: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Compute options features for multiple tickers in batch.
    
    Args:
        options_data_dict: Dict mapping ticker -> options data dict
        prices_df: Optional DataFrame with price history for context
        date: Optional date string
    
    Returns:
        Dict mapping ticker -> 40 features dict
    """
    
    extractor = OptionsFeatureExtractor()
    all_features = {}
    
    for ticker, options_data in options_data_dict.items():
        try:
            # Get current price from options_data or prices_df
            current_price = options_data.get('current_price')
            if not current_price and prices_df is not None:
                ticker_prices = prices_df[prices_df['ticker'] == ticker]
                if not ticker_prices.empty:
                    current_price = ticker_prices['close'].iloc[-1]
                    options_data['current_price'] = current_price
            
            # Get price change for composite signals
            if prices_df is not None:
                ticker_prices = prices_df[prices_df['ticker'] == ticker]
                if len(ticker_prices) >= 5:
                    price_5d_ago = ticker_prices['close'].iloc[-5]
                    current_price = ticker_prices['close'].iloc[-1]
                    options_data['price_change_5d'] = (current_price - price_5d_ago) / price_5d_ago
            
            features = extractor.extract_all(ticker, date or '', options_data)
            all_features[ticker] = features
            
        except Exception as e:
            logger.error(f"[{ticker}] Options features failed: {str(e)}")
            all_features[ticker] = get_default_options_features()
    
    return all_features

