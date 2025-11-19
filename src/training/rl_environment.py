"""Trading Environment for RL Portfolio Agent"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from src.data.rl_state_builder import RLStateBuilder

logger = logging.getLogger(__name__)


class TradingEnvironment:
    """
    Gym-style trading environment for RL training.
    
    Simulates portfolio trading using historical data and twin predictions.
    """
    
    def __init__(
        self,
        historical_predictions: pd.DataFrame,
        historical_prices: pd.DataFrame,
        historical_features: pd.DataFrame,
        macro_data: pd.DataFrame,
        initial_capital: float = 100000.0,
        max_positions: int = 15,
        max_position_size: float = 0.10,
        max_sector_exposure: float = 0.25,
        transaction_cost: float = 0.001,
        config: Optional[Dict] = None,
        historical_options_features: Optional[pd.DataFrame] = None
    ):
        """
        Initialize trading environment.
        
        Args:
            historical_predictions: DataFrame with columns [date, ticker, expected_return, hit_prob, volatility, ...]
            historical_prices: DataFrame with columns [date, ticker, close, ...]
            historical_features: DataFrame with features per ticker per date
            macro_data: DataFrame with macro indicators per date
            initial_capital: Starting capital
            max_positions: Maximum concurrent positions
            max_position_size: Maximum position size (fraction of portfolio)
            max_sector_exposure: Maximum sector exposure
            transaction_cost: Cost per trade (fraction)
            config: Optional config dict
            historical_options_features: Optional DataFrame with options features per ticker per date
        """
        
        self.historical_predictions = historical_predictions.copy()
        self.historical_prices = historical_prices.copy()
        self.historical_features = historical_features.copy()
        self.macro_data = macro_data.copy()
        self.historical_options_features = historical_options_features.copy() if historical_options_features is not None else pd.DataFrame()
        
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.transaction_cost = transaction_cost
        self.config = config or {}
        
        # State builder
        self.state_builder = RLStateBuilder(config)
        
        # Get date range
        if 'date' in self.historical_predictions.columns:
            dates_raw = self.historical_predictions['date'].unique()
            # Convert to date objects if needed
            if isinstance(dates_raw[0], str):
                self.dates = sorted([pd.to_datetime(d).date() for d in dates_raw])
            else:
                self.dates = sorted([d.date() if hasattr(d, 'date') else d for d in dates_raw])
        elif 'time' in self.historical_predictions.columns:
            self.dates = sorted(pd.to_datetime(self.historical_predictions['time']).dt.date.unique())
        else:
            raise ValueError("No date column found in historical_predictions")
        
        # Portfolio state
        self.reset()
    
    def reset(self, start_date: Optional[str] = None) -> Dict:
        """
        Reset environment to initial state.
        
        Args:
            start_date: Optional start date (defaults to random)
        
        Returns:
            Initial state dict
        """
        
        # Select random start date (leave room for episode length)
        if start_date is None:
            episode_length = 60  # 60 trading days
            if len(self.dates) > episode_length:
                max_start_idx = len(self.dates) - episode_length
                start_idx = np.random.randint(0, max_start_idx)
                self.current_date_idx = start_idx
            else:
                self.current_date_idx = 0
        else:
            # Convert start_date to date object if string
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date).date()
            # Find matching date
            try:
                self.current_date_idx = self.dates.index(start_date)
            except ValueError:
                # Find closest date
                closest_idx = min(range(len(self.dates)), key=lambda i: abs((self.dates[i] - start_date).days))
                self.current_date_idx = closest_idx
        
        # Initialize portfolio
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}  # ticker -> {'size': float, 'entry_price': float, 'days_held': int}
        self.peak_value = self.initial_capital
        self.closed_trades = []
        
        # Get initial state
        state = self._get_state()
        
        return state
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one trading step.
        
        Args:
            action: Dict with 'stock_selection' and 'position_sizes'
        
        Returns:
            (next_state, reward, done, info)
        """
        
        if self.current_date_idx >= len(self.dates) - 1:
            return self._get_state(), 0.0, True, {'reason': 'episode_end'}
        
        current_date = self.dates[self.current_date_idx]
        next_date = self.dates[self.current_date_idx + 1]
        
        # Execute trades (rebalance portfolio)
        self._execute_trades(action, current_date)
        
        # Update portfolio value based on next day prices
        portfolio_value_before = self.portfolio_value
        self._update_portfolio_value(next_date)
        
        # Calculate reward
        reward = self._compute_reward(portfolio_value_before, action)
        
        # Move to next day
        self.current_date_idx += 1
        
        # Check if done
        done = (self.current_date_idx >= len(self.dates) - 1)
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'num_positions': len([p for p in self.positions.values() if p['size'] > 0]),
            'cash': self.cash,
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> Dict:
        """Get current state."""
        
        if self.current_date_idx >= len(self.dates):
            # Return empty state if out of bounds
            return self._empty_state()
        
        current_date = self.dates[self.current_date_idx]
        
        # Get predictions for current date
        date_preds = self.historical_predictions[
            self.historical_predictions['date'] == current_date
        ] if 'date' in self.historical_predictions.columns else self.historical_predictions[
            pd.to_datetime(self.historical_predictions['time']).dt.date == current_date
        ]
        
        # Convert to dict format
        twin_predictions = {}
        for _, row in date_preds.iterrows():
            ticker = row['ticker']
            twin_predictions[ticker] = {
                'expected_return': float(row.get('expected_return', 0.0)),
                'hit_prob': float(row.get('hit_probability', row.get('hit_prob', 0.5))),
                'volatility': float(row.get('volatility', 0.02)),
                'regime': int(row.get('regime', 0)),
                'idiosyncratic_alpha': float(row.get('idiosyncratic_alpha', 0.0)),
                'quantiles': {
                    'q10': float(row.get('quantile_10', -0.05)),
                    'q90': float(row.get('quantile_90', 0.05)),
                }
            }
        
        # Get features
        date_features = self.historical_features[
            self.historical_features['date'] == current_date
        ] if 'date' in self.historical_features.columns else self.historical_features[
            pd.to_datetime(self.historical_features['time']).dt.date == current_date
        ]
        
        # Get macro context
        date_macro = self.macro_data[
            self.macro_data['date'] == current_date
        ] if 'date' in self.macro_data.columns else self.macro_data[
            pd.to_datetime(self.macro_data['time']).dt.date == current_date
        ]
        
        macro_context = {}
        if not date_macro.empty:
            row = date_macro.iloc[0]
            macro_context = {
                'vix': float(row.get('vix', 20.0)),
                'spy_return_5d': float(row.get('spy_return_5d', 0.0)),
                'treasury_10y': float(row.get('treasury_10y', 4.0)),
                'market_regime': str(row.get('market_regime', 'bull')),
            }
        else:
            macro_context = {
                'vix': 20.0,
                'spy_return_5d': 0.0,
                'treasury_10y': 4.0,
                'market_regime': 'bull',
            }
        
        # Build portfolio state
        portfolio_state = {
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'peak_value': self.peak_value,
            'positions': {
                ticker: {
                    'size': pos['size'],
                    'entry_price': pos['entry_price'],
                    'days_held': pos['days_held']
                }
                for ticker, pos in self.positions.items()
            },
            'sector_exposure': self._compute_sector_exposure(),
        }
        
        # Get prices for graph
        prices_df = self.historical_prices[
            pd.to_datetime(self.historical_prices['date'] if 'date' in self.historical_prices.columns else self.historical_prices['time']).dt.date <= current_date
        ].tail(60)  # Last 60 days for correlation
        
        # Get options features for current date
        options_features_dict = {}
        if not self.historical_options_features.empty:
            date_options = self.historical_options_features[
                self.historical_options_features['date'] == current_date
            ] if 'date' in self.historical_options_features.columns else self.historical_options_features[
                pd.to_datetime(self.historical_options_features['time']).dt.date == current_date
            ]
            
            if not date_options.empty:
                for _, row in date_options.iterrows():
                    ticker = row['ticker']
                    options_features_dict[ticker] = {
                        'trend_signal': float(row.get('trend_signal', 0.0)),
                        'sentiment_signal': float(row.get('sentiment_signal', 0.0)),
                        'gamma_signal': float(row.get('gamma_signal', 0)),
                        'pcr_zscore': float(row.get('pcr_zscore', 0.0)),
                        'pcr_extreme_bullish': float(row.get('pcr_extreme_bullish', False)),
                        'pcr_extreme_bearish': float(row.get('pcr_extreme_bearish', False)),
                        'max_pain_distance_pct': float(row.get('max_pain_distance_pct', 0.0)),
                        'iv_percentile': float(row.get('iv_percentile', 0.5)),
                        'net_delta': float(row.get('net_delta', 0.0)),
                    }
        
        # Build state
        state = self.state_builder.build_state(
            twin_predictions=twin_predictions,
            features_df=date_features,
            portfolio_state=portfolio_state,
            macro_context=macro_context,
            prices_df=prices_df if not prices_df.empty else None,
            date=str(current_date),
            neo4j_client=None,
            options_features=options_features_dict if options_features_dict else None
        )
        
        return state
    
    def _empty_state(self) -> Dict:
        """Return empty state when out of bounds."""
        return {
            'twin_predictions': {},
            'features': {},
            'portfolio': {
                'cash': 1.0,
                'num_positions': 0,
                'positions': {},
                'sector_exposure': {},
                'portfolio_value': self.initial_capital,
                'peak_value': self.initial_capital,
            },
            'macro': {
                'vix': 0.4,
                'spy_return_5d': 0.0,
                'treasury_10y': 0.4,
                'market_regime': 'bull',
            },
            'correlation_graph': None,
            'ticker_to_idx': {},
            'tickers': []
        }
    
    def _execute_trades(self, action: Dict, date: datetime):
        """Execute trades based on action."""
        
        stock_selection = action.get('stock_selection', {})
        position_sizes = action.get('position_sizes', {})
        selected_indices = action.get('selected_indices', [])
        
        # Convert indices to tickers if needed
        if selected_indices and 'tickers' in action:
            selected_tickers = [action['tickers'][idx] for idx in selected_indices if idx < len(action['tickers'])]
        elif isinstance(stock_selection, dict):
            selected_tickers = [ticker for ticker, selected in stock_selection.items() if selected > 0.5]
        else:
            selected_tickers = []
        
        # Close positions not in selection
        to_close = [ticker for ticker in self.positions.keys() if ticker not in selected_tickers]
        for ticker in to_close:
            self._close_position(ticker, date)
        
        # Get current prices
        current_prices = self._get_prices(date)
        
        # Open/update positions
        total_allocation = 0.0
        for ticker in selected_tickers:
            if ticker not in current_prices:
                continue
            
            size = position_sizes.get(ticker, 0.0) if isinstance(position_sizes, dict) else 0.0
            if size <= 0:
                continue
            
            # Enforce max position size
            size = min(size, self.max_position_size)
            total_allocation += size
        
        # Normalize if exceeds 1.0
        if total_allocation > 1.0:
            scale = 1.0 / total_allocation
            for ticker in selected_tickers:
                if ticker in position_sizes:
                    position_sizes[ticker] *= scale
        
        # Execute trades
        for ticker in selected_tickers:
            if ticker not in current_prices:
                continue
            
            size = position_sizes.get(ticker, 0.0) if isinstance(position_sizes, dict) else 0.0
            if size <= 0:
                continue
            
            price = current_prices[ticker]
            
            if ticker in self.positions:
                # Update existing position
                old_size = self.positions[ticker]['size']
                if abs(old_size - size) > 0.01:  # Significant change
                    # Rebalance
                    self._close_position(ticker, date)
                    self._open_position(ticker, size, price, date)
            else:
                # New position
                self._open_position(ticker, size, price, date)
    
    def _open_position(self, ticker: str, size: float, price: float, date: datetime):
        """Open a new position."""
        
        position_value = self.portfolio_value * size
        shares = position_value / price
        
        # Apply transaction cost
        cost = position_value * self.transaction_cost
        self.cash -= (position_value + cost)
        
        self.positions[ticker] = {
            'size': size,
            'entry_price': price,
            'shares': shares,
            'days_held': 0,
        }
    
    def _close_position(self, ticker: str, date: datetime):
        """Close an existing position."""
        
        if ticker not in self.positions:
            return
        
        position = self.positions[ticker]
        current_price = self._get_price(ticker, date)
        
        if current_price is None:
            # Use entry price if no current price
            current_price = position['entry_price']
        
        exit_value = position['shares'] * current_price
        
        # Apply transaction cost
        cost = exit_value * self.transaction_cost
        self.cash += (exit_value - cost)
        
        # Record trade
        pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        self.closed_trades.append({
            'ticker': ticker,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'pnl_pct': pnl_pct,
            'days_held': position['days_held'],
        })
        
        del self.positions[ticker]
    
    def _update_portfolio_value(self, date: datetime):
        """Update portfolio value based on current prices."""
        
        current_prices = self._get_prices(date)
        
        # Update days held
        for ticker in self.positions:
            self.positions[ticker]['days_held'] += 1
        
        # Calculate position values
        position_value = 0.0
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                price = current_prices[ticker]
                position_value += position['shares'] * price
        
        # Total portfolio value
        self.portfolio_value = self.cash + position_value
        
        # Update peak
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
    
    def _get_prices(self, date: datetime) -> Dict[str, float]:
        """Get prices for all tickers on given date."""
        
        date_prices = self.historical_prices[
            pd.to_datetime(self.historical_prices['date'] if 'date' in self.historical_prices.columns else self.historical_prices['time']).dt.date == date
        ]
        
        prices = {}
        for _, row in date_prices.iterrows():
            ticker = row['ticker']
            prices[ticker] = float(row['close'])
        
        return prices
    
    def _get_price(self, ticker: str, date: datetime) -> Optional[float]:
        """Get price for a specific ticker."""
        prices = self._get_prices(date)
        return prices.get(ticker)
    
    def _compute_sector_exposure(self) -> Dict[str, float]:
        """Compute current sector exposure."""
        from src.utils.tickers import get_ticker_sector
        
        sector_exposure = {}
        for ticker, position in self.positions.items():
            if position['size'] > 0:
                sector = get_ticker_sector(ticker)
                if sector:
                    sector_exposure[sector] = sector_exposure.get(sector, 0.0) + position['size']
        
        return sector_exposure
    
    def _compute_sector_entropy(self, sector_exposure: Dict[str, float]) -> float:
        """Compute Shannon entropy of sector exposure."""
        if not sector_exposure:
            return 0.0
        
        import numpy as np
        
        # Normalize to probabilities
        total = sum(sector_exposure.values())
        if total == 0:
            return 0.0
        
        probs = [v / total for v in sector_exposure.values()]
        
        # Compute entropy
        entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
        
        return entropy
    
    def _compute_portfolio_volatility(self) -> float:
        """
        Compute portfolio volatility from closed trades.
        
        Returns rolling 20-day volatility of portfolio returns.
        """
        if len(self.closed_trades) < 5:
            return 0.02  # Default volatility
        
        import numpy as np
        
        # Get last 20 trades
        recent_trades = self.closed_trades[-20:]
        returns = [t['pnl_pct'] for t in recent_trades]
        
        # Compute volatility
        vol = np.std(returns)
        
        return max(vol, 0.001)  # Minimum volatility of 0.1%
    
    def _compute_reward(self, portfolio_value_before: float, action: Dict) -> float:
        """
        Compute reward based on portfolio performance and constraints.
        
        Reward components (as per RL Portfolio layer doc):
        1. Portfolio return (scaled 100x)
        2. Drawdown penalty
        3. Transaction costs
        4. Diversification bonus
        5. Sharpe adjustment
        6. Constraint penalties
        """
        
        # Portfolio return
        portfolio_return = (self.portfolio_value - portfolio_value_before) / portfolio_value_before if portfolio_value_before > 0 else 0.0
        
        # Drawdown penalty
        current_dd = (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0.0
        drawdown_penalty = -10.0 * max(0.0, current_dd)
        
        # Transaction costs (already applied, but penalize excessive trading)
        num_trades = action.get('num_new_positions', 0) + action.get('num_closed_positions', 0)
        transaction_cost_penalty = -0.001 * num_trades
        
        # Diversification bonus
        sector_exposure = self._compute_sector_exposure()
        sector_entropy = self._compute_sector_entropy(sector_exposure)
        diversification_bonus = 0.5 * sector_entropy
        
        # Sharpe adjustment (use actual portfolio volatility)
        portfolio_vol = self._compute_portfolio_volatility()
        sharpe_adjustment = portfolio_return / (portfolio_vol + 0.01) if portfolio_vol > 0 else 0.0
        
        # Constraint penalties
        penalties = 0.0
        
        # Max position size
        for position in self.positions.values():
            if position['size'] > self.max_position_size:
                penalties -= 5.0
        
        # Max sector exposure
        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector_exposure:
                penalties -= 5.0
        
        # Max concurrent positions
        num_positions = len([p for p in self.positions.values() if p['size'] > 0])
        if num_positions > self.max_positions:
            penalties -= 2.0
        
        # Total reward
        reward = (
            100.0 * portfolio_return +
            drawdown_penalty +
            transaction_cost_penalty +
            diversification_bonus +
            5.0 * sharpe_adjustment +
            penalties
        )
        
        return reward

