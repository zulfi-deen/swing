"""Backtesting framework"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import torch
import logging

logger = logging.getLogger(__name__)


class BacktestPortfolio:
    """Simple backtest portfolio tracker."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # ticker -> position dict
        self.closed_trades = []
    
    def enter_position(
        self,
        ticker: str,
        side: str,
        size: float,  # Fraction of portfolio
        target_pct: float,
        stop_pct: float,
        entry_price: float
    ):
        """Enter a new position."""
        
        position_value = self.cash * size
        shares = position_value / entry_price
        
        self.positions[ticker] = {
            'side': side,
            'shares': shares,
            'entry_price': entry_price,
            'target_price': entry_price * (1 + target_pct) if side == 'buy' else entry_price * (1 - target_pct),
            'stop_price': entry_price * (1 - stop_pct) if side == 'buy' else entry_price * (1 + stop_pct),
            'entry_date': datetime.now(),
            'target_pct': target_pct,
            'stop_pct': stop_pct
        }
        
        self.cash -= position_value
    
    def update(self, date: datetime, prices: Dict[str, float]):
        """Update positions and check for exits."""
        
        to_close = []
        
        for ticker, position in self.positions.items():
            if ticker not in prices:
                continue
            
            current_price = prices[ticker]
            
            if position['side'] == 'buy':
                if current_price >= position['target_price']:
                    to_close.append((ticker, current_price, 'target'))
                elif current_price <= position['stop_price']:
                    to_close.append((ticker, current_price, 'stop'))
            else:  # sell
                if current_price <= position['target_price']:
                    to_close.append((ticker, current_price, 'target'))
                elif current_price >= position['stop_price']:
                    to_close.append((ticker, current_price, 'stop'))
            
            # Time-based exit (5 days)
            if (date - position['entry_date']).days >= 5:
                to_close.append((ticker, current_price, 'time'))
        
        # Close positions
        for ticker, exit_price, reason in to_close:
            self._close_position(ticker, exit_price, reason)
    
    def _close_position(self, ticker: str, exit_price: float, reason: str):
        """Close a position."""
        
        if ticker not in self.positions:
            return
        
        position = self.positions[ticker]
        
        if position['side'] == 'buy':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        exit_value = position['shares'] * exit_price
        self.cash += exit_value
        
        self.closed_trades.append({
            'ticker': ticker,
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'entry_date': position['entry_date'],
            'exit_date': datetime.now(),
            'exit_reason': reason,
            'days_held': (datetime.now() - position['entry_date']).days
        })
        
        del self.positions[ticker]
    
    def total_return(self) -> float:
        """Calculate total return."""
        current_value = self.cash + sum(
            pos['shares'] * pos['entry_price'] for pos in self.positions.values()
        )
        return (current_value - self.initial_capital) / self.initial_capital
    
    def win_rate(self) -> float:
        """Calculate win rate."""
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t['pnl_pct'] > 0)
        return wins / len(self.closed_trades)
    
    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if not self.closed_trades:
            return 0.0
        
        returns = [t['pnl_pct'] for t in self.closed_trades]
        excess_returns = np.array(returns) - risk_free_rate / 252  # Daily
        
        if excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        # Simplified - would need equity curve in production
        if not self.closed_trades:
            return 0.0
        
        cumulative = np.cumsum([t['pnl_pct'] for t in self.closed_trades])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(drawdown.min())


def backtest_strategy(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    max_positions: int = 12
) -> Dict:
    """
    Walk-forward backtest using priority scoring.
    
    Args:
        predictions: DataFrame with recommendations (must have priority_score)
        prices: DataFrame with historical prices
        start_date: Start date for backtest
        end_date: End date for backtest
        max_positions: Maximum concurrent positions
    
    Returns:
        Metrics dictionary
    """
    
    portfolio = BacktestPortfolio(initial_capital=100000)
    
    # Filter predictions by date range
    predictions['date'] = pd.to_datetime(predictions['date']).dt.date
    prices['date'] = pd.to_datetime(prices['date'] if 'date' in prices.columns else prices['time']).dt.date
    
    start_date_obj = pd.to_datetime(start_date).date()
    end_date_obj = pd.to_datetime(end_date).date()
    
    predictions = predictions[
        (predictions['date'] >= start_date_obj) & 
        (predictions['date'] <= end_date_obj)
    ]
    
    dates = sorted(predictions['date'].unique())
    
    logger.info(f"Backtesting priority scoring strategy from {start_date} to {end_date}")
    logger.info(f"Processing {len(dates)} trading days")
    
    for date in dates:
        # Get today's recommendations (top N by priority score)
        today_preds = predictions[predictions['date'] == date].nlargest(max_positions, 'priority_score')
        
        # Get prices for today
        today_prices = prices[prices['date'] == date]
        price_dict = dict(zip(today_prices['ticker'], today_prices['close']))
        
        # Update existing positions (check for exits)
        portfolio.update(datetime(date.year, date.month, date.day), price_dict)
        
        # Enter new positions for stocks not already held
        for _, pred in today_preds.iterrows():
            ticker = pred['ticker']
            
            if ticker in portfolio.positions:
                continue  # Already holding
            
            if len(portfolio.positions) >= max_positions:
                break  # Portfolio full
            
            if ticker not in price_dict:
                continue  # No price data
            
            entry_price = price_dict[ticker]
            
            # Position size: Equal weight across max_positions
            position_size = pred.get('position_size_pct', 1.0 / max_positions)
            
            # Enter position
            portfolio.enter_position(
                ticker=ticker,
                side=pred.get('side', 'buy'),
                size=position_size,
                target_pct=pred.get('target_pct', 0.05),
                stop_pct=pred.get('stop_pct', 0.03),
                entry_price=entry_price
            )
    
    # Close all remaining positions at end
    final_prices = prices[prices['date'] == dates[-1]]
    final_price_dict = dict(zip(final_prices['ticker'], final_prices['close']))
    
    for ticker in list(portfolio.positions.keys()):
        if ticker in final_price_dict:
            portfolio._close_position(ticker, final_price_dict[ticker], 'backtest_end')
    
    # Metrics
    metrics = {
        'total_return': portfolio.total_return(),
        'sharpe_ratio': portfolio.sharpe_ratio(),
        'max_drawdown': portfolio.max_drawdown(),
        'win_rate': portfolio.win_rate(),
        'num_trades': len(portfolio.closed_trades),
        'final_value': portfolio.cash + sum(pos['shares'] * pos['entry_price'] for pos in portfolio.positions.values())
    }
    
    logger.info(f"Backtest complete: Return={metrics['total_return']:.2%}, Sharpe={metrics['sharpe_ratio']:.2f}, Win Rate={metrics['win_rate']:.2%}")
    
    return metrics


def backtest_rl_agent(
    agent,
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    features: pd.DataFrame,
    macro_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0
) -> Dict:
    """
    Backtest RL agent on historical data.
    
    Args:
        agent: Trained PortfolioRLAgent
        predictions: DataFrame with twin predictions
        prices: DataFrame with historical prices
        features: DataFrame with features
        macro_data: DataFrame with macro indicators
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Initial capital
    
    Returns:
        Metrics dictionary
    """
    
    from src.training.rl_environment import TradingEnvironment
    from src.data.rl_state_builder import RLStateBuilder
    
    # Create environment
    env = TradingEnvironment(
        historical_predictions=predictions,
        historical_prices=prices,
        historical_features=features,
        macro_data=macro_data,
        initial_capital=initial_capital,
        max_positions=15,
        max_position_size=0.10,
        max_sector_exposure=0.25,
        transaction_cost=0.001
    )
    
    # Run backtest
    state = env.reset(start_date=start_date)
    agent.eval()
    
    portfolio_values = [initial_capital]
    returns = []
    
    done = False
    while not done:
        with torch.no_grad():
            action, _, _, _ = agent(state, deterministic=True)
        
        state, reward, done, info = env.step(action)
        
        portfolio_value = info.get('portfolio_value', initial_capital)
        portfolio_values.append(portfolio_value)
        
        if len(portfolio_values) > 1:
            daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            returns.append(daily_return)
    
    # Calculate metrics
    total_return = (portfolio_values[-1] - initial_capital) / initial_capital
    
    if returns:
        returns_array = np.array(returns)
        sharpe = np.sqrt(252) * returns_array.mean() / (returns_array.std() + 1e-8)
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
    else:
        sharpe = 0.0
        max_drawdown = 0.0
    
    metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'final_portfolio_value': portfolio_values[-1],
        'num_trades': len(env.closed_trades),
        'win_rate': sum(1 for t in env.closed_trades if t['pnl_pct'] > 0) / len(env.closed_trades) if env.closed_trades else 0.0,
    }
    
    return metrics


def compare_rl_vs_priority_scoring(
    rl_agent,
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    features: pd.DataFrame,
    macro_data: pd.DataFrame,
    start_date: str,
    end_date: str
) -> Dict:
    """
    Compare RL agent vs priority scoring on same period.
    
    Returns:
        Dict with metrics for both approaches
    """
    
    # RL agent backtest
    rl_metrics = backtest_rl_agent(
        agent=rl_agent,
        predictions=predictions,
        prices=prices,
        features=features,
        macro_data=macro_data,
        start_date=start_date,
        end_date=end_date
    )
    
    # Priority scoring backtest
    priority_predictions = predictions.copy()
    priority_predictions['priority_score'] = (
        priority_predictions['expected_return'] *
        priority_predictions['hit_probability'] *
        (1 - priority_predictions['volatility'] * 10)
    )
    
    # Backtest using priority scores
    priority_metrics = backtest_strategy(
        predictions=priority_predictions,
        prices=prices,
        start_date=start_date,
        end_date=end_date,
        max_positions=12
    )
    
    comparison = {
        'rl_agent': rl_metrics,
        'priority_scoring': priority_metrics,
        'improvement': {
            'sharpe_ratio': rl_metrics['sharpe_ratio'] - priority_metrics['sharpe_ratio'],
            'max_drawdown': priority_metrics['max_drawdown'] - rl_metrics['max_drawdown'],  # Lower is better
            'total_return': rl_metrics['total_return'] - priority_metrics['total_return'],
        }
    }
    
    return comparison

