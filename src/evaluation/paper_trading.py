"""Paper trading tracker"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional
import logging

from src.data.storage import get_timescaledb_engine, save_to_timescaledb

logger = logging.getLogger(__name__)


class PaperTradingTracker:
    """Track live recommendations without real money."""
    
    def __init__(self):
        self.db = get_timescaledb_engine()
    
    def record_recommendation(self, date: str, trades: List[Dict]):
        """Save today's recommendations as paper trades."""
        
        records = []
        for trade in trades:
            # Get current price (simplified - would fetch from API)
            entry_price = self._get_current_price(trade['ticker'])
            
            if entry_price is None:
                logger.warning(f"Could not get price for {trade['ticker']}, skipping")
                continue
            
            target_price = entry_price * (1 + trade['target_pct']) if trade['side'] == 'buy' else entry_price * (1 - trade['target_pct'])
            stop_price = entry_price * (1 - trade['stop_pct']) if trade['side'] == 'buy' else entry_price * (1 + trade['stop_pct'])
            
            records.append({
                'date': date,
                'ticker': trade['ticker'],
                'side': trade['side'],
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_price': stop_price,
                'predicted_probability': trade['probability'],
                'status': 'open',
                'created_at': datetime.now()
            })
        
        if records:
            df = pd.DataFrame(records)
            save_to_timescaledb(df, table="paper_trades", if_exists="append")
            logger.info(f"Recorded {len(records)} paper trades for {date}")
    
    def update_positions(self, date: str):
        """Check if any paper trades hit target/stop."""
        
        query = """
            SELECT * FROM paper_trades
            WHERE status = 'open' AND date <= %s
        """
        
        open_trades = pd.read_sql(query, self.db, params=(date,))
        
        if open_trades.empty:
            return
        
        for _, trade in open_trades.iterrows():
            current_price = self._get_current_price(trade['ticker'])
            
            if current_price is None:
                continue
            
            exit_reason = None
            
            if trade['side'] == 'buy':
                if current_price >= trade['target_price']:
                    exit_reason = 'target'
                elif current_price <= trade['stop_price']:
                    exit_reason = 'stop'
            else:  # sell
                if current_price <= trade['target_price']:
                    exit_reason = 'target'
                elif current_price >= trade['stop_price']:
                    exit_reason = 'stop'
            
            # Time-based exit (5 days)
            days_held = (datetime.strptime(date, "%Y-%m-%d") - trade['created_at'].date()).days
            if days_held >= 5 and exit_reason is None:
                exit_reason = 'time'
            
            if exit_reason:
                self._close_trade(trade['id'], current_price, exit_reason, date)
    
    def _close_trade(self, trade_id: int, exit_price: float, reason: str, exit_date: str):
        """Close a paper trade."""
        
        # Get trade details
        query = "SELECT * FROM paper_trades WHERE id = %s"
        trade = pd.read_sql(query, self.db, params=(trade_id,)).iloc[0]
        
        # Calculate P&L
        if trade['side'] == 'buy':
            pnl_pct = (exit_price - trade['entry_price']) / trade['entry_price']
        else:
            pnl_pct = (trade['entry_price'] - exit_price) / trade['entry_price']
        
        # Update trade
        from sqlalchemy import text
        
        update_query = text("""
            UPDATE paper_trades
            SET status = 'closed',
                exit_price = :exit_price,
                exit_date = :exit_date,
                pnl_pct = :pnl_pct,
                exit_reason = :exit_reason
            WHERE id = :trade_id
        """)
        
        with self.db.connect() as conn:
            conn.execute(update_query, {
                'exit_price': exit_price,
                'exit_date': exit_date,
                'pnl_pct': pnl_pct,
                'exit_reason': reason,
                'trade_id': trade_id
            })
            conn.commit()
        
        logger.info(f"Closed trade {trade_id}: {reason}, P&L: {pnl_pct:.2%}")
    
    def _get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for a ticker."""
        
        # Simplified - would fetch from real-time API
        # For now, get latest from database
        query = """
            SELECT close FROM prices
            WHERE ticker = %s
            ORDER BY time DESC
            LIMIT 1
        """
        
        try:
            result = pd.read_sql(query, self.db, params=(ticker,))
            if not result.empty:
                return float(result.iloc[0]['close'])
        except Exception as e:
            logger.error(f"Error getting price for {ticker}: {e}")
        
        return None
    
    def get_performance_report(self) -> Dict:
        """Generate performance report."""
        
        query = """
            SELECT * FROM paper_trades
            WHERE status = 'closed'
        """
        
        closed_trades = pd.read_sql(query, self.db)
        
        if closed_trades.empty:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        wins = closed_trades[closed_trades['pnl_pct'] > 0]
        losses = closed_trades[closed_trades['pnl_pct'] <= 0]
        
        metrics = {
            'num_trades': len(closed_trades),
            'win_rate': len(wins) / len(closed_trades) if len(closed_trades) > 0 else 0.0,
            'total_return': closed_trades['pnl_pct'].sum(),
            'avg_win': wins['pnl_pct'].mean() if len(wins) > 0 else 0.0,
            'avg_loss': losses['pnl_pct'].mean() if len(losses) > 0 else 0.0,
        }
        
        # Profit factor
        total_profit = wins['pnl_pct'].sum() if len(wins) > 0 else 0.0
        total_loss = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0.0
        metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else 0.0
        
        # Calibration error
        if 'predicted_probability' in closed_trades.columns:
            hit_target = (closed_trades['exit_reason'] == 'target').astype(float)
            metrics['calibration_error'] = (
                closed_trades['predicted_probability'] - hit_target
            ).abs().mean()
        
        return metrics

