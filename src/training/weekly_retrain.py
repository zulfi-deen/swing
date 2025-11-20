"""Weekly Twin Retraining Pipeline

Automated weekly fine-tuning of all pilot twins.
Runs every Sunday at 2 AM.
"""

import logging
from datetime import datetime
from typing import List, Optional
from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect.blocks.system import Secret

from src.models.foundation import load_foundation_model
from src.training.train_twins import fine_tune_twin
from src.utils.config import load_config
from src.utils.mlflow_logger import log_training_run
from src.utils.colab_utils import setup_colab_environment, is_colab
import mlflow
import os

logger = logging.getLogger(__name__)


def check_alpha_decay(characteristics: dict, threshold: float = 0.0) -> dict:
    """
    Check if stock alpha has decayed below threshold (alpha decay guardrail).
    
    Args:
        characteristics: Dictionary with stock characteristics including 'current_alpha'
        threshold: Minimum acceptable alpha (default 0.0)
    
    Returns:
        Dictionary with:
            - 'decayed': bool indicating if alpha is below threshold
            - 'alpha': current alpha value
            - 'status': 'active' or 'decayed'
            - 'message': descriptive message
    """
    current_alpha = characteristics.get('current_alpha', 0.0)
    decayed = current_alpha < threshold
    
    result = {
        'decayed': decayed,
        'alpha': current_alpha,
        'status': 'decayed' if decayed else 'active',
        'message': (
            f"Alpha decay detected: {current_alpha:.4f} < {threshold:.4f}"
            if decayed
            else f"Alpha acceptable: {current_alpha:.4f} >= {threshold:.4f}"
        )
    }
    
    if decayed:
        logger.warning(f"ALPHA DECAY GUARDRAIL TRIGGERED: {result['message']}")
    else:
        logger.info(f"Alpha guardrail check passed: {result['message']}")
    
    return result


@task(cache_key_fn=task_input_hash)
def update_stock_characteristics_task(ticker: str):
    """Update stock characteristics for a ticker."""
    from src.data.storage import compute_stock_characteristics, update_stock_characteristics, get_timescaledb_engine
    import pandas as pd
    from datetime import timedelta
    
    engine = get_timescaledb_engine()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # Fetch stock price data
    stock_query = f"""
        SELECT * FROM prices
        WHERE ticker = '{ticker}'
        AND time >= '{start_date}'
        AND time <= '{end_date}'
        ORDER BY time
    """
    
    prices_df = pd.read_sql(stock_query, engine)
    
    # Fetch market (SPY) data for alpha calculation
    market_query = f"""
        SELECT * FROM prices
        WHERE ticker = 'SPY'
        AND time >= '{start_date}'
        AND time <= '{end_date}'
        ORDER BY time
    """
    
    market_df = pd.read_sql(market_query, engine)
    
    if not prices_df.empty:
        characteristics = compute_stock_characteristics(prices_df, market_df if not market_df.empty else None)
        update_stock_characteristics(ticker, characteristics)
        logger.info(f"Updated stock characteristics for {ticker}: beta={characteristics.get('beta', 1.0):.2f}, alpha={characteristics.get('current_alpha', 0.0):.4f}")
    else:
        logger.warning(f"No price data found for {ticker}")


@task
def fine_tune_twin_task(ticker: str, foundation_model, config: dict):
    """Fine-tune twin for a ticker."""
    try:
        twin = fine_tune_twin(ticker, foundation_model, config)
        logger.info(f"Successfully fine-tuned twin for {ticker}")
        # Extract metrics from twin if available
        metrics = {}
        if hasattr(twin, 'get_stock_characteristics'):
            chars = twin.get_stock_characteristics()
            metrics = {'current_alpha': chars.get('current_alpha', 0.0)}
        return {'ticker': ticker, 'status': 'success', 'metrics': metrics}
    except Exception as e:
        logger.error(f"Error fine-tuning twin for {ticker}: {e}")
        return {'ticker': ticker, 'status': 'error', 'error': str(e), 'metrics': {}}


@flow(name="weekly-twin-retraining")
def weekly_twin_retrain_flow(config_path: Optional[str] = None):
    """
    Weekly retraining flow for all pilot twins.
    
    Runs every Sunday at 2 AM.
    """
    
    config = load_config(config_path)
    
    # Setup Colab environment if needed
    data_path, models_path = setup_colab_environment(config)
    logger.info(f"Using data_path: {data_path}, models_path: {models_path}")
    
    # Update foundation checkpoint path if in Colab
    if is_colab():
        foundation_config = config.get('models', {}).get('foundation', {})
        if not foundation_config.get('checkpoint_path'):
            foundation_config['checkpoint_path'] = os.path.join(models_path, 'foundation', 'foundation_v1.0.pt')
    
    # Get pilot tickers
    pilot_tickers = config.get('models', {}).get('twins', {}).get('pilot_tickers', [])
    
    if not pilot_tickers:
        logger.warning("No pilot tickers configured, skipping retraining")
        return
    
    logger.info(f"Starting weekly retraining for {len(pilot_tickers)} twins")
    
    # Load foundation model
    foundation_path = config.get('models', {}).get('foundation', {}).get('checkpoint_path')
    if not foundation_path:
        logger.error("Foundation model checkpoint path not configured")
        return
    
    # Update path if in Colab and path doesn't exist
    if is_colab() and not os.path.exists(foundation_path):
        # Try to find it in models_path
        alt_path = os.path.join(models_path, 'foundation', os.path.basename(foundation_path))
        if os.path.exists(alt_path):
            foundation_path = alt_path
            logger.info(f"Using foundation model from: {foundation_path}")
    
    try:
        foundation_model = load_foundation_model(foundation_path, config.get('models', {}).get('foundation', {}))
    except Exception as e:
        logger.error(f"Error loading foundation model: {e}")
        return
    
    # Get alpha decay threshold from config (default 0.0)
    alpha_threshold = config.get('trading_rules', {}).get('min_alpha', 0.0)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"weekly_retrain_{datetime.now().strftime('%Y%m%d')}"):
        results = []
        decayed_tickers = []
        
        for ticker in pilot_tickers:
            logger.info(f"Processing {ticker}")
            
            # Update stock characteristics
            update_stock_characteristics_task(ticker)
            
            # Check alpha decay guardrail
            from src.data.storage import get_stock_characteristics
            characteristics = get_stock_characteristics(ticker)
            
            if characteristics:
                guardrail_result = check_alpha_decay(characteristics, threshold=alpha_threshold)
                
                # Log guardrail status to MLflow
                mlflow.log_metric(f"{ticker}_alpha", guardrail_result['alpha'])
                mlflow.log_metric(f"{ticker}_alpha_decayed", 1 if guardrail_result['decayed'] else 0)
                
                # If alpha has decayed, mark ticker and optionally skip fine-tuning
                if guardrail_result['decayed']:
                    decayed_tickers.append(ticker)
                    logger.warning(
                        f"ALPHA DECAY GUARDRAIL: {ticker} has decayed alpha ({guardrail_result['alpha']:.4f}). "
                        f"Skipping fine-tuning for this ticker."
                    )
                    
                    # Mark as inactive in characteristics
                    characteristics['status'] = 'decayed'
                    characteristics['guardrail_message'] = guardrail_result['message']
                    from src.data.storage import update_stock_characteristics
                    update_stock_characteristics(ticker, characteristics)
                    
                    # Add to results with decayed status
                    results.append({
                        'ticker': ticker,
                        'status': 'decayed',
                        'error': guardrail_result['message'],
                        'metrics': {'current_alpha': guardrail_result['alpha']}
                    })
                    continue
            
            # Fine-tune twin (only if alpha is acceptable)
            result = fine_tune_twin_task(ticker, foundation_model, config)
            results.append(result)
            
            # Log to MLflow
            mlflow.log_metric(f"{ticker}_status", 1 if result['status'] == 'success' else 0)
            
            # Log twin metrics if available
            if result.get('metrics'):
                for metric_name, metric_value in result['metrics'].items():
                    mlflow.log_metric(f"{ticker}_{metric_name}", metric_value)
        
        # Summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')
        decayed = sum(1 for r in results if r['status'] == 'decayed')
        
        logger.info(
            f"Retraining complete: {successful} successful, {failed} failed, {decayed} decayed (alpha guardrail)"
        )
        
        mlflow.log_metric("total_twins", len(results))
        mlflow.log_metric("successful", successful)
        mlflow.log_metric("failed", failed)
        mlflow.log_metric("decayed", decayed)
        
        if decayed_tickers:
            logger.warning(
                f"Alpha decay guardrail triggered for {len(decayed_tickers)} tickers: {', '.join(decayed_tickers)}"
            )
        
        return results


if __name__ == '__main__':
    # Run the flow
    weekly_twin_retrain_flow()


