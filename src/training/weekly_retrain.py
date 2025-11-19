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


@task(cache_key_fn=task_input_hash)
def update_stock_characteristics_task(ticker: str):
    """Update stock characteristics for a ticker."""
    from src.data.storage import compute_stock_characteristics, update_stock_characteristics, get_timescaledb_engine
    import pandas as pd
    from datetime import timedelta
    
    engine = get_timescaledb_engine()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    query = f"""
        SELECT * FROM prices
        WHERE ticker = '{ticker}'
        AND time >= '{start_date}'
        AND time <= '{end_date}'
        ORDER BY time
    """
    
    prices_df = pd.read_sql(query, engine)
    
    if not prices_df.empty:
        characteristics = compute_stock_characteristics(prices_df, ticker)
        update_stock_characteristics(ticker, characteristics)
        logger.info(f"Updated stock characteristics for {ticker}")
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
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"weekly_retrain_{datetime.now().strftime('%Y%m%d')}"):
        results = []
        
        for ticker in pilot_tickers:
            logger.info(f"Processing {ticker}")
            
            # Update stock characteristics
            update_stock_characteristics_task(ticker)
            
            # Fine-tune twin
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
        failed = len(results) - successful
        
        logger.info(f"Retraining complete: {successful} successful, {failed} failed")
        
        mlflow.log_metric("total_twins", len(results))
        mlflow.log_metric("successful", successful)
        mlflow.log_metric("failed", failed)
        
        return results


if __name__ == '__main__':
    # Run the flow
    weekly_twin_retrain_flow()


