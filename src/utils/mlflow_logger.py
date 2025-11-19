"""MLflow integration for experiment tracking"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from typing import Dict, Optional
import os
import logging

logger = logging.getLogger(__name__)


def init_mlflow(tracking_uri: Optional[str] = None, experiment_name: str = "swing_trading"):
    """Initialize MLflow tracking."""
    
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"Could not set experiment: {e}")


def log_training_run(
    metrics: Dict[str, float],
    params: Dict,
    model_version: str,
    tags: Optional[Dict] = None
):
    """Log a training run to MLflow."""
    
    init_mlflow()
    
    with mlflow.start_run(run_name=f"training_{model_version}"):
        # Log parameters
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, str, bool)):
                        mlflow.log_param(f"{key}.{sub_key}", sub_value)
        
        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Log tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        mlflow.set_tag("model_version", model_version)
        
        logger.info(f"Logged training run: {model_version}")


def log_model(
    model,
    model_name: str,
    model_type: str = "pytorch",
    artifact_path: str = "model",
    registered_model_name: Optional[str] = None
):
    """Log a model to MLflow."""
    
    init_mlflow()
    
    try:
        if model_type == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path)
        elif model_type == "sklearn" or model_type == "lightgbm":
            mlflow.sklearn.log_model(model, artifact_path)
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return
        
        if registered_model_name:
            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}", registered_model_name)
        
        logger.info(f"Logged model: {model_name}")
    except Exception as e:
        logger.error(f"Error logging model: {e}")


def load_model(model_name: str, version: Optional[str] = None, model_type: str = "pytorch"):
    """Load a model from MLflow."""
    
    init_mlflow()
    
    try:
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        if model_type == "pytorch":
            return mlflow.pytorch.load_model(model_uri)
        elif model_type == "sklearn" or model_type == "lightgbm":
            return mlflow.sklearn.load_model(model_uri)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

