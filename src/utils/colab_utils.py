"""DEPRECATED: Colab utilities for environment detection and Google Drive mounting

This module is deprecated. The codebase now uses Lightning.ai for training and deployment.
All path resolution should use config['storage']['local'] paths directly.

This file is kept for backward compatibility only and will be removed in a future version.
"""

import os
import sys
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """
    DEPRECATED: Detect if running in Google Colab environment.
    
    Returns:
        False (always returns False as Colab is no longer supported)
    """
    logger.warning("is_colab() is deprecated. Use Lightning.ai for training instead.")
    return False


def mount_google_drive(mount_path: str = "/content/drive") -> bool:
    """
    DEPRECATED: Mount Google Drive in Colab.
    
    Args:
        mount_path: Path to mount Google Drive
    
    Returns:
        False (always returns False as Colab is no longer supported)
    """
    logger.warning("mount_google_drive() is deprecated. Use Lightning.ai for training instead.")
    return False


def get_data_paths(config: Dict) -> Tuple[str, str]:
    """
    DEPRECATED: Get data and model paths based on environment.
    
    Use config['storage']['local'] paths directly instead.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (data_path, models_path) from config
    """
    logger.warning("get_data_paths() is deprecated. Use config['storage']['local'] paths directly.")
    
    # Return paths from config (for backward compatibility)
    storage_config = config.get('storage', {}).get('local', {})
    data_path = storage_config.get('data_dir', 'data/')
    models_path = storage_config.get('models_dir', 'models/')
    
    # Make absolute paths
    project_root = Path(__file__).parent.parent.parent
    if not os.path.isabs(data_path):
        data_path = str(project_root / data_path)
    if not os.path.isabs(models_path):
        models_path = str(project_root / models_path)
    
    return data_path, models_path


def ensure_directories(data_path: str, models_path: str):
    """
    DEPRECATED: Ensure data and model directories exist.
    
    Args:
        data_path: Path to data directory
        models_path: Path to models directory
    """
    logger.warning("ensure_directories() is deprecated. Create directories directly as needed.")
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(data_path, 'raw', 'prices'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'raw', 'news'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'raw', 'fundamentals'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'raw', 'macro'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'processed', 'features'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'feast'), exist_ok=True)
    os.makedirs(os.path.join(models_path, 'foundation'), exist_ok=True)
    os.makedirs(os.path.join(models_path, 'twins'), exist_ok=True)
    
    logger.info(f"Ensured directories exist: {data_path}, {models_path}")


def setup_colab_environment(config: Optional[Dict] = None) -> Tuple[str, str]:
    """
    DEPRECATED: Setup Colab environment (mount Drive, ensure directories).
    
    Use config['storage']['local'] paths directly instead.
    
    Args:
        config: Optional configuration dict
    
    Returns:
        Tuple of (data_path, models_path) from config
    """
    logger.warning("setup_colab_environment() is deprecated. Use Lightning.ai for training instead.")
    
    if config is None:
        from src.utils.config import load_config
        config = load_config()
    
    return get_data_paths(config)
