"""Colab utilities for environment detection and Google Drive mounting"""

import os
import sys
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """
    Detect if running in Google Colab environment.
    
    Returns:
        True if running in Colab, False otherwise
    """
    try:
        # Check for Colab environment variables
        if os.getenv('COLAB_GPU') is not None:
            return True
        
        # Check if google.colab module is available
        if 'google.colab' in sys.modules:
            return True
        
        # Check for Colab-specific paths
        if '/content' in str(Path.cwd()):
            return True
        
        return False
    except Exception:
        return False


def mount_google_drive(mount_path: str = "/content/drive") -> bool:
    """
    Mount Google Drive in Colab.
    
    Args:
        mount_path: Path to mount Google Drive
    
    Returns:
        True if mount successful, False otherwise
    """
    if not is_colab():
        logger.warning("Not running in Colab, skipping Google Drive mount")
        return False
    
    try:
        from google.colab import drive
        drive.mount(mount_path)
        logger.info(f"Google Drive mounted at {mount_path}")
        return True
    except ImportError:
        logger.error("google.colab.drive not available")
        return False
    except Exception as e:
        logger.error(f"Error mounting Google Drive: {e}")
        return False


def get_data_paths(config: Dict) -> Tuple[str, str]:
    """
    Get data and model paths based on environment.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (data_path, models_path)
    """
    if is_colab():
        # Use Google Drive paths in Colab
        colab_config = config.get('training', {}).get('colab', {})
        data_path = colab_config.get('data_path', '/content/drive/MyDrive/swing_trading/data')
        models_path = colab_config.get('models_path', '/content/drive/MyDrive/swing_trading/models')
        
        # Ensure Google Drive is mounted
        mount_path = config.get('storage', {}).get('google_drive', {}).get('mount_path', '/content/drive')
        if not os.path.exists(mount_path):
            logger.info("Mounting Google Drive...")
            mount_google_drive(mount_path)
        
        return data_path, models_path
    else:
        # Use local paths on MacBook
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
    Ensure data and model directories exist.
    
    Args:
        data_path: Path to data directory
        models_path: Path to models directory
    """
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
    Setup Colab environment (mount Drive, ensure directories).
    
    Args:
        config: Optional configuration dict
    
    Returns:
        Tuple of (data_path, models_path)
    """
    if config is None:
        from src.utils.config import load_config
        config = load_config()
    
    data_path, models_path = get_data_paths(config)
    ensure_directories(data_path, models_path)
    
    return data_path, models_path

