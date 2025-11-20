"""Configuration loader"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            '../../config/config.yaml'
        )
    
    if not os.path.exists(config_path):
        # Try example config
        example_path = config_path.replace('config.yaml', 'config.example.yaml')
        if os.path.exists(example_path):
            print(f"Warning: Using example config. Copy to config.yaml and update values.")
            config_path = example_path
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables
    if 'api_keys' in config:
        config['api_keys']['polygon']['api_key'] = os.getenv(
            'POLYGON_API_KEY',
            config['api_keys']['polygon'].get('api_key', '')
        )
        if 'alphavantage' in config['api_keys']:
            config['api_keys']['alphavantage']['api_key'] = os.getenv(
                'ALPHAVANTAGE_API_KEY',
                config['api_keys']['alphavantage'].get('api_key', '')
            )
        config['api_keys']['finnhub']['api_key'] = os.getenv(
            'FINNHUB_API_KEY',
            config['api_keys']['finnhub'].get('api_key', '')
        )
        config['api_keys']['openai']['api_key'] = os.getenv(
            'OPENAI_API_KEY',
            config['api_keys']['openai'].get('api_key', '')
        )
    
    # Override database config with environment variables
    if 'storage' in config and 'timescaledb' in config['storage']:
        db_config = config['storage']['timescaledb']
        db_config['host'] = os.getenv('TIMESCALEDB_HOST', db_config.get('host', 'localhost'))
        db_config['port'] = int(os.getenv('TIMESCALEDB_PORT', db_config.get('port', 5432)))
        db_config['database'] = os.getenv('TIMESCALEDB_DATABASE', db_config.get('database', 'swing_trading'))
        db_config['user'] = os.getenv('TIMESCALEDB_USER', db_config.get('user', 'postgres'))
        db_config['password'] = os.getenv('TIMESCALEDB_PASSWORD', db_config.get('password', ''))

    # Graph storage is parquet-based (graphs stored in data/graphs/ directory)
    
    return config

