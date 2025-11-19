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
    
    # Override Neo4j config with environment variables
    if 'storage' in config and 'neo4j' in config['storage']:
        config['storage']['neo4j']['uri'] = os.getenv(
            'NEO4J_URI',
            config['storage']['neo4j'].get('uri', 'bolt://localhost:7687')
        )
        config['storage']['neo4j']['user'] = os.getenv(
            'NEO4J_USER',
            config['storage']['neo4j'].get('user', 'neo4j')
        )
        config['storage']['neo4j']['password'] = os.getenv(
            'NEO4J_PASSWORD',
            config['storage']['neo4j'].get('password', '')
        )
        config['storage']['neo4j']['database'] = os.getenv(
            'NEO4J_DATABASE',
            config['storage']['neo4j'].get('database', 'neo4j')
        )
    
    return config

