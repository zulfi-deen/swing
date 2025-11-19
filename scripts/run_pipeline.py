#!/usr/bin/env python3
"""Run the daily EOD pipeline"""

import sys
import os
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.orchestrator import run_daily_pipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run daily pipeline."""
    
    parser = argparse.ArgumentParser(description='Run daily EOD pipeline')
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Date to process (YYYY-MM-DD). Defaults to today.'
    )
    
    args = parser.parse_args()
    
    try:
        result = run_daily_pipeline(date=args.date)
        
        logger.info("Pipeline completed successfully")
        logger.info(f"Result: {result}")
        
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

