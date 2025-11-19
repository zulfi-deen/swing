"""Legacy training script - use scripts/train.py for Lightning-based training

This file is kept for backward compatibility but delegates to the new Lightning-based
training scripts.
"""

import logging

logger = logging.getLogger(__name__)


def main():
    """Main training function - redirects to new Lightning-based training."""
    logger.warning(
        "This script is deprecated. Use scripts/train.py for Lightning-based training.\n"
        "Example: python scripts/train.py foundation --use-synthetic"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

