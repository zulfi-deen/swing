"""DEPRECATED: Sync trained models from Google Drive to local MacBook

This script is deprecated. The codebase now uses Lightning.ai for training and deployment.
Google Drive sync is no longer needed.

This file is kept for backward compatibility only and will be removed in a future version.
"""

import os
import shutil
import logging
from pathlib import Path
import argparse

from src.utils.config import load_config

logger = logging.getLogger(__name__)


def sync_from_drive(
    drive_path: str,
    local_models_dir: str,
    dry_run: bool = False
):
    """
    Sync trained models from Google Drive to local.
    
    Args:
        drive_path: Google Drive path (e.g., /Users/username/Google Drive/swing_trading)
        local_models_dir: Local models directory
        dry_run: If True, only print what would be copied
    """
    
    drive_models_path = os.path.join(drive_path, 'models')
    
    if not os.path.exists(drive_models_path):
        logger.error(f"Google Drive models directory not found: {drive_models_path}")
        logger.info("Make sure Google Drive is synced and models are in the correct location")
        return
    
    if not dry_run:
        os.makedirs(local_models_dir, exist_ok=True)
    
    # Copy models
    copied_files = []
    
    # Copy foundation models
    foundation_src = os.path.join(drive_models_path, 'foundation')
    foundation_dest = os.path.join(local_models_dir, 'foundation')
    
    if os.path.exists(foundation_src):
        if not dry_run:
            os.makedirs(foundation_dest, exist_ok=True)
        
        for file in os.listdir(foundation_src):
            src_file = os.path.join(foundation_src, file)
            dest_file = os.path.join(foundation_dest, file)
            
            if os.path.isfile(src_file):
                if dry_run:
                    logger.info(f"Would copy: {src_file} -> {dest_file}")
                else:
                    shutil.copy2(src_file, dest_file)
                    logger.info(f"Copied: {src_file} -> {dest_file}")
                copied_files.append(dest_file)
    
    # Copy twin models
    twins_src = os.path.join(drive_models_path, 'twins')
    twins_dest = os.path.join(local_models_dir, 'twins')
    
    if os.path.exists(twins_src):
        if not dry_run:
            os.makedirs(twins_dest, exist_ok=True)
        
        for ticker_dir in os.listdir(twins_src):
            ticker_src = os.path.join(twins_src, ticker_dir)
            ticker_dest = os.path.join(twins_dest, ticker_dir)
            
            if os.path.isdir(ticker_src):
                if not dry_run:
                    os.makedirs(ticker_dest, exist_ok=True)
                
                for file in os.listdir(ticker_src):
                    src_file = os.path.join(ticker_src, file)
                    dest_file = os.path.join(ticker_dest, file)
                    
                    if os.path.isfile(src_file):
                        if dry_run:
                            logger.info(f"Would copy: {src_file} -> {dest_file}")
                        else:
                            shutil.copy2(src_file, dest_file)
                            logger.info(f"Copied: {src_file} -> {dest_file}")
                        copied_files.append(dest_file)
    
    logger.info(f"Sync complete. {len(copied_files)} files {'would be ' if dry_run else ''}copied to {local_models_dir}")
    
    if dry_run:
        logger.info("\nTo actually sync, run without --dry-run flag")
        logger.info(f"Make sure Google Drive is syncing from: {drive_path}")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Sync trained models from Google Drive to local')
    parser.add_argument('--drive-path', required=True, help='Google Drive path (e.g., /Users/username/Google Drive/swing_trading)')
    parser.add_argument('--local-dir', help='Local models directory (default: models/)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (print what would be copied)')
    parser.add_argument('--config', help='Path to config file')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Get local models directory
    if args.local_dir:
        local_models_dir = args.local_dir
    else:
        config = load_config(args.config)
        models_dir = config.get('storage', {}).get('local', {}).get('models_dir', 'models/')
        project_root = Path(__file__).parent.parent
        local_models_dir = str(project_root / models_dir)
    
    sync_from_drive(args.drive_path, local_models_dir, args.dry_run)


if __name__ == '__main__':
    main()

