"""DEPRECATED: Sync local data to Google Drive

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


def sync_to_drive(
    local_data_dir: str,
    drive_path: str,
    dry_run: bool = False
):
    """
    Sync local data to Google Drive.
    
    Args:
        local_data_dir: Local directory containing training data
        drive_path: Google Drive path (e.g., /Users/username/Google Drive/swing_trading)
        dry_run: If True, only print what would be copied
    """
    
    if not os.path.exists(local_data_dir):
        logger.error(f"Local data directory not found: {local_data_dir}")
        return
    
    # Ensure drive path exists
    drive_data_path = os.path.join(drive_path, 'data', 'training')
    
    if not dry_run:
        os.makedirs(drive_data_path, exist_ok=True)
    
    # Copy files
    copied_files = []
    for root, dirs, files in os.walk(local_data_dir):
        # Calculate relative path
        rel_path = os.path.relpath(root, local_data_dir)
        if rel_path == '.':
            dest_dir = drive_data_path
        else:
            dest_dir = os.path.join(drive_data_path, rel_path)
        
        if not dry_run:
            os.makedirs(dest_dir, exist_ok=True)
        
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, file)
            
            if dry_run:
                logger.info(f"Would copy: {src_file} -> {dest_file}")
            else:
                shutil.copy2(src_file, dest_file)
                logger.info(f"Copied: {src_file} -> {dest_file}")
            
            copied_files.append(dest_file)
    
    logger.info(f"Sync complete. {len(copied_files)} files {'would be ' if dry_run else ''}copied to {drive_data_path}")
    
    if dry_run:
        logger.info("\nTo actually sync, run without --dry-run flag")
        logger.info(f"Make sure Google Drive is syncing to: {drive_path}")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Sync local data to Google Drive')
    parser.add_argument('--local-dir', help='Local data directory (default: data/training)')
    parser.add_argument('--drive-path', required=True, help='Google Drive path (e.g., /Users/username/Google Drive/swing_trading)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (print what would be copied)')
    parser.add_argument('--config', help='Path to config file')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Get local directory
    if args.local_dir:
        local_data_dir = args.local_dir
    else:
        config = load_config(args.config)
        data_dir = config.get('storage', {}).get('local', {}).get('data_dir', 'data/')
        project_root = Path(__file__).parent.parent
        local_data_dir = str(project_root / data_dir / 'training')
    
    sync_to_drive(local_data_dir, args.drive_path, args.dry_run)


if __name__ == '__main__':
    main()

