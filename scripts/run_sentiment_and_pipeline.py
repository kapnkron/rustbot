#!/usr/bin/env python3
"""
Run Sentiment Crawler and ML Pipeline

This script first runs the sentiment crawler to collect sentiment data,
then runs the ML pipeline to incorporate this data into the model training process.
"""

import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Path to the Python executable in the correct virtual environment
VENV_PYTHON = os.path.expanduser("~/.pytorch_venv/bin/python")

def run_command(cmd, description):
    """Run a command and log its output."""
    logging.info(f"Running: {description}")
    logging.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        logging.info(f"{description} completed successfully")
        logging.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{description} failed with exit code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run sentiment crawler and ML pipeline')
    parser.add_argument('--days', type=int, default=30, help='Number of days of historical data to fetch')
    parser.add_argument('--token', type=str, help='Process only a specific token symbol')
    parser.add_argument('--max-tweets', type=int, default=500, help='Maximum number of tweets to fetch per token')
    parser.add_argument('--skip-sentiment', action='store_true', help='Skip running the sentiment crawler')
    parser.add_argument('--skip-pipeline', action='store_true', help='Skip running the ML pipeline')
    args = parser.parse_args()

    # Path to the scripts
    script_dir = Path(__file__).parent
    sentiment_crawler = script_dir / 'sentiment_crawler.py'
    pipeline_script = script_dir / 'run_pipeline.py'
    
    # Check if scripts exist
    if not sentiment_crawler.exists():
        logging.error(f"Sentiment crawler script not found at {sentiment_crawler}")
        return False
    
    if not pipeline_script.exists():
        logging.error(f"Pipeline script not found at {pipeline_script}")
        return False
    
    # Check if the virtual environment Python exists
    if not os.path.exists(VENV_PYTHON):
        logging.error(f"Python executable not found at {VENV_PYTHON}")
        logging.error("Please ensure the .pytorch_venv virtual environment is set up correctly")
        return False
    
    success = True
    
    # Step 1: Run sentiment crawler
    if not args.skip_sentiment:
        cmd = [VENV_PYTHON, str(sentiment_crawler)]
        
        if args.days:
            cmd.extend(['--days', str(args.days)])
        
        if args.token:
            cmd.extend(['--token', args.token])
        
        if args.max_tweets:
            cmd.extend(['--max-tweets', str(args.max_tweets)])
        
        success = run_command(cmd, "Sentiment Crawler") and success
    else:
        logging.info("Skipping sentiment crawler as requested")
    
    # Step 2: Run ML pipeline if sentiment crawler was successful
    if success and not args.skip_pipeline:
        # Run the pipeline script with appropriate arguments
        cmd = [VENV_PYTHON, str(pipeline_script)]
        
        # Add any necessary pipeline arguments here
        if args.token:
            cmd.extend(['--token', args.token])
        
        success = run_command(cmd, "ML Pipeline") and success
    elif args.skip_pipeline:
        logging.info("Skipping ML pipeline as requested")
    else:
        logging.error("Skipping ML pipeline due to previous errors")
    
    if success:
        logging.info("Complete process finished successfully")
    else:
        logging.error("Process completed with errors")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 