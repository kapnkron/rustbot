import subprocess
import sys
import os
import logging
import time
import argparse
import pandas as pd
import numpy as np
import json
import glob
import threading
import queue
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Global variables for progressive training
PROGRESSIVE_TRAINING = True  # Enable/disable progressive training
MIN_DATA_POINTS_FOR_TRAINING = 10000  # Minimum data points needed before first training
RETRAINING_INTERVAL_HOURS = 6  # How often to retrain during long data collection

# Global state for progressive training
last_training_time = None
is_training_running = False
training_thread = None
data_collection_start_time = None
total_data_points = 0
training_lock = threading.Lock()

def get_scripts(initial_run: bool, days_override: int = None) -> list:
    """Returns the list of scripts to be executed based on the run mode."""
    python_executable = sys.executable # Use the same python that's running this script
    snscrape_python = os.path.expanduser('~/.snscrape_venv/bin/python')
    
    if days_override is not None:
        fetch_days = str(days_override)
        fetch_desc = f"Fetching {fetch_days}-day historical data for all viable pools (manual override)"
    elif initial_run:
        fetch_days = "30"
        fetch_desc = "Fetching initial 30-day historical data for all viable pools"
    else:
        fetch_days = "2" # Fetch last 2 days for rolling updates
        fetch_desc = "Fetching rolling 2-day historical data for all viable pools"

    scripts = [
        ("Fetching top tokens", [python_executable, "scripts/fetch_top_solana_tokens.py"]),
        ("Delay for rate limit", ["sleep", "2"]),
        ("Filtering by viable pools", [python_executable, "scripts/fetch_pool_addresses.py"])
    ]
    
    # Use snscrape venv for sentiment step
    if not os.environ.get('SKIP_SENTIMENT') == 'true':
        scripts.append(("Collecting sentiment data", [snscrape_python, "scripts/sentiment_crawler.py", "--days", fetch_days]))
    else:
        logging.info("Skipping sentiment data collection as SKIP_SENTIMENT is set to true")
    
    scripts.extend([
        (fetch_desc, [python_executable, "scripts/fetch_historical_ohlcv.py", "--days", fetch_days, "--threads", "10"]),
        ("Preparing training data (including existing 6-month SOL-USDC data)", [python_executable, "scripts/prepare_training_data.py", "--include-6mo"]),
        ("Engineering features", [python_executable, "scripts/add_features.py"]),
        ("Training ML model", [python_executable, "train_model.py"]),
    ])
    
    return scripts

def get_data_collection_scripts(initial_run: bool, days_override: int = None) -> list:
    """Returns just the data collection scripts without training."""
    python_executable = sys.executable
    snscrape_python = os.path.expanduser('~/.snscrape_venv/bin/python')
    
    if days_override is not None:
        fetch_days = str(days_override)
        fetch_desc = f"Fetching {fetch_days}-day historical data for all viable pools (manual override)"
    elif initial_run:
        fetch_days = "30"
        fetch_desc = "Fetching initial 30-day historical data for all viable pools"
    else:
        fetch_days = "2" # Fetch last 2 days for rolling updates
        fetch_desc = "Fetching rolling 2-day historical data for all viable pools"

    scripts = [
        ("Fetching top tokens", [python_executable, "scripts/fetch_top_solana_tokens.py"]),
        ("Delay for rate limit", ["sleep", "2"]),
        ("Filtering by viable pools", [python_executable, "scripts/fetch_pool_addresses.py"])
    ]
    
    # Use snscrape venv for sentiment step
    if not os.environ.get('SKIP_SENTIMENT') == 'true':
        scripts.append(("Collecting sentiment data", [snscrape_python, "scripts/sentiment_crawler.py", "--days", fetch_days]))
    else:
        logging.info("Skipping sentiment data collection as SKIP_SENTIMENT is set to true")
    
    scripts.append((fetch_desc, [python_executable, "scripts/fetch_historical_ohlcv.py", "--days", fetch_days, "--threads", "10"]))
    
    return scripts

def get_training_scripts() -> list:
    """Returns just the training part of the pipeline."""
    python_executable = sys.executable
    
    return [
        ("Preparing training data (including existing 6-month SOL-USDC data)", [python_executable, "scripts/prepare_training_data.py", "--include-6mo"]),
        ("Engineering features", [python_executable, "scripts/add_features.py"]),
        ("Training ML model", [python_executable, "train_model.py"]),
    ]

SIGNALS_FILE = "signals.jsonl"

# Dummy signal generation for demonstration (replace with real logic as needed)
def generate_signals():
    # In a real pipeline, this would use model outputs to generate signals
    # Here, we just write a placeholder file
    import json
    signals = [
        {"token": "So11111111111111111111111111111111111111112", "signal": "buy", "confidence": 0.95, "price": 156.23, "timestamp": "2025-06-01T12:00:00Z"},
        {"token": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "signal": "sell", "confidence": 0.81, "price": 1.00, "timestamp": "2025-06-01T12:00:00Z"}
    ]
    with open(SIGNALS_FILE, "w") as f:
        for sig in signals:
            f.write(json.dumps(sig) + "\n")
    logging.info(f"Wrote {len(signals)} trading signals to {SIGNALS_FILE}")

def count_total_data_points():
    """Count the total number of data points available for training."""
    global total_data_points
    
    try:
        # Count rows in all OHLCV files
        count = 0
        for file in glob.glob("data/processed/DATA_*.csv"):
            try:
                df = pd.read_csv(file)
                count += len(df)
            except Exception as e:
                logging.error(f"Error counting rows in {file}: {e}")
        
        total_data_points = count
        logging.info(f"Total data points available for training: {count}")
        return count
    except Exception as e:
        logging.error(f"Error counting total data points: {e}")
        return 0

def should_start_training():
    """Determine if we should start a training run based on available data and time since last training."""
    global last_training_time, data_collection_start_time, total_data_points
    
    # Check if any token has complete data
    token_has_data = False
    
    # Get all token feature files
    feature_files = glob.glob("data/processed/*_features.csv")
    ohlcv_files = glob.glob("data/processed/DATA_*.csv")
    
    if feature_files or ohlcv_files:
        # If we have any processed data files, we can start training
        token_has_data = True
        
        # Count data points for logging
        current_data_points = count_total_data_points()
        logging.info(f"Found {len(feature_files)} feature files and {len(ohlcv_files)} OHLCV files with {current_data_points} total data points")
    
    # First time training check
    if last_training_time is None and token_has_data:
        logging.info(f"Starting first training run with processed data available")
        return True
    
    # Retraining based on time interval
    if last_training_time is not None:
        hours_since_last_training = (datetime.now() - last_training_time).total_seconds() / 3600
        if hours_since_last_training >= RETRAINING_INTERVAL_HOURS:
            logging.info(f"Starting retraining after {hours_since_last_training:.1f} hours")
            return True
    
    return False

def training_worker():
    """Worker thread to run the training pipeline."""
    global is_training_running, last_training_time
    
    with training_lock:
        is_training_running = True
    
    try:
        # Run the training scripts
        training_scripts = get_training_scripts()
        for desc, cmd in training_scripts:
            logging.info(f"[Training] Running: {desc}")
            try:
                subprocess.run(cmd, check=True)
                logging.info(f"[Training] Completed: {desc}")
            except subprocess.CalledProcessError as e:
                logging.error(f"[Training] Error running {desc}: {e}")
                break
        
        # Update last training time
        with training_lock:
            last_training_time = datetime.now()
            is_training_running = False
        
        logging.info(f"[Training] Pipeline completed successfully at {last_training_time}")
    except Exception as e:
        logging.error(f"[Training] Error in training thread: {e}")
        with training_lock:
            is_training_running = False

def monitor_and_train():
    """Monitor data collection and trigger training when appropriate."""
    global training_thread, is_training_running, data_collection_start_time
    
    if data_collection_start_time is None:
        data_collection_start_time = datetime.now()
    
    while True:
        # Check if we should start training
        if PROGRESSIVE_TRAINING and not is_training_running and should_start_training():
            # Start training in a separate thread
            with training_lock:
                training_thread = threading.Thread(target=training_worker)
                training_thread.daemon = True
                training_thread.start()
        
        # Sleep for a while before checking again
        time.sleep(300)  # Check every 5 minutes

def run_pipeline_steps(initial_run=False, days_override=None):
    """Execute each script in the pipeline in order."""
    global data_collection_start_time
    
    # Start the data collection monitor thread if progressive training is enabled
    if PROGRESSIVE_TRAINING:
        data_collection_start_time = datetime.now()
        monitor_thread = threading.Thread(target=monitor_and_train)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    # Get the appropriate scripts to run
    scripts = get_scripts(initial_run, days_override)
    
    for desc, cmd in scripts:
        logging.info(f"Running: {desc}")
        try:
            subprocess.run(cmd, check=True)
            logging.info(f"Completed: {desc}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running {desc}: {e}")
            return False
    
    logging.info("All pipeline steps completed successfully")
    return True

def find_ohlcv_file(token_symbol):
    """
    Find the OHLCV file for a token by trying different possible formats.
    
    Args:
        token_symbol: Symbol of the token (e.g., "SOL")
        
    Returns:
        Path to the OHLCV file if found, None otherwise
    """
    logging.info(f"Looking for OHLCV file for {token_symbol}")
    
    # Check for the standardized format first
    standard_file = f"data/processed/{token_symbol.lower()}_ohlcv.csv"
    if os.path.exists(standard_file):
        logging.info(f"Found standard file: {standard_file}")
        return standard_file
    
    # Check for data/raw format
    raw_file = f"data/raw/{token_symbol.lower()}_usd_ohlcv.csv"
    if os.path.exists(raw_file):
        logging.info(f"Found raw file: {raw_file}")
        return raw_file
    
    # Special case for SOL (wrapped SOL has a specific address)
    if token_symbol.upper() == "SOL":
        sol_address = "So11111111111111111111111111111111111111112"
        sol_files = glob.glob(f"data/processed/DATA_*{sol_address}*.csv")
        if sol_files:
            logging.info(f"Found SOL file through address matching: {sol_files[0]}")
            return sol_files[0]
        
        # Also check the root directory
        sol_root_files = glob.glob(f"DATA_*SOL*.csv")
        if sol_root_files:
            logging.info(f"Found SOL file in root: {sol_root_files[0]}")
            return sol_root_files[0]
    
    # Convert to lowercase for case-insensitive matching
    token_symbol_lower = token_symbol.lower()
    
    # Try to find the file based on pattern
    data_processed_dir = "data/processed"
    if os.path.exists(data_processed_dir):
        # Get all the DATA_*.csv files in the processed directory
        all_data_files = glob.glob(f"{data_processed_dir}/DATA_*.csv")
        logging.info(f"Found {len(all_data_files)} DATA_*.csv files in {data_processed_dir}")
        
        # Find token address from pool_addresses.json
        token_address = None
        try:
            with open("pool_addresses.json", "r") as f:
                pool_addresses = json.load(f)
            
            for market_name, info in pool_addresses.items():
                if info.get("base", {}).get("symbol", "").lower() == token_symbol_lower:
                    token_address = info.get("base", {}).get("address")
                    if token_address:
                        logging.info(f"Found token address for {token_symbol}: {token_address}")
                        break
        except Exception as e:
            logging.error(f"Error finding token address for {token_symbol}: {e}")
        
        # If we found the token address, look for files with that address
        if token_address:
            for file_path in all_data_files:
                if token_address in file_path:
                    logging.info(f"Found file with token address {token_address}: {file_path}")
                    return file_path
        
        # If not found by address, try to find by symbol in filename
        for file_path in all_data_files:
            file_name = os.path.basename(file_path).lower()
            if token_symbol_lower in file_name:
                logging.info(f"Found file with symbol {token_symbol} in name: {file_path}")
                return file_path
    
    # Check in the root directory
    root_ohlcv_files = glob.glob(f"DATA_*{token_symbol}*.csv")
    logging.info(f"Looking for DATA_*{token_symbol}*.csv in root: found {len(root_ohlcv_files)} files")
    if root_ohlcv_files:
        logging.info(f"Found file in root directory: {root_ohlcv_files[0]}")
        return root_ohlcv_files[0]
    
    # List all files in data/processed to help debugging
    logging.info("Available DATA_ files in data/processed:")
    for f in glob.glob(f"{data_processed_dir}/DATA_*.csv"):
        logging.info(f"  {os.path.basename(f)}")
    
    # List all files in root with DATA_ prefix
    logging.info("Available DATA_ files in root:")
    for f in glob.glob("DATA_*.csv"):
        logging.info(f"  {f}")
    
    # Not found in any format
    logging.warning(f"No OHLCV file found for {token_symbol} after trying all patterns")
    return None

def load_sentiment_data(token_symbol):
    """
    Load sentiment data for a token.
    
    Args:
        token_symbol: Symbol of the token (e.g., "SOL")
        
    Returns:
        DataFrame with date and sentiment_score columns, or None if not found
    """
    # Convert token symbol to lowercase for consistency
    token_symbol = token_symbol.lower()
    
    # Path to sentiment data file
    sentiment_file = f"data/sentiment/{token_symbol}_sentiment.csv"
    
    if not os.path.exists(sentiment_file):
        logging.warning(f"No sentiment data found for {token_symbol}")
        return None
    
    try:
        # Load sentiment data
        sentiment_df = pd.read_csv(sentiment_file)
        
        # Ensure date column is datetime
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        logging.info(f"Loaded sentiment data for {token_symbol}")
        return sentiment_df
    
    except Exception as e:
        logging.error(f"Error loading sentiment data for {token_symbol}: {e}")
        return None

def load_market_sentiment():
    """
    Load market-wide sentiment data.
    
    Returns:
        DataFrame with date and market_sentiment columns, or None if not found
    """
    # Path to market sentiment data file
    market_sentiment_file = "data/sentiment/market_sentiment.csv"
    
    if not os.path.exists(market_sentiment_file):
        logging.warning("No market sentiment data found")
        return None
    
    try:
        # Load market sentiment data
        market_df = pd.read_csv(market_sentiment_file)
        
        # Ensure date column is datetime
        market_df['date'] = pd.to_datetime(market_df['date'])
        
        logging.info("Loaded market sentiment data")
        return market_df
    
    except Exception as e:
        logging.error(f"Error loading market sentiment data: {e}")
        return None

def prepare_features(ohlcv_data, token_symbol):
    """
    Prepare features for the model, including sentiment data if available.
    
    Args:
        ohlcv_data: DataFrame with OHLCV data
        token_symbol: Symbol of the token
        
    Returns:
        DataFrame with features
    """
    # Convert OHLCV data to features (column standardization now happens in compute_technical_indicators)
    features = compute_technical_indicators(ohlcv_data)
    
    # Load sentiment data
    sentiment_df = load_sentiment_data(token_symbol)
    market_sentiment_df = load_market_sentiment()
    
    # Add sentiment features if available
    if sentiment_df is not None:
        # Normalize timezone info to prevent merge errors
        features['date'] = pd.to_datetime(features['date']).dt.tz_localize(None)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.tz_localize(None)
        
        # Merge on date
        features = pd.merge(features, sentiment_df, on='date', how='left')
        
        # Fill missing sentiment values with NaN (do not use 0)
        features['sentiment_score'] = features['sentiment_score']
        
        # Add sentiment change features
        features['sentiment_change_1d'] = features['sentiment_score'].diff(1)
        features['sentiment_change_3d'] = features['sentiment_score'].diff(3)
        features['sentiment_change_7d'] = features['sentiment_score'].diff(7)
    # If no sentiment data, do not add synthetic columns
    
    # Add market sentiment features if available
    if market_sentiment_df is not None:
        # Normalize timezone info to prevent merge errors
        if 'date' in features.columns and 'date' in market_sentiment_df.columns:
            features['date'] = pd.to_datetime(features['date']).dt.tz_localize(None)
            market_sentiment_df['date'] = pd.to_datetime(market_sentiment_df['date']).dt.tz_localize(None)
        
        # Merge on date
        features = pd.merge(features, market_sentiment_df, on='date', how='left')
        
        # Fill missing sentiment values with NaN (do not use 0)
        features['market_sentiment'] = features['market_sentiment']
        
        # Add market sentiment change features
        features['market_sentiment_change_1d'] = features['market_sentiment'].diff(1)
        features['market_sentiment_change_3d'] = features['market_sentiment'].diff(3)
        features['market_sentiment_change_7d'] = features['market_sentiment'].diff(7)
    # If no market sentiment data, do not add synthetic columns
    
    return features

def compute_technical_indicators(ohlcv_data):
    """
    Compute technical indicators from OHLCV data.
    
    Args:
        ohlcv_data: DataFrame with OHLCV data
        
    Returns:
        DataFrame with technical indicators
    """
    df = ohlcv_data.copy()
    
    # Handle column name differences - rename 'datetime' to 'date' if it exists
    if 'datetime' in df.columns and 'date' not in df.columns:
        df = df.rename(columns={'datetime': 'date'})
        logging.info("Renamed 'datetime' column to 'date'")
    
    # Ensure the dataframe has the required columns
    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in OHLCV data")
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Moving averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential moving averages
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Relative price changes
    df['price_change_1d'] = df['close'].pct_change(1)
    df['price_change_3d'] = df['close'].pct_change(3)
    df['price_change_5d'] = df['close'].pct_change(5)
    df['price_change_10d'] = df['close'].pct_change(10)
    
    # Volatility
    df['volatility_5d'] = df['close'].rolling(window=5).std() / df['close'].rolling(window=5).mean()
    df['volatility_10d'] = df['close'].rolling(window=10).std() / df['close'].rolling(window=10).mean()
    
    # Volume changes
    df['volume_change_1d'] = df['volume'].pct_change(1)
    df['volume_change_3d'] = df['volume'].pct_change(3)
    df['volume_change_5d'] = df['volume'].pct_change(5)
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Rate of Change (ROC)
    df['roc_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
    df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
    
    # Average True Range (ATR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = df['true_range'].rolling(window=14).mean()
    
    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df

def process_tokens():
    """Process individual token data using the already fetched OHLCV data."""
    # Create data directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Get all tokens from pool_addresses.json
    try:
        with open("pool_addresses.json", "r") as f:
            pool_addresses = json.load(f)
        
        tokens = []
        for market_name, info in pool_addresses.items():
            base_symbol = info.get("base", {}).get("symbol")
            if base_symbol:
                tokens.append(base_symbol)
        
        tokens = list(set(tokens))  # Remove duplicates
    except Exception as e:
        logging.error(f"Error loading pool_addresses.json: {e}")
        return False
    
    logging.info(f"Processing {len(tokens)} tokens")
    
    # Process each token
    for token_symbol in tokens:
        logging.info(f"Processing {token_symbol}")
        
        # 1. Find OHLCV data file
        ohlcv_file = find_ohlcv_file(token_symbol)
        if not ohlcv_file:
            logging.warning(f"No OHLCV data found for {token_symbol}. Skipping.")
            continue
        
        try:
            ohlcv_data = pd.read_csv(ohlcv_file)
            logging.info(f"Loaded OHLCV data for {token_symbol} from {ohlcv_file}")
        except Exception as e:
            logging.error(f"Error loading OHLCV data for {token_symbol}: {e}")
            continue
        
        # 2. Prepare features including sentiment
        try:
            features = prepare_features(ohlcv_data, token_symbol)
            logging.info(f"Prepared features for {token_symbol}")
        except Exception as e:
            logging.error(f"Error preparing features for {token_symbol}: {e}")
            continue
        
        # 3. Save prepared features
        features_file = f"data/processed/{token_symbol.lower()}_features.csv"
        features.to_csv(features_file, index=False)
        logging.info(f"Saved features to {features_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run the full ML pipeline.")
    parser.add_argument("--token", help="Token symbol to process")
    parser.add_argument("--initial-run", action="store_true", help="Perform an initial run with 30 days of data")
    parser.add_argument("--days", type=int, help="Override the number of days to fetch")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip running the scripts pipeline and only process tokens")
    parser.add_argument("--progressive", action="store_true", help="Enable progressive training during data collection")
    parser.add_argument("--min-data", type=int, default=10000, 
                      help="Minimum data points needed before first training (default: 10000)")
    parser.add_argument("--retrain-hours", type=int, default=6,
                      help="Hours between retraining runs (default: 6)")
    args = parser.parse_args()
    
    # Update progressive training settings if provided
    global PROGRESSIVE_TRAINING
    
    if args.progressive:
        PROGRESSIVE_TRAINING = True
        logging.info("Progressive training enabled")
    
    if args.min_data:
        global MIN_DATA_POINTS_FOR_TRAINING
        MIN_DATA_POINTS_FOR_TRAINING = args.min_data
        logging.info(f"Minimum data points for training set to {MIN_DATA_POINTS_FOR_TRAINING}")
    
    if args.retrain_hours:
        global RETRAINING_INTERVAL_HOURS
        RETRAINING_INTERVAL_HOURS = args.retrain_hours
        logging.info(f"Retraining interval set to {RETRAINING_INTERVAL_HOURS} hours")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    success = True
    
    # Run the pipeline scripts if not skipped
    if not args.skip_pipeline:
        logging.info("Starting ML pipeline execution")
        success = run_pipeline_steps(args.initial_run, args.days)
        
        if not success:
            logging.error("Pipeline execution failed")
            return 1
    
    # Process specific token if provided
    if args.token:
        logging.info(f"Processing single token: {args.token}")
        # Set environment variable for other scripts
        os.environ["TOKEN"] = args.token
    
    # Process token data
    if success:
        process_tokens()
    
    # Generate trading signals
    generate_signals()
    
    logging.info("Pipeline completed successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 