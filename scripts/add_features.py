import pandas as pd
import numpy as np
import os
from ta.momentum import RSIIndicator, AwesomeOscillatorIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import logging
import argparse # For command-line arguments
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_sentiment_data(sentiment_dir='data/sentiment'):
    """
    Load sentiment data from the sentiment directory.
    
    Returns a dictionary mapping token addresses to sentiment data.
    If no sentiment data is available, returns an empty dictionary.
    """
    sentiment_data = {}
    sentiment_dir = Path(sentiment_dir)
    
    if not sentiment_dir.exists():
        logging.warning(f"Sentiment directory {sentiment_dir} does not exist. No sentiment features will be added.")
        return sentiment_data
    
    # Load sentiment files - each file should be named with the token address and contain time series sentiment data
    for sentiment_file in sentiment_dir.glob('*.csv'):
        try:
            token_address = sentiment_file.stem  # Get filename without extension
            token_sentiment = pd.read_csv(sentiment_file)
            
            # Ensure sentiment data has a date column
            if 'date' not in token_sentiment.columns:
                logging.warning(f"Sentiment file {sentiment_file} has no date column. Skipping.")
                continue
                
            # Convert date to datetime if it's not already
            token_sentiment['date'] = pd.to_datetime(token_sentiment['date'])
            
            # Add to dictionary
            sentiment_data[token_address] = token_sentiment
            logging.info(f"Loaded sentiment data for {token_address}: {len(token_sentiment)} records")
        except Exception as e:
            logging.error(f"Error loading sentiment file {sentiment_file}: {e}")
    
    return sentiment_data

def engineer_features(input_csv_path: str, output_csv_path: str):
    """
    Loads data from input_csv_path, engineers features, and saves to output_csv_path.
    """
    logging.info(f"Starting feature engineering for {input_csv_path}")
    
    if not os.path.exists(input_csv_path):
        logging.error(f"Input file not found: {input_csv_path}")
        return

    df = pd.read_csv(input_csv_path, parse_dates=['datetime'])
    logging.info(f"Initial DataFrame shape from {input_csv_path}: {df.shape}")

    if 'token_address' not in df.columns:
        logging.error(f"Critical error: 'token_address' column not found in {input_csv_path}.")
        return
    if 'price' not in df.columns:
        logging.error(f"Critical error: 'price' column not found in {input_csv_path}. This is needed for most features.")
        return

    df = df.sort_values(['token_address', 'datetime']).reset_index(drop=True)
    logging.info(f"DataFrame shape after sorting: {df.shape}")

    unique_tokens = df['token_address'].unique()
    logging.info(f"Found {len(unique_tokens)} unique token_addresses in {input_csv_path}")

    # Load the pool_addresses.json to get token categories
    pool_map = {}
    try:
        with open('pool_addresses.json', 'r') as f:
            pool_map = json.load(f)
        logging.info(f"Loaded pool_addresses.json with {len(pool_map)} entries")
    except FileNotFoundError:
        logging.warning("pool_addresses.json not found. Token categories will not be available.")
    
    # Build a map of token address to category
    token_categories = {}
    for market_name, info in pool_map.items():
        base_address = info.get('base', {}).get('address')
        if base_address:
            category = info.get('category', 'general_token')
            token_categories[base_address] = category
    
    # Load sentiment data
    sentiment_data = load_sentiment_data()
    logging.info(f"Loaded sentiment data for {len(sentiment_data)} tokens")

    features_list = []

    for token_address, group in df.groupby('token_address'):
        # logging.info(f"Processing token: {token_address} from {input_csv_path}")
        # logging.info(f"  Shape of group for {token_address} BEFORE feature calculation: {group.shape}")
        group = group.copy()

        # --- Sanity Check ---
        # Ensure we have enough data to calculate features. Min window is for ROC(1)
        if len(group) < 2:
            # logging.warning(f"Skipping token {token_address} due to insufficient data (rows: {len(group)})")
            continue

        # --- Add token category ---
        group['token_category'] = token_categories.get(token_address, 'general_token')

        # --- Momentum Features ---
        group['return'] = group['price'].pct_change()
        group['log_return'] = np.log(group['price'] / group['price'].shift(1))
        group['momentum_1'] = group['price'].diff(1)
        group['momentum_5'] = group['price'].diff(5)
        group['momentum_10'] = group['price'].diff(10)
        
        # RSI
        group['rsi_5'] = RSIIndicator(group['price'], window=5, fillna=True).rsi()
        group['rsi_10'] = RSIIndicator(group['price'], window=10, fillna=True).rsi()
        group['rsi_14'] = RSIIndicator(group['price'], window=14, fillna=True).rsi()
        
        # MACD
        macd_indicator = MACD(group['price'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        group['macd'] = macd_indicator.macd()
        group['macd_signal'] = macd_indicator.macd_signal()
        group['macd_diff'] = macd_indicator.macd_diff()

        # Rate of Change (ROC)
        group['roc_1'] = ROCIndicator(group['price'], window=1, fillna=True).roc()
        group['roc_3'] = ROCIndicator(group['price'], window=3, fillna=True).roc()
        group['roc_5'] = ROCIndicator(group['price'], window=5, fillna=True).roc()

        # Awesome Oscillator
        group['awesome_oscillator'] = AwesomeOscillatorIndicator(group['high'], group['low'], window1=5, window2=34, fillna=True).awesome_oscillator()

        # --- Trend Features ---
        group['sma_10'] = SMAIndicator(group['price'], window=10, fillna=True).sma_indicator()
        group['ema_10'] = EMAIndicator(group['price'], window=10, fillna=True).ema_indicator()
        group['sma_50'] = SMAIndicator(group['price'], window=50, fillna=True).sma_indicator()
        group['ema_50'] = EMAIndicator(group['price'], window=50, fillna=True).ema_indicator()
        
        # --- Volatility Features ---
        # Bollinger Bands
        bb_indicator = BollingerBands(group['price'], window=20, window_dev=2, fillna=True)
        group['bb_high'] = bb_indicator.bollinger_hband()
        group['bb_low'] = bb_indicator.bollinger_lband()
        group['bb_width'] = bb_indicator.bollinger_wband()
        
        # Historical Volatility
        group['volatility_10'] = group['log_return'].rolling(window=10).std() * np.sqrt(252) 
        group['volatility_50'] = group['log_return'].rolling(window=50).std() * np.sqrt(252)

        # Average True Range (ATR)
        group['atr_14'] = AverageTrueRange(high=group['high'], low=group['low'], close=group['price'], window=14, fillna=True).average_true_range()

        # --- Price Transformation ---
        group['price_lag_1'] = group['price'].shift(1)
        group['price_lag_2'] = group['price'].shift(2)
        group['price_lag_3'] = group['price'].shift(3)
        
        # --- Volume Features ---
        if 'volume' in group.columns and not group['volume'].isnull().all():
            group['volume_sma_10'] = SMAIndicator(group['volume'], window=10, fillna=True).sma_indicator()
            group['volume_sma_50'] = SMAIndicator(group['volume'], window=50, fillna=True).sma_indicator()
            group['obv'] = OnBalanceVolumeIndicator(close=group['price'], volume=group['volume'], fillna=True).on_balance_volume()
        else:
            group['volume_sma_10'] = np.nan 
            group['volume_sma_50'] = np.nan
            group['obv'] = np.nan
        
        # --- Sentiment Features ---
        # Add sentiment features if available for this token
        if token_address in sentiment_data:
            token_sentiment = sentiment_data[token_address]
            
            # Merge sentiment data with price data based on date
            merged_sentiment = pd.merge_asof(
                group.sort_values('date'), 
                token_sentiment.sort_values('date'),
                on='date', 
                direction='backward'  # Use the most recent sentiment before the price data
            )
            
            # If merge successful, add sentiment features
            if 'sentiment_score' in merged_sentiment.columns:
                group['sentiment_score'] = merged_sentiment['sentiment_score']
                
                # Add derived sentiment features
                group['sentiment_sma_5'] = group['sentiment_score'].rolling(window=5).mean()
                group['sentiment_sma_10'] = group['sentiment_score'].rolling(window=10).mean()
                group['sentiment_momentum'] = group['sentiment_score'].diff(1)
                
                # Interaction between price and sentiment
                group['price_sentiment_ratio'] = group['price'] / (group['sentiment_score'] + 1e-5)  # Avoid division by zero
                
                logging.info(f"Added sentiment features for {token_address}")
            else:
                # Fill with NaN if no sentiment data available
                group['sentiment_score'] = np.nan
                group['sentiment_sma_5'] = np.nan
                group['sentiment_sma_10'] = np.nan
                group['sentiment_momentum'] = np.nan
                group['price_sentiment_ratio'] = np.nan
        else:
            # Fill with NaN if no sentiment data available
            group['sentiment_score'] = np.nan
            group['sentiment_sma_5'] = np.nan
            group['sentiment_sma_10'] = np.nan
            group['sentiment_momentum'] = np.nan
            group['price_sentiment_ratio'] = np.nan
        
        # logging.info(f"  Shape of group for {token_address} AFTER feature calculation: {group.shape}")
        features_list.append(group)

    if features_list:
        features_df = pd.concat(features_list)
        logging.info(f"Shape of concatenated features_df for {input_csv_path}: {features_df.shape}")
    else:
        features_df = pd.DataFrame() 
        logging.warning(f"No features were generated for {input_csv_path} (features_list is empty). Saving an empty CSV to {output_csv_path}.")

    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    features_df.to_csv(output_csv_path, index=False)
    logging.info(f"Feature engineering complete for {input_csv_path}. Saved to {output_csv_path}")

if __name__ == "__main__":
    # Define input and output paths for train and test data
    train_input_path = 'data/processed/train_data.csv'
    train_output_path = 'data/features_train_ohlcv.csv' # New name for training features
    
    test_input_path = 'data/processed/test_data.csv'
    test_output_path = 'data/features_test_ohlcv.csv'   # New file for test features

    # Process training data
    engineer_features(train_input_path, train_output_path)
    
    # Process test data
    engineer_features(test_input_path, test_output_path)

    # Deprecation warning for old combined file if it exists
    old_combined_features_path = 'data/features_ohlcv.csv'
    if os.path.exists(old_combined_features_path):
        logging.warning(f"The file {old_combined_features_path} is deprecated. \
                         Training features are now in {train_output_path} and \
                         test features are in {test_output_path}.") 