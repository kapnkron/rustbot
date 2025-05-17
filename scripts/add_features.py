import pandas as pd
import numpy as np
import os
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
import logging
import argparse # For command-line arguments

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def engineer_features(input_csv_path: str, output_csv_path: str):
    """
    Loads data from input_csv_path, engineers features, and saves to output_csv_path.
    """
    logging.info(f"Starting feature engineering for {input_csv_path}")
    
    if not os.path.exists(input_csv_path):
        logging.error(f"Input file not found: {input_csv_path}")
        return

    df = pd.read_csv(input_csv_path, parse_dates=['date'])
    logging.info(f"Initial DataFrame shape from {input_csv_path}: {df.shape}")

    if 'token_address' not in df.columns:
        logging.error(f"Critical error: 'token_address' column not found in {input_csv_path}.")
        return
    if 'price' not in df.columns:
        logging.error(f"Critical error: 'price' column not found in {input_csv_path}. This is needed for most features.")
        return

    df = df.sort_values(['token_address', 'date']).reset_index(drop=True)
    logging.info(f"DataFrame shape after sorting: {df.shape}")

    unique_tokens = df['token_address'].unique()
    logging.info(f"Found {len(unique_tokens)} unique token_addresses in {input_csv_path}")

    features_list = []

    for token_address, group in df.groupby('token_address'):
        # logging.info(f"Processing token: {token_address} from {input_csv_path}")
        # logging.info(f"  Shape of group for {token_address} BEFORE feature calculation: {group.shape}")
        group = group.copy()

        # Calculate features
        group['return'] = group['price'].pct_change()
        group['log_return'] = np.log(group['price'] / group['price'].shift(1))
        
        group['sma_10'] = SMAIndicator(group['price'], window=10, fillna=True).sma_indicator()
        group['ema_10'] = EMAIndicator(group['price'], window=10, fillna=True).ema_indicator()
        group['sma_50'] = SMAIndicator(group['price'], window=50, fillna=True).sma_indicator()
        group['ema_50'] = EMAIndicator(group['price'], window=50, fillna=True).ema_indicator()
        group['rsi_14'] = RSIIndicator(group['price'], window=14, fillna=True).rsi()
        
        macd_indicator = MACD(group['price'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        group['macd'] = macd_indicator.macd()
        group['macd_signal'] = macd_indicator.macd_signal()
        group['macd_diff'] = macd_indicator.macd_diff()

        bb_indicator = BollingerBands(group['price'], window=20, window_dev=2, fillna=True)
        group['bb_high'] = bb_indicator.bollinger_hband()
        group['bb_low'] = bb_indicator.bollinger_lband()
        group['bb_width'] = bb_indicator.bollinger_wband()
        
        group['volatility_10'] = group['log_return'].rolling(window=10).std() * np.sqrt(252) 
        group['volatility_50'] = group['log_return'].rolling(window=50).std() * np.sqrt(252)

        group['momentum_1'] = group['price'].diff(1)
        group['momentum_5'] = group['price'].diff(5)
        group['momentum_10'] = group['price'].diff(10)

        group['price_lag_1'] = group['price'].shift(1)
        group['price_lag_2'] = group['price'].shift(2)
        group['price_lag_3'] = group['price'].shift(3)
        
        if 'volume' in group.columns and not group['volume'].isnull().all():
            group['volume_sma_10'] = SMAIndicator(group['volume'], window=10, fillna=True).sma_indicator()
            group['volume_sma_50'] = SMAIndicator(group['volume'], window=50, fillna=True).sma_indicator()
        else:
            group['volume_sma_10'] = np.nan 
            group['volume_sma_50'] = np.nan

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