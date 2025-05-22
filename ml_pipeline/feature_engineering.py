import pandas as pd
import numpy as np
import os
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
import logging
# No longer need argparse here if called as a library, 
# but keep if __name__ == "__main__" needs it (it doesn't currently)

# Setup logging (this will be configured by the API server if called from there,
# or use this default if run standalone)
# To avoid duplicate handlers if imported, standard practice is:
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PRICE_CHANGE_THRESHOLD = 0.002  # 0.2%
LOOKAHEAD_PERIODS = 5

def generate_classification_labels(group_df):
    """Generates classification labels for a single token's data group."""
    group_df = group_df.copy()
    group_df['future_price'] = group_df['price'].shift(-LOOKAHEAD_PERIODS)
    
    conditions = [
        (group_df['future_price'] > group_df['price'] * (1 + PRICE_CHANGE_THRESHOLD)),
        (group_df['future_price'] < group_df['price'] * (1 - PRICE_CHANGE_THRESHOLD))
    ]
    choices = [
        0,  # Buy
        1   # Sell
    ]
    group_df['action_label'] = np.select(conditions, choices, default=2) # Hold
    
    # Drop rows where future_price is NaN (i.e., last LOOKAHEAD_PERIODS rows)
    group_df.dropna(subset=['future_price'], inplace=True)
    group_df.drop(columns=['future_price'], inplace=True)
    return group_df

def engineer_features(input_csv_path: str, output_csv_path: str) -> dict:
    """
    Loads data from input_csv_path, engineers features, generates classification labels, 
    and saves to output_csv_path.
    Returns a dictionary with status and metadata.
    """
    logger.info(f"Starting feature engineering for {input_csv_path}")
    result = {
        "status": "failed",
        "input_path": input_csv_path,
        "output_path": output_csv_path,
        "message": "",
        "initial_shape": None,
        "final_shape": None,
        "num_features_generated": None # Will be number of new columns added
    }
    
    if not os.path.exists(input_csv_path):
        logger.error(f"Input file not found: {input_csv_path}")
        result["message"] = f"Input file not found: {input_csv_path}"
        return result

    try:
        df = pd.read_csv(input_csv_path, parse_dates=['date'])
        result["initial_shape"] = df.shape
        logger.info(f"Initial DataFrame shape from {input_csv_path}: {df.shape}")

        if 'token_address' not in df.columns:
            logger.error(f"Critical error: 'token_address' column not found in {input_csv_path}.")
            result["message"] = "'token_address' column not found."
            return result
        if 'price' not in df.columns:
            logger.error(f"Critical error: 'price' column not found in {input_csv_path}.")
            result["message"] = "'price' column not found."
            return result

        original_columns = set(df.columns)
        df = df.sort_values(['token_address', 'date']).reset_index(drop=True)
        # logger.info(f"DataFrame shape after sorting: {df.shape}")

        # unique_tokens = df['token_address'].unique()
        # logger.info(f"Found {len(unique_tokens)} unique token_addresses in {input_csv_path}")

        features_list = []

        for token_address, group in df.groupby('token_address'):
            group = group.copy()

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

            # Generate classification labels
            group = generate_classification_labels(group)

            if group.empty:
                logger.warning(f"Token {token_address} has insufficient data after label generation. Skipping.")
                continue
            
            features_list.append(group)

        if features_list:
            features_df = pd.concat(features_list)
            result["final_shape"] = features_df.shape
            new_columns = set(features_df.columns) - original_columns
            result["num_features_generated"] = len(new_columns)
            logger.info(f"Shape of concatenated features_df for {input_csv_path}: {features_df.shape}")
            logger.info(f"Generated {len(new_columns)} new feature columns.")
        else:
            features_df = df.copy() # Save original df if no features generated, or an empty one
            logger.warning(f"No features were generated for {input_csv_path} (features_list is empty). Saving original data to {output_csv_path}.")
            result["message"] = "No new features generated, features_list was empty."
            result["num_features_generated"] = 0
            result["final_shape"] = features_df.shape # Shape of what's being saved


        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        features_df.to_csv(output_csv_path, index=False)
        logger.info(f"Feature engineering complete for {input_csv_path}. Saved to {output_csv_path}")
        result["status"] = "success"
        result["message"] = f"Successfully engineered features and saved to {output_csv_path}"
        return result
        
    except Exception as e:
        import traceback
        logger.error(f"An error occurred during feature engineering for {input_csv_path}: {e}")
        logger.error(traceback.format_exc())
        result["message"] = f"Error during feature engineering: {str(e)}"
        return result


# This main block allows the script to be run standalone for convenience
# (e.g., for initial data setup or testing feature engineering directly)
if __name__ == "__main__":
    logger.info("Running feature_engineering.py standalone.")
    
    # Define input and output paths for train and test data
    # These paths assume the script is run from the project root.
    # Ensure these input files exist.
    train_input = 'data/processed/train_data.csv'
    train_output = 'data/features_train_ohlcv.csv'
    
    test_input = 'data/processed/test_data.csv'
    test_output = 'data/features_test_ohlcv.csv'

    logger.info(f"Standalone run: Processing training data: {train_input} -> {train_output}")
    train_result = engineer_features(train_input, train_output)
    logger.info(f"Training data processing result: {train_result}")
    
    logger.info(f"Standalone run: Processing test data: {test_input} -> {test_output}")
    test_result = engineer_features(test_input, test_output)
    logger.info(f"Test data processing result: {test_result}")

    # Deprecation warning (can be removed if data/features_ohlcv.csv is no longer relevant)
    old_combined_features_path = 'data/features_ohlcv.csv'
    if os.path.exists(old_combined_features_path):
        logger.warning(f"The file {old_combined_features_path} is deprecated. \
                         Training features are now in {train_output} and \
                         test features are in {test_output}.") 