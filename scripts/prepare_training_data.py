import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)

class DataPreparator:
    def __init__(self):
        self.raw_data_dir = 'data/raw'
        self.processed_data_dir = 'data/processed'
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def load_ohlcv_data(self) -> Dict[str, pd.DataFrame]:
        """Load all OHLCV data files and return a dictionary of DataFrames."""
        ohlcv_files = glob.glob(os.path.join(self.raw_data_dir, '*_usd_ohlcv.csv'))
        data_dict = {}
        
        for file in ohlcv_files:
            try:
                # Extract token address from filename
                token_address = os.path.basename(file).split('_')[0]
                df = pd.read_csv(file, index_col='date', parse_dates=True)
                # Rename 'close' to 'price' for compatibility with the rest of the script
                df = df.rename(columns={'close': 'price'})
                df['token_address'] = token_address  # Add token_address column
                data_dict[token_address] = df
                logging.info(f"Loaded OHLCV data for {token_address}")
            except Exception as e:
                logging.error(f"Error loading {file}: {str(e)}")
        
        return data_dict

    def merge_ohlcv_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all OHLCV data into a single DataFrame with multi-index."""
        merged_data = []
        
        for token_address, df in data_dict.items():
            # Add token address as a column
            df['token_address'] = token_address
            merged_data.append(df)
        
        if not merged_data:
            raise ValueError("No OHLCV data found to merge")
        
        # Concatenate all DataFrames
        merged_df = pd.concat(merged_data, axis=0)
        
        # Sort by timestamp and token
        merged_df.sort_index(inplace=True)
        
        return merged_df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data."""
        # Group by token to calculate indicators for each token separately
        grouped = df.groupby('token_address')
        
        # Calculate returns
        df['returns'] = grouped['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / grouped['price'].shift(1))
        
        # Calculate moving averages
        df['sma_10'] = grouped['price'].transform(lambda x: x.rolling(window=10).mean())
        df['sma_50'] = grouped['price'].transform(lambda x: x.rolling(window=50).mean())
        df['ema_10'] = grouped['price'].transform(lambda x: x.ewm(span=10).mean())
        df['ema_50'] = grouped['price'].transform(lambda x: x.ewm(span=50).mean())
        
        # Calculate volatility
        df['volatility_10'] = grouped['returns'].transform(lambda x: x.rolling(window=10).std())
        df['volatility_50'] = grouped['returns'].transform(lambda x: x.rolling(window=50).std())
        
        # Calculate RSI
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['rsi_14'] = grouped['price'].transform(calculate_rsi)
        
        # Calculate MACD and signal line for each group
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_histogram'] = np.nan
        for token, group in grouped:
            exp1 = group['price'].ewm(span=12, adjust=False).mean()
            exp2 = group['price'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - signal_line
            df.loc[group.index, 'macd'] = macd
            df.loc[group.index, 'macd_signal'] = signal_line
            df.loc[group.index, 'macd_histogram'] = macd_hist
        
        # Add price momentum
        df['momentum_1'] = grouped['price'].transform(lambda x: x.pct_change(1))
        df['momentum_5'] = grouped['price'].transform(lambda x: x.pct_change(5))
        df['momentum_10'] = grouped['price'].transform(lambda x: x.pct_change(10))
        
        return df

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare the final training and testing datasets."""
        # Load and merge data
        ohlcv_data = self.load_ohlcv_data()
        merged_data = self.merge_ohlcv_data(ohlcv_data)
        
        # Add technical indicators
        processed_data = self.add_technical_indicators(merged_data)
        
        # Remove rows with NaN values
        processed_data = processed_data.dropna()
        
        # Save the processed data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        processed_path = os.path.join(self.processed_data_dir, f'processed_data_{timestamp}.csv')
        processed_data.to_csv(processed_path)
        logging.info(f"Saved processed data to {processed_path}")
        
        # Split into training and testing sets (80/20 split)
        train_size = int(len(processed_data) * 0.8)
        train_data = processed_data.iloc[:train_size]
        test_data = processed_data.iloc[train_size:]
        
        # Save training and testing sets
        train_path = os.path.join(self.processed_data_dir, 'train_data.csv')
        test_path = os.path.join(self.processed_data_dir, 'test_data.csv')
        train_data.to_csv(train_path)
        test_data.to_csv(test_path)
        logging.info(f"Saved training data to {train_path}")
        logging.info(f"Saved testing data to {test_path}")
        
        return train_data, test_data

if __name__ == "__main__":
    preparator = DataPreparator()
    train_data, test_data = preparator.prepare_training_data()
    logging.info("Data preparation completed successfully") 