import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Optional, Tuple
import glob
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_data_fetch.log'),
        logging.StreamHandler()
    ]
)

class DataValidator:
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate OHLCV data for completeness and quality."""
        if df.empty:
            return False, "DataFrame is empty"
        
        # Check for required columns
        required_cols = ['price']
        if not all(col in df.columns for col in required_cols):
            return False, f"Missing required columns. Found: {df.columns.tolist()}"
        
        # Check for missing values
        if df['price'].isnull().any():
            return False, "Found null values in price data"
        
        # Check for zero or negative prices
        if (df['price'] <= 0).any():
            return False, "Found zero or negative prices"
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            return False, "Found duplicate timestamps"
        
        # Check for sufficient data points
        if len(df) < 50:  # Minimum required for technical indicators
            return False, f"Insufficient data points: {len(df)}"
        
        return True, "Data validation passed"

class DataCleanup:
    def __init__(self):
        self.retention_periods = {
            'raw': 7,  # Keep raw data for 7 days
            'features': 30,  # Keep processed features for 30 days
            'processed': 30  # Keep merged data for 30 days
        }
    
    def cleanup_old_files(self, directory: str, pattern: str, data_type: str):
        """Remove files older than retention period for the specified data type."""
        current_time = datetime.now()
        files = glob.glob(os.path.join(directory, pattern))
        
        for file in files:
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(file))
                age_days = (current_time - file_time).days
                
                if age_days > self.retention_periods[data_type]:
                    os.remove(file)
                    logging.info(f"Removed old {data_type} file: {file}")
            except Exception as e:
                logging.error(f"Error cleaning up {file}: {str(e)}")

class MarketDataFetcher:
    def __init__(self):
        self.dextools_api_key = "L8tBdVzmLNaVJfrmhlGr87efKnpkj1Ez4lO58nBi"
        self.dextools_headers = {"X-API-KEY": self.dextools_api_key}
        self.dextools_endpoints = {
            "gainers": "https://public-api.dextools.io/trial/v2/ranking/solana/gainers",
            "losers": "https://public-api.dextools.io/trial/v2/ranking/solana/losers"
        }
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        
        # Create necessary directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('data/features', exist_ok=True)
        
        # Initialize validator and cleanup
        self.validator = DataValidator()
        self.cleanup = DataCleanup()

    def fetch_dextools_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch top 100 data from DEXTools."""
        results = {}
        for name, url in self.dextools_endpoints.items():
            logging.info(f"Fetching {name} from DEXTools...")
            for attempt in range(3):
                try:
                    response = requests.get(url, headers=self.dextools_headers)
                    if response.status_code == 200:
                        data = response.json()
                        df = pd.DataFrame(data)
                        top_100 = df.head(100)
                        
                        # Save raw data
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        raw_path = f'data/raw/top_100_solana_{name}_{timestamp}.csv'
                        top_100.to_csv(raw_path, index=False)
                        
                        # Save latest version
                        latest_path = f'top_100_solana_{name}.csv'
                        top_100.to_csv(latest_path, index=False)
                        
                        results[name] = top_100
                        logging.info(f"Saved {latest_path} ({len(top_100)} rows)")
                        break
                    elif response.status_code == 429:
                        logging.warning(f"Rate limited on {name}, attempt {attempt+1}/3. Waiting 10 seconds...")
                        time.sleep(10)
                    else:
                        logging.error(f"Failed to fetch {name}: {response.status_code} {response.text}")
                        break
                except Exception as e:
                    logging.error(f"Error fetching {name}: {str(e)}")
                    if attempt == 2:
                        raise
                time.sleep(1.1)  # Wait between requests
            # Add a longer delay between gainers and losers
            time.sleep(3)
        return results

    def get_token_addresses(self) -> List[str]:
        """Extract token addresses from the latest DEXTools data."""
        addresses = set()
        for name in self.dextools_endpoints.keys():
            try:
                df = pd.read_csv(f'top_100_solana_{name}.csv')
                if 'data' in df.columns:
                    # Parse the data column which contains string representations of dictionaries
                    for data_str in df['data']:
                        try:
                            # Convert string representation to actual dictionary
                            data_dict = eval(data_str)
                            # Extract addresses from mainToken and sideToken
                            if 'mainToken' in data_dict and isinstance(data_dict['mainToken'], dict):
                                addresses.add(data_dict['mainToken']['address'])
                            if 'sideToken' in data_dict and isinstance(data_dict['sideToken'], dict):
                                addresses.add(data_dict['sideToken']['address'])
                        except Exception as e:
                            logging.error(f"Error parsing data string: {str(e)}")
                            continue
            except Exception as e:
                logging.error(f"Error reading {name} data: {str(e)}")
        
        addresses = list(addresses)
        logging.info(f"Extracted {len(addresses)} unique token addresses")
        return addresses

    def fetch_coingecko_ohlcv(self, token_address: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from CoinGecko for a specific token with retry logic."""
        max_retries = 3
        base_delay = 1.2  # Base delay in seconds
        
        for attempt in range(max_retries):
            try:
                url = f"{self.coingecko_base_url}/coins/solana/contract/{token_address}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': str(days)
                }
                
                # Calculate delay with exponential backoff
                delay = 2.0 * (2 ** attempt)
                time.sleep(delay)
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    return df
                elif response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        logging.warning(f"Rate limited on attempt {attempt + 1}, waiting {delay} seconds...")
                        continue
                    else:
                        logging.error(f"Max retries reached for {token_address}")
                        return None
                elif response.status_code == 404:
                    logging.warning(f"Token not found: {token_address}")
                    return None
                else:
                    logging.error(f"Failed to fetch OHLCV for {token_address}: {response.status_code}")
                    return None
                    
            except Exception as e:
                logging.error(f"Error fetching OHLCV for {token_address}: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                continue
        
        return None

    def add_technical_indicators(self, df: pd.DataFrame, token_address: str) -> pd.DataFrame:
        """Add technical indicators to the OHLCV data."""
        # Basic price transformations
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        # Moving Averages
        df['sma_10'] = SMAIndicator(close=df['price'], window=10).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['price'], window=50).sma_indicator()
        df['ema_10'] = EMAIndicator(close=df['price'], window=10).ema_indicator()
        df['ema_50'] = EMAIndicator(close=df['price'], window=50).ema_indicator()
        df['ema_200'] = EMAIndicator(close=df['price'], window=200).ema_indicator()
        
        # Volatility Indicators
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        df['volatility_50'] = df['returns'].rolling(window=50).std()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['price'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_pct'] = bb.bollinger_pband()
        
        # RSI
        df['rsi_14'] = RSIIndicator(close=df['price'], window=14).rsi()
        df['rsi_21'] = RSIIndicator(close=df['price'], window=21).rsi()
        
        # MACD
        macd = MACD(close=df['price'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['price'], low=df['price'], close=df['price'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX (Trend Strength)
        adx = ADXIndicator(high=df['price'], low=df['price'], close=df['price'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # ATR (Volatility)
        atr = AverageTrueRange(high=df['price'], low=df['price'], close=df['price'])
        df['atr'] = atr.average_true_range()
        
        # Price Momentum
        df['momentum_1'] = df['price'].pct_change(1)
        df['momentum_5'] = df['price'].pct_change(5)
        df['momentum_10'] = df['price'].pct_change(10)
        df['momentum_20'] = df['price'].pct_change(20)
        
        # Price Acceleration
        df['acceleration'] = df['momentum_1'].diff()
        
        # Price Range
        df['daily_range'] = df['price'].rolling(window=1).max() - df['price'].rolling(window=1).min()
        df['range_5'] = df['price'].rolling(window=5).max() - df['price'].rolling(window=5).min()
        
        # Add token address
        df['token_address'] = token_address
        
        return df

    def get_historical_data(self, token_address: str) -> Optional[pd.DataFrame]:
        """Get historical data for a token from the last 7 days."""
        try:
            # Get all raw data files for this token from the last 7 days
            pattern = f'data/raw/ohlcv_{token_address}_*.csv'
            files = glob.glob(pattern)
            
            if not files:
                return None
            
            # Read and combine all historical data
            historical_data = []
            for file in files:
                try:
                    df = pd.read_csv(file, index_col='timestamp', parse_dates=True)
                    historical_data.append(df)
                except Exception as e:
                    logging.error(f"Error reading historical file {file}: {str(e)}")
            
            if not historical_data:
                return None
            
            # Combine and sort all data
            combined_data = pd.concat(historical_data, axis=0)
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data.sort_index(inplace=True)
            
            return combined_data
            
        except Exception as e:
            logging.error(f"Error getting historical data for {token_address}: {str(e)}")
            return None

    def process_token_data(self, token_address: str) -> Optional[pd.DataFrame]:
        """Fetch and process data for a single token."""
        try:
            # Fetch new OHLCV data
            new_ohlcv_data = self.fetch_coingecko_ohlcv(token_address)
            if new_ohlcv_data is None:
                return None
            
            # Get historical data
            historical_data = self.get_historical_data(token_address)
            
            # Combine new and historical data
            if historical_data is not None:
                combined_data = pd.concat([historical_data, new_ohlcv_data], axis=0)
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data.sort_index(inplace=True)
            else:
                combined_data = new_ohlcv_data
            
            # Validate combined data
            is_valid, message = self.validator.validate_ohlcv_data(combined_data)
            if not is_valid:
                logging.warning(f"Data validation failed for {token_address}: {message}")
                return None
            
            # Add technical indicators
            processed_data = self.add_technical_indicators(combined_data, token_address)
            
            # Save processed data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            processed_path = f'data/features/{token_address}_{timestamp}.csv'
            processed_data.to_csv(processed_path)
            
            # Save latest version
            latest_path = f'data/features/{token_address}_latest.csv'
            processed_data.to_csv(latest_path)
            
            logging.info(f"Processed and saved data for {token_address}")
            return processed_data
            
        except Exception as e:
            logging.error(f"Error processing data for {token_address}: {str(e)}")
            return None

    def run_daily_update(self):
        """Run the complete daily data update process."""
        try:
            # Clean up old files with different retention periods
            self.cleanup.cleanup_old_files('data/raw', 'ohlcv_*.csv', 'raw')
            self.cleanup.cleanup_old_files('data/features', '*_*.csv', 'features')
            self.cleanup.cleanup_old_files('data/processed', 'merged_data_*.csv', 'processed')
            
            # Fetch DEXTools data
            dextools_data = self.fetch_dextools_data()
            
            # Get token addresses
            token_addresses = self.get_token_addresses()
            logging.info(f"Found {len(token_addresses)} unique token addresses")
            
            # Process each token
            all_processed_data = []
            for address in token_addresses:
                processed_data = self.process_token_data(address)
                if processed_data is not None:
                    all_processed_data.append(processed_data)
            
            # Merge all processed data
            if all_processed_data:
                merged_data = pd.concat(all_processed_data, axis=0)
                merged_data.sort_index(inplace=True)
                
                # Save merged data
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                merged_path = f'data/processed/merged_data_{timestamp}.csv'
                merged_data.to_csv(merged_path)
                
                # Save latest version
                latest_merged_path = 'data/processed/merged_data_latest.csv'
                merged_data.to_csv(latest_merged_path)
                
                logging.info(f"Saved merged data with {len(merged_data)} rows")
            
            logging.info("Daily update completed successfully")
            
        except Exception as e:
            logging.error(f"Error in daily update: {str(e)}")
            raise

if __name__ == "__main__":
    fetcher = MarketDataFetcher()
    fetcher.run_daily_update() 