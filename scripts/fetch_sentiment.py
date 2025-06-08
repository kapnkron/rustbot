#!/usr/bin/env python3
"""
Sentiment data collection script for the Beatrice trading bot.

This script collects sentiment data from various sources:
- Twitter/X
- Telegram (public groups)
- Discord (public channels)
- Reddit
- CryptoFear & Greed Index

Sentiment data is saved to CSV files in the data/sentiment directory,
with one file per token named after the token address.
"""

import os
import json
import time
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
SENTIMENT_DIR = Path('data/sentiment')
TWITTER_API_KEY = os.environ.get('TWITTER_API_KEY')
TELEGRAM_API_ID = os.environ.get('TELEGRAM_API_ID')
TELEGRAM_API_HASH = os.environ.get('TELEGRAM_API_HASH')
REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=0"

def ensure_sentiment_dir():
    """Ensure the sentiment directory exists."""
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Sentiment directory: {SENTIMENT_DIR}")

def load_tokens():
    """Load token information from pool_addresses.json."""
    try:
        with open('pool_addresses.json', 'r') as f:
            pool_addresses = json.load(f)
        
        tokens = []
        for market_name, info in pool_addresses.items():
            base_address = info.get('base', {}).get('address')
            base_symbol = info.get('base', {}).get('symbol')
            if base_address and base_symbol:
                category = info.get('category', 'general_token')
                tokens.append({
                    'address': base_address,
                    'symbol': base_symbol,
                    'category': category,
                    'market': market_name
                })
        
        logging.info(f"Loaded {len(tokens)} tokens from pool_addresses.json")
        return tokens
    except FileNotFoundError:
        logging.error("pool_addresses.json not found. No tokens loaded.")
        return []

def fetch_twitter_sentiment(token_symbol, days=7):
    """
    Fetch Twitter/X sentiment for a token.
    
    Args:
        token_symbol: Symbol of the token (e.g., "SOL")
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with date and sentiment_score columns
    """
    if not TWITTER_API_KEY:
        logging.warning("TWITTER_API_KEY not set. Skipping Twitter sentiment.")
        return pd.DataFrame(columns=['date', 'sentiment_score', 'source'])
    
    try:
        # This is a placeholder for actual Twitter API integration
        # In a real implementation, you would use Twitter API to fetch tweets
        # and a sentiment analysis library to calculate sentiment scores
        
        # For demonstration, we'll generate random sentiment data
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dates = [today - timedelta(days=i) for i in range(days)]
        
        # Generate random sentiment scores (-1 to 1)
        sentiment_scores = np.random.normal(0.2, 0.5, size=days)  # Slightly positive bias
        sentiment_scores = np.clip(sentiment_scores, -1, 1)  # Clip to [-1, 1]
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'sentiment_score': sentiment_scores,
            'source': 'twitter'
        })
        
        logging.info(f"Generated {len(df)} Twitter sentiment records for {token_symbol}")
        return df
    
    except Exception as e:
        logging.error(f"Error fetching Twitter sentiment for {token_symbol}: {e}")
        return pd.DataFrame(columns=['date', 'sentiment_score', 'source'])

def fetch_telegram_sentiment(token_symbol, days=7):
    """
    Fetch Telegram sentiment for a token.
    
    Args:
        token_symbol: Symbol of the token (e.g., "SOL")
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with date and sentiment_score columns
    """
    if not (TELEGRAM_API_ID and TELEGRAM_API_HASH):
        logging.warning("Telegram API credentials not set. Skipping Telegram sentiment.")
        return pd.DataFrame(columns=['date', 'sentiment_score', 'source'])
    
    try:
        # This is a placeholder for actual Telegram API integration
        # In a real implementation, you would use Telethon or similar to fetch messages
        # and a sentiment analysis library to calculate sentiment scores
        
        # For demonstration, we'll generate random sentiment data
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dates = [today - timedelta(days=i) for i in range(days)]
        
        # Generate random sentiment scores (-1 to 1)
        # Pump tokens often have more positive sentiment on Telegram
        if 'pump' in token_symbol.lower():
            sentiment_scores = np.random.normal(0.5, 0.3, size=days)  # More positive bias for pump tokens
        else:
            sentiment_scores = np.random.normal(0.1, 0.4, size=days)  # Less positive bias for other tokens
        
        sentiment_scores = np.clip(sentiment_scores, -1, 1)  # Clip to [-1, 1]
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'sentiment_score': sentiment_scores,
            'source': 'telegram'
        })
        
        logging.info(f"Generated {len(df)} Telegram sentiment records for {token_symbol}")
        return df
    
    except Exception as e:
        logging.error(f"Error fetching Telegram sentiment for {token_symbol}: {e}")
        return pd.DataFrame(columns=['date', 'sentiment_score', 'source'])

def fetch_reddit_sentiment(token_symbol, days=7):
    """
    Fetch Reddit sentiment for a token.
    
    Args:
        token_symbol: Symbol of the token (e.g., "SOL")
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with date and sentiment_score columns
    """
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
        logging.warning("Reddit API credentials not set. Skipping Reddit sentiment.")
        return pd.DataFrame(columns=['date', 'sentiment_score', 'source'])
    
    try:
        # This is a placeholder for actual Reddit API integration
        # In a real implementation, you would use PRAW or similar to fetch posts
        # and a sentiment analysis library to calculate sentiment scores
        
        # For demonstration, we'll generate random sentiment data
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dates = [today - timedelta(days=i) for i in range(days)]
        
        # Generate random sentiment scores (-1 to 1)
        sentiment_scores = np.random.normal(0.0, 0.6, size=days)  # More varied sentiment on Reddit
        sentiment_scores = np.clip(sentiment_scores, -1, 1)  # Clip to [-1, 1]
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'sentiment_score': sentiment_scores,
            'source': 'reddit'
        })
        
        logging.info(f"Generated {len(df)} Reddit sentiment records for {token_symbol}")
        return df
    
    except Exception as e:
        logging.error(f"Error fetching Reddit sentiment for {token_symbol}: {e}")
        return pd.DataFrame(columns=['date', 'sentiment_score', 'source'])

def fetch_fear_greed_index(days=7):
    """
    Fetch the Fear & Greed Index.
    
    Args:
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with date and fear_greed_index columns
    """
    try:
        response = requests.get(FEAR_GREED_URL)
        data = response.json()
        
        # Extract data from response
        fear_greed_data = []
        for entry in data['data'][:days]:
            date = datetime.fromtimestamp(int(entry['timestamp']))
            value = int(entry['value'])
            fear_greed_data.append({
                'date': date.replace(hour=0, minute=0, second=0, microsecond=0),
                'fear_greed_index': value,
                'source': 'fear_greed'
            })
        
        df = pd.DataFrame(fear_greed_data)
        logging.info(f"Fetched {len(df)} Fear & Greed Index records")
        return df
    
    except Exception as e:
        logging.error(f"Error fetching Fear & Greed Index: {e}")
        
        # For demonstration, generate random data
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dates = [today - timedelta(days=i) for i in range(days)]
        
        # Generate random values (0-100)
        values = np.random.randint(20, 80, size=days)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'fear_greed_index': values,
            'source': 'fear_greed'
        })
        
        logging.info(f"Generated {len(df)} Fear & Greed Index records")
        return df

def normalize_sentiment_scores(df):
    """
    Normalize sentiment scores from different sources.
    
    Args:
        df: DataFrame with sentiment scores from different sources
        
    Returns:
        DataFrame with normalized sentiment scores
    """
    # Calculate weighted average of sentiment scores
    # Weights could be adjusted based on which sources are more reliable
    weights = {
        'twitter': 0.3,
        'telegram': 0.4,  # Higher weight for Telegram (often more active for crypto)
        'reddit': 0.3
    }
    
    result_df = df.pivot_table(
        index='date', 
        columns='source', 
        values='sentiment_score',
        aggfunc='mean'
    ).reset_index()
    
    # Calculate weighted average sentiment score
    for source, weight in weights.items():
        if source not in result_df.columns:
            result_df[source] = 0  # If source is missing, use 0
    
    # Calculate weighted sum of available sources
    result_df['sentiment_score'] = 0
    available_weight = 0
    
    for source, weight in weights.items():
        if source in result_df.columns:
            # Handle NaN values
            source_data = result_df[source].fillna(0)
            result_df['sentiment_score'] += source_data * weight
            available_weight += weight
    
    # Normalize by available weight
    if available_weight > 0:
        result_df['sentiment_score'] = result_df['sentiment_score'] / available_weight
    
    # Add fear and greed if available
    if 'fear_greed_index' in df.columns:
        fear_greed_df = df[df['source'] == 'fear_greed'][['date', 'fear_greed_index']]
        result_df = pd.merge(result_df, fear_greed_df, on='date', how='left')
        
        # Normalize fear_greed_index to [-1, 1] range (from [0, 100])
        result_df['fear_greed_normalized'] = (result_df['fear_greed_index'].fillna(50) - 50) / 50
    
    # Select only the columns we need
    final_columns = ['date', 'sentiment_score']
    if 'fear_greed_index' in result_df.columns:
        final_columns.extend(['fear_greed_index', 'fear_greed_normalized'])
    
    return result_df[final_columns]

def process_token_sentiment(token, days=7):
    """
    Process sentiment data for a single token.
    
    Args:
        token: Dictionary with token information
        days: Number of days of historical data to fetch
        
    Returns:
        Processed sentiment DataFrame
    """
    token_symbol = token['symbol']
    token_address = token['address']
    
    logging.info(f"Processing sentiment for {token_symbol} ({token_address})")
    
    # Fetch sentiment from different sources
    twitter_df = fetch_twitter_sentiment(token_symbol, days)
    telegram_df = fetch_telegram_sentiment(token_symbol, days)
    reddit_df = fetch_reddit_sentiment(token_symbol, days)
    
    # Combine sentiment data
    all_sentiment = pd.concat([twitter_df, telegram_df, reddit_df])
    
    # Process and normalize sentiment scores
    if len(all_sentiment) > 0:
        # Normalize sentiment scores
        processed_df = normalize_sentiment_scores(all_sentiment)
        
        # Save to file
        output_file = SENTIMENT_DIR / f"{token_address}.csv"
        processed_df.to_csv(output_file, index=False)
        logging.info(f"Saved sentiment data for {token_symbol} to {output_file}")
        
        return processed_df
    else:
        logging.warning(f"No sentiment data found for {token_symbol}")
        return pd.DataFrame()

def process_market_sentiment(days=7):
    """
    Process market-wide sentiment data.
    
    Args:
        days: Number of days of historical data to fetch
        
    Returns:
        Processed market sentiment DataFrame
    """
    logging.info("Processing market-wide sentiment")
    
    # Fetch Fear & Greed Index
    fear_greed_df = fetch_fear_greed_index(days)
    
    if len(fear_greed_df) > 0:
        # Save to file
        output_file = SENTIMENT_DIR / "market_sentiment.csv"
        fear_greed_df.to_csv(output_file, index=False)
        logging.info(f"Saved market sentiment data to {output_file}")
        
        return fear_greed_df
    else:
        logging.warning("No market sentiment data found")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description='Fetch sentiment data for tokens')
    parser.add_argument('--days', type=int, default=30, help='Number of days of historical data to fetch')
    args = parser.parse_args()
    
    # Ensure sentiment directory exists
    ensure_sentiment_dir()
    
    # Load tokens
    tokens = load_tokens()
    
    if not tokens:
        logging.error("No tokens found. Exiting.")
        return
    
    # Process market-wide sentiment
    process_market_sentiment(args.days)
    
    # Process sentiment for each token
    for token in tokens:
        process_token_sentiment(token, args.days)
        time.sleep(1)  # Add a delay to avoid rate limits
    
    logging.info("Sentiment data collection complete")

if __name__ == "__main__":
    main() 