#!/usr/bin/env python3
"""
Crypto Sentiment Crawler

This script collects sentiment data from various sources including:
1. Twitter (via Twitter API v2) - requires a Twitter developer account and bearer token
2. Crypto forums (via web scraping)
3. Free crypto sentiment APIs (CryptoFear & Greed Index)
4. Public Telegram channels (via web scraping)

The data is processed using TextBlob for sentiment analysis and stored in CSV files.

Note: This script requires the .pytorch_venv virtual environment to be activated:
    source ~/.pytorch_venv/bin/activate
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
import re
from bs4 import BeautifulSoup
from textblob import TextBlob
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import math

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
SENTIMENT_DIR = Path('data/sentiment')
PLOTS_DIR = Path('data/sentiment/plots')
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=0"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "")

# Target URLs for web scraping
CRYPTO_FORUM_URLS = {
    # General crypto forums
    'bitcointalk': 'https://bitcointalk.org/index.php?board=159.0',  # Altcoin Discussion
    'reddit_cryptomoonshots': 'https://old.reddit.com/r/CryptoMoonShots/',
    'reddit_solana': 'https://old.reddit.com/r/solana/',
    
    # DexScreener and similar sites (token-specific pages will be generated dynamically)
    'dexscreener_base': 'https://dexscreener.com/solana',
    'geckoterminal_base': 'https://www.geckoterminal.com/solana/pools'
}

def ensure_directories():
    """Ensure all necessary directories exist."""
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Sentiment directory: {SENTIMENT_DIR}")
    logging.info(f"Plots directory: {PLOTS_DIR}")

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
        logging.warning("pool_addresses.json not found. Using default tokens.")
        # Add default tokens
        default_tokens = [
            # Major tokens
            {
                'address': 'So11111111111111111111111111111111111111112',
                'symbol': 'SOL',
                'category': 'major_token',
                'market': 'SOL/USD'
            },
            
            # Popular Solana tokens
            {
                'address': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
                'symbol': 'BONK',
                'category': 'meme_token',
                'market': 'BONK/USD'
            },
            {
                'address': 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',
                'symbol': 'WIF',
                'category': 'meme_token',
                'market': 'WIF/USD'
            },
            {
                'address': '7aU8xFMNBiJBrxKpH2ZQSkhgVLKCHGgXgEv9mkruSXh4',
                'symbol': 'TRUMP',
                'category': 'meme_token',
                'market': 'TRUMP/USD'
            },
            {
                'address': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
                'symbol': 'JUP',
                'category': 'utility_token',
                'market': 'JUP/USD'
            }
        ]
        return default_tokens
    except Exception as e:
        logging.error(f"Error loading pool_addresses.json: {e}")
        return []

def fetch_fear_greed_index(days=7):
    """
    Fetch the Fear & Greed Index.
    
    Args:
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with date and fear_greed_index columns
    """
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(FEAR_GREED_URL, headers=headers)
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
        return pd.DataFrame(columns=['date', 'fear_greed_index', 'source'])

def scrape_twitter_api(token_symbol, days=7, max_tweets=500, token_category=None):
    """
    Scrape Twitter for mentions of a token using Twitter API v2.
    
    Args:
        token_symbol: Symbol of the token (e.g., "SOL")
        days: Number of days of historical data to fetch
        max_tweets: Maximum number of tweets to fetch
        token_category: Category of the token (e.g., "meme_token", "utility_token")
        
    Returns:
        DataFrame with date and raw_text columns
    """
    if not TWITTER_BEARER_TOKEN:
        logging.error(f"TWITTER_BEARER_TOKEN not set. Cannot fetch sentiment for {token_symbol}.")
        return pd.DataFrame(columns=['date', 'raw_text', 'username', 'source', 'sentiment_score'])
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for query (Twitter API v2 format)
        start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Build query
        query = f"({token_symbol} OR ${token_symbol} OR #{token_symbol.lower()})"
        logging.info(f"Twitter API query: {query}")
        
        # Twitter API v2 search endpoint
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {
            "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}",
            "User-Agent": USER_AGENT
        }
        
        params = {
            "query": query,
            "max_results": 100,  # Max allowed by the API
            "tweet.fields": "created_at,text",
            "start_time": start_date_str,
        }
        
        tweets = []
        next_token = None
        remaining_tweets = max_tweets
        
        # Paginate through results
        while remaining_tweets > 0:
            if next_token:
                params["next_token"] = next_token
                
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                logging.error(f"Twitter API error: {response.status_code} - {response.text}")
                break
                
            data = response.json()
            
            if "data" not in data:
                logging.warning("No tweets found in response")
                break
                
            for tweet in data["data"]:
                tweet_date = datetime.strptime(tweet["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
                tweets.append({
                    'date': tweet_date.replace(hour=0, minute=0, second=0, microsecond=0),
                    'raw_text': tweet["text"],
                    'username': "unknown",  # API doesn't return this in basic query
                    'source': 'twitter'
                })
                
                remaining_tweets -= 1
                if remaining_tweets <= 0:
                    break
            
            # Check if there are more results
            if "meta" in data and "next_token" in data["meta"]:
                next_token = data["meta"]["next_token"]
            else:
                break
        
        df = pd.DataFrame(tweets)
        logging.info(f"Fetched {len(df)} Twitter mentions for {token_symbol}")
        
        # Analyze sentiment
        if len(df) > 0:
            df['sentiment_score'] = df['raw_text'].apply(analyze_sentiment)
            
            # Plot the data
            try:
                # Group by day
                df['date_only'] = pd.to_datetime(df['date']).dt.date
                daily_sentiment = df.groupby('date_only')['sentiment_score'].mean().reset_index()
                
                # Plot
                plt.figure(figsize=(10, 5))
                plt.plot(daily_sentiment['date_only'], daily_sentiment['sentiment_score'], marker='o')
                plt.title(f"{token_symbol} Twitter Sentiment Over Time")
                plt.xlabel("Date")
                plt.ylabel("Average Sentiment")
                plt.grid(True)
                plt.tight_layout()
                
                # Save plot
                plot_path = PLOTS_DIR / f"{token_symbol.lower()}_twitter_sentiment.png"
                plt.savefig(plot_path)
                plt.close()
                logging.info(f"Saved Twitter sentiment plot to {plot_path}")
            except Exception as e:
                logging.error(f"Error creating Twitter sentiment plot: {e}")
        
        return df
    
    except Exception as e:
        logging.error(f"Error fetching Twitter data for {token_symbol}: {e}")
        return pd.DataFrame(columns=['date', 'raw_text', 'username', 'source', 'sentiment_score'])

def scrape_dexscreener(token_symbol, days=7):
    """
    Scrape DexScreener comments for a token.
    
    Args:
        token_symbol: Symbol of the token (e.g., "SOL")
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with date and raw_text columns
    """
    # First try to find the token page
    search_url = f"{CRYPTO_FORUM_URLS['dexscreener_base']}?q={token_symbol}"
    
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find token links (this selector needs to be adjusted based on actual site structure)
        token_links = soup.select('a[href*="/solana/"]')
        
        results = []
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = today - timedelta(days=days)
        
        # Try to find a matching token page
        token_url = None
        for link in token_links:
            link_text = link.text.strip()
            if token_symbol.lower() in link_text.lower():
                token_url = "https://dexscreener.com" + link['href']
                break
        
        if not token_url:
            logging.warning(f"No DexScreener page found for {token_symbol}")
            return pd.DataFrame(columns=['date', 'raw_text', 'source'])
        
        # Visit the token page
        response = requests.get(token_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find comments (this selector needs to be adjusted based on actual site structure)
        comments = soup.select('div.comments-section .comment')
        
        for comment in comments:
            try:
                # Extract comment text
                text_elem = comment.select_one('.comment-text')
                if not text_elem:
                    continue
                
                text = text_elem.text.strip()
                
                # Extract date (simplified, adjust based on actual format)
                date_elem = comment.select_one('.comment-date')
                if date_elem:
                    date_text = date_elem.text.strip()
                    # Simplified date parsing - adjust as needed
                    try:
                        post_date = datetime.strptime(date_text, "%Y-%m-%d").replace(
                            hour=0, minute=0, second=0, microsecond=0
                        )
                    except:
                        post_date = today
                else:
                    post_date = today
                
                # Only include comments within the requested time range
                if post_date >= cutoff_date:
                    results.append({
                        'date': post_date,
                        'raw_text': text,
                        'source': 'dexscreener'
                    })
            except Exception as e:
                logging.error(f"Error processing DexScreener comment: {e}")
                continue
        
        df = pd.DataFrame(results)
        logging.info(f"Scraped {len(df)} DexScreener mentions for {token_symbol}")
        return df
    
    except Exception as e:
        logging.error(f"Error scraping DexScreener for {token_symbol}: {e}")
        return pd.DataFrame(columns=['date', 'raw_text', 'source'])

def analyze_sentiment(text):
    """
    Analyze sentiment of text using TextBlob.
    
    Args:
        text: Text to analyze
        
    Returns:
        Sentiment score between -1 (negative) and 1 (positive)
    """
    if not text or not isinstance(text, str):
        return 0
    
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return 0

def process_token_sentiment(token, days=7, max_tweets=500):
    """
    Process sentiment data for a single token.
    
    Args:
        token: Dictionary with token information
        days: Number of days of historical data to fetch
        max_tweets: Maximum number of tweets to fetch per token
        
    Returns:
        Processed sentiment DataFrame
    """
    token_symbol = token['symbol']
    token_address = token['address']
    
    logging.info(f"Processing sentiment for {token_symbol} ({token_address})")
    
    # Scrape mentions from different sources
    twitter_df = scrape_twitter_api(token_symbol, days, max_tweets, token['category'])
    dexscreener_df = scrape_dexscreener(token_symbol, days)
    
    # Combine all mentions - improved to avoid FutureWarning
    dataframes_to_concat = []
    if len(twitter_df) > 0:
        dataframes_to_concat.append(twitter_df)
    if len(dexscreener_df) > 0:
        dataframes_to_concat.append(dexscreener_df)
        
    if dataframes_to_concat:
        all_mentions = pd.concat(dataframes_to_concat)
    else:
        all_mentions = pd.DataFrame()
    
    if len(all_mentions) == 0:
        logging.warning(f"No mentions found for {token_symbol}")
        # Create empty sentiment dataframe with proper structure for pipeline compatibility
        empty_df = pd.DataFrame({
            'date': [datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)],
            'sentiment_score': [0]  # Neutral sentiment
        })
        output_file = SENTIMENT_DIR / f"{token_symbol.lower()}_sentiment.csv"
        empty_df.to_csv(output_file, index=False)
        logging.info(f"Saved empty sentiment data for {token_symbol} to {output_file}")
        return empty_df
    
    # Analyze sentiment for mentions that don't already have a sentiment score
    if 'sentiment_score' not in all_mentions.columns:
        all_mentions['sentiment_score'] = all_mentions['raw_text'].apply(analyze_sentiment)
    
    # Group by date and source, calculate average sentiment
    sentiment_by_date = all_mentions.groupby(['date', 'source'])['sentiment_score'].mean().reset_index()
    
    # Pivot to get one row per date with columns for each source
    pivoted = sentiment_by_date.pivot_table(
        index='date', 
        columns='source', 
        values='sentiment_score',
        aggfunc='mean'
    ).reset_index()
    
    # Calculate overall sentiment (average across all sources)
    for col in pivoted.columns:
        if col != 'date':
            pivoted[col] = pivoted[col].fillna(0)
    
    source_columns = [col for col in pivoted.columns if col != 'date']
    if source_columns:
        # Weight Twitter sentiment more heavily (adjust weights as needed)
        if 'twitter' in source_columns:
            weights = {'twitter': 0.7}
            weights.update({col: (0.3 / (len(source_columns) - 1)) for col in source_columns if col != 'twitter'})
            
            pivoted['sentiment_score'] = sum(pivoted[col] * weights.get(col, 1/len(source_columns)) for col in source_columns)
        else:
            pivoted['sentiment_score'] = pivoted[source_columns].mean(axis=1)
    else:
        pivoted['sentiment_score'] = 0
    
    # Save to file using token symbol as the filename
    output_file = SENTIMENT_DIR / f"{token_symbol.lower()}_sentiment.csv"
    pivoted[['date', 'sentiment_score']].to_csv(output_file, index=False)
    logging.info(f"Saved sentiment data for {token_symbol} to {output_file}")
    
    # Create combined sentiment plot
    try:
        if len(pivoted) > 0:
            plt.figure(figsize=(12, 6))
            
            # Plot sentiment from each source if available
            for source in source_columns:
                if source in pivoted.columns:
                    plt.plot(pivoted['date'], pivoted[source], marker='o', linestyle='--', alpha=0.7, label=f"{source}")
            
            # Plot overall sentiment
            plt.plot(pivoted['date'], pivoted['sentiment_score'], marker='s', linewidth=2, color='black', label='Combined')
            
            plt.title(f"{token_symbol} Sentiment Analysis")
            plt.xlabel("Date")
            plt.ylabel("Sentiment Score")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plot_path = PLOTS_DIR / f"{token_symbol.lower()}_sentiment.png"
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Saved combined sentiment plot to {plot_path}")
    except Exception as e:
        logging.error(f"Error creating combined sentiment plot: {e}")
    
    return pivoted

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
    
    # Fetch general SOL sentiment
    sol_sentiment = scrape_twitter_api("SOL", days, token_category="major_token")
    
    if 'sentiment_score' not in sol_sentiment.columns:
        sol_sentiment['sentiment_score'] = sol_sentiment['raw_text'].apply(analyze_sentiment)
    
    sol_daily = sol_sentiment.groupby('date')['sentiment_score'].mean().reset_index()
    sol_daily['source'] = 'sol_twitter'
    
    # Combine market sentiment sources
    combined = pd.concat([
        fear_greed_df.rename(columns={'fear_greed_index': 'score'}).assign(source='fear_greed'),
        sol_daily.rename(columns={'sentiment_score': 'score'}).assign(source='sol_twitter')
    ])
    
    # Create pivot table
    if len(combined) > 0:
        market_pivot = combined.pivot_table(
            index='date',
            columns='source',
            values='score',
            aggfunc='mean'
        ).reset_index()
        
        # Normalize fear_greed_index to [-1, 1] range (from [0, 100])
        if 'fear_greed' in market_pivot.columns:
            market_pivot['fear_greed_normalized'] = (market_pivot['fear_greed'].fillna(50) - 50) / 50
        
        # Calculate combined market sentiment
        sentiment_cols = []
        if 'fear_greed_normalized' in market_pivot.columns:
            sentiment_cols.append('fear_greed_normalized')
        if 'sol_twitter' in market_pivot.columns:
            sentiment_cols.append('sol_twitter')
        
        if sentiment_cols:
            market_pivot['market_sentiment'] = market_pivot[sentiment_cols].mean(axis=1)
        else:
            market_pivot['market_sentiment'] = 0
        
        # Save to file
        output_file = SENTIMENT_DIR / "market_sentiment.csv"
        market_pivot[['date', 'market_sentiment']].to_csv(output_file, index=False)
        logging.info(f"Saved market sentiment data to {output_file}")
        
        # Create market sentiment plot
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot individual sentiment sources
            for col in sentiment_cols:
                plt.plot(market_pivot['date'], market_pivot[col], marker='o', linestyle='--', alpha=0.7, label=col)
            
            # Plot combined sentiment
            plt.plot(market_pivot['date'], market_pivot['market_sentiment'], marker='s', linewidth=2, color='black', label='Combined')
            
            plt.title("Market Sentiment Analysis")
            plt.xlabel("Date")
            plt.ylabel("Sentiment Score")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plot_path = PLOTS_DIR / "market_sentiment.png"
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Saved market sentiment plot to {plot_path}")
        except Exception as e:
            logging.error(f"Error creating market sentiment plot: {e}")
        
        return market_pivot
    else:
        logging.warning("No market sentiment data found")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description='Crawl and analyze sentiment data for tokens')
    parser.add_argument('--days', type=int, default=30, help='Number of days of historical data to fetch')
    parser.add_argument('--token', type=str, help='Process only a specific token symbol')
    parser.add_argument('--max-tweets', type=int, default=500, help='Maximum number of tweets to fetch per token')
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    # Process market-wide sentiment
    process_market_sentiment(args.days)
    
    # Load tokens from pool_addresses.json
    tokens = load_tokens()
    
    # Common tokens that might not be in pool_addresses.json
    common_tokens = {
        # Major tokens
        "SOL": {"address": "So11111111111111111111111111111111111111112", "category": "major_token"},
        "BTC": {"address": "BTC_placeholder_address", "category": "major_token"},
        "ETH": {"address": "ETH_placeholder_address", "category": "major_token"},
        
        # Popular Solana tokens
        "BONK": {"address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263", "category": "meme_token"},
        "WIF": {"address": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm", "category": "meme_token"},
        "TRUMP": {"address": "7aU8xFMNBiJBrxKpH2ZQSkhgVLKCHGgXgEv9mkruSXh4", "category": "meme_token"},
        "JUP": {"address": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN", "category": "utility_token"},
        "PYTH": {"address": "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3", "category": "utility_token"},
        "HADES": {"address": "4TLQ91Xz3eBiTqgqt3KpPLNyBv8CnbowJFcU3sYcWzYj", "category": "gaming_token"},
        "BOOK": {"address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263", "category": "meme_token"},
        "BOME": {"address": "D6bQ5YmD4exVrED2HXFYwu81xsRqWX9o77KEP6DphQFx", "category": "meme_token"},
        "RAY": {"address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R", "category": "utility_token"},
        "COPE": {"address": "8HGyAAB1yoM1ttS7pXjHMa3dukTFGQggnFFH3hJZgzQh", "category": "utility_token"},
        "MSOL": {"address": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So", "category": "staking_token"},
        "STSOL": {"address": "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj", "category": "staking_token"}
    }
    
    # Process sentiment for tokens
    if args.token:
        # Process only the specified token
        token_symbol = args.token.upper()
        
        # First check if the token is in our loaded tokens
        matching_tokens = [t for t in tokens if t['symbol'].upper() == token_symbol]
        
        if matching_tokens:
            for token in matching_tokens:
                process_token_sentiment(token, args.days, args.max_tweets)
        elif token_symbol in common_tokens:
            # If it's a common token, create a temporary token entry
            token_info = common_tokens[token_symbol]
            temp_token = {
                'address': token_info["address"],
                'symbol': token_symbol,
                'category': token_info["category"],
                'market': f"{token_symbol}/USD"
            }
            process_token_sentiment(temp_token, args.days, args.max_tweets)
        else:
            # For unknown tokens, create a generic entry
            logging.warning(f"Token '{token_symbol}' not found in known tokens. Creating generic entry.")
            temp_token = {
                'address': f"{token_symbol.lower()}_address",
                'symbol': token_symbol,
                'category': 'unknown',
                'market': f"{token_symbol}/USD"
            }
            process_token_sentiment(temp_token, args.days, args.max_tweets)
    else:
        # Process all tokens from pool_addresses.json
        # Plus the common tokens if they're not already in the list
        token_symbols = set(t['symbol'].upper() for t in tokens)
        
        # Process loaded tokens
        for token in tokens:
            process_token_sentiment(token, args.days, args.max_tweets)
            time.sleep(2)  # Add a delay to avoid being blocked by websites
        
        # Add common tokens if not already processed
        for symbol, info in common_tokens.items():
            if symbol not in token_symbols:
                logging.info(f"Adding common token {symbol}")
                temp_token = {
                    'address': info["address"],
                    'symbol': symbol,
                    'category': info["category"],
                    'market': f"{symbol}/USD"
                }
                process_token_sentiment(temp_token, args.days, args.max_tweets)
                time.sleep(2)  # Add a delay to avoid being blocked by websites
    
    logging.info("Sentiment data collection complete")

if __name__ == "__main__":
    main() 