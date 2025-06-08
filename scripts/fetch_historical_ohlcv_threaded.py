#!/usr/bin/env python3

import sys
import time
import pandas as pd
from solana.rpc.api import Client
from solders.pubkey import Pubkey
from solders.signature import Signature
from datetime import datetime, timezone, timedelta
import logging
import os
import json
import requests
from typing import Dict, Any, Optional, List, Set
from dotenv import load_dotenv
from solders.rpc.responses import GetTransactionResp
from solana.rpc.core import RPCException
from solana.rpc.types import TxOpts
from pathlib import Path
from solders.rpc.config import RpcSignaturesForAddressConfig
from solders.transaction_status import (
    UiParsedInstruction, 
    UiPartiallyDecodedInstruction,
    ParsedInstruction
)
from solana.rpc.commitment import Confirmed, Commitment
import argparse
import threading
import queue
import concurrent.futures
from tqdm import tqdm

# --- Environment Loading ---
load_dotenv()

# --- Configuration ---
ALCHEMY_API_KEY = os.environ.get("ALCHEMY_API_KEY", "").strip('\"')

if ALCHEMY_API_KEY:
    RPC_ENDPOINT = f"https://solana-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
else:
    logging.critical("ALCHEMY_API_KEY not set. Please set it in your environment. Exiting.")
    sys.exit(1)

# Constants
USDC_MINT_STR = "EPjFWdd5AufqSSqeM2qN2UjpeeF4AB9Gg7LgVem31"
RAYDIUM_LIQUIDITY_API_URL = "https://api.raydium.io/v2/sdk/liquidity/mainnet.json"
JUP_MINT_STR = "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"

# --- DEX Program IDs ---
DEX_PROGRAM_IDS = {
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "Raydium AMM V4",
    "whirLbMiFpDGpd1pfuBYwzD2DFWT4B5s84aP1MTp23i": "Orca Whirlpools",
    "JUP6LkbZbjS1jKKwapdHch4GTfBNCLdtNF5BkvsrMC8": "Jupiter Aggregator V6",
    "CURVGoZn8zySK71KBT62qf2wBsX_Cz4spWhH7aYw9wo": "Curve",
}

# --- Market Configurations ---
MANUAL_MARKET_CONFIGS = {
    "SOL-USDC": {
        "id": "8BnEgHoWFysVcuFFX7QztDmzuH8r5ZFvyP3sYwn1XTh6", 
        "name": "SOL-USDC",
        "base_mint": "So11111111111111111111111111111111111111112", # WSOL
        "quote_mint": USDC_MINT_STR,
        "base_decimals": 9,
        "quote_decimals": 6
    }
}

# --- Script Settings ---
CANDLE_INTERVAL_SECONDS = 300
REQUEST_DELAY_SECONDS = 0.25
MAX_DAILY_REQUESTS_LIMIT = 40000
MAX_SIGNATURES_TO_FETCH = 1000
MAX_TRANSACTIONS_PER_BATCH = 50
MAX_TX_TO_INSPECT = 10
INSPECTION_MODE = False
NUM_DAYS_TO_FETCH = 7
BATCH_DELAY_SECONDS = 1.0
EMPTY_RESPONSE_LIMIT = 5
MAX_WORKER_THREADS = 5  # Number of parallel threads

# --- Logging Setup ---
error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
error_log_handler = logging.FileHandler("data_fetcher_error.log")
error_log_handler.setLevel(logging.ERROR)
error_log_handler.setFormatter(error_formatter)

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("data_fetcher.log"),
                        logging.StreamHandler(),
                        error_log_handler
                    ])

# Thread-safe global state
processed_signatures_global = set()
processed_signatures_lock = threading.Lock()
request_count_lock = threading.Lock()
current_request_count = 0

# --- Utility Functions ---
def load_market_configs_from_json(file_path: Path, token_mints_whitelist: Optional[List[str]] = None) -> Dict[str, Any]:
    """Loads market configurations from the pool_addresses.json file."""
    configs = {}
    if not file_path.exists():
        logging.error(f"Market config file not found at {file_path}")
        return configs

    with open(file_path, 'r') as f:
        pool_data = json.load(f)

    for market_name, info in pool_data.items():
        base_mint = info.get("base", {}).get("address")
        quote_mint = info.get("quote", {}).get("address")
        
        if not base_mint or not quote_mint:
            continue
            
        if token_mints_whitelist and base_mint not in token_mints_whitelist:
            continue

        base_symbol = info.get("base", {}).get("symbol", base_mint[:6])
        quote_symbol = info.get("quote", {}).get("symbol", "QUOTE")
        
        configs[market_name] = {
            "id": info.get("poolId", ""),
            "name": market_name,
            "base_mint": base_mint,
            "quote_mint": quote_mint,
            "base_decimals": info.get("base", {}).get("decimals", 9),
            "quote_decimals": info.get("quote", {}).get("decimals", 6),
        }
    
    logging.info(f"Loaded {len(configs)} market configurations from {file_path}.")
    return configs

def load_processed_signatures(processed_signatures_log_file: str) -> set[str]:
    processed_market_signatures = set()
    if os.path.exists(processed_signatures_log_file):
        with open(processed_signatures_log_file, 'r') as f:
            valid_signatures = set()
            for line_num, line in enumerate(f):
                stripped_line = line.strip()
                if stripped_line:
                    try:
                        Signature.from_string(stripped_line)
                        valid_signatures.add(stripped_line)
                    except ValueError:
                        logging.warning(f"Skipping invalid signature at line {line_num+1}: '{stripped_line}'")
            processed_market_signatures = valid_signatures
        logging.info(f"Loaded {len(processed_market_signatures)} processed signatures from {processed_signatures_log_file}.")
    return processed_market_signatures

def log_processed_signature(signature: str, processed_signatures_log_file: str, current_market_processed_set: set[str]):
    with processed_signatures_lock:
        current_market_processed_set.add(signature)
        processed_signatures_global.add(signature)
    
    with open(processed_signatures_log_file, 'a') as f:
        f.write(signature + '\n')

def aggregate_trades_to_ohlcv(trades_df: pd.DataFrame, interval_seconds: int, market_name: str) -> pd.DataFrame:
    if trades_df.empty:
        logging.info(f"[{market_name}] No trades to aggregate.")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'trade_count'])

    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df = trades_df.set_index('timestamp')
    trades_df['price'] = pd.to_numeric(trades_df['price'])
    trades_df['volume'] = pd.to_numeric(trades_df['volume'])
    interval_td = pd.Timedelta(seconds=interval_seconds)
    ohlcv = trades_df['price'].resample(interval_td).ohlc()
    volume = trades_df['volume'].resample(interval_td).sum()
    trade_count = trades_df['price'].resample(interval_td).count()
    ohlcv['volume'] = volume
    ohlcv['trade_count'] = trade_count
    return ohlcv.dropna()

# Import the other needed functions from your original script
# parse_trade_from_inner_instructions, parse_trade_from_balances, identify_dex_and_parse, etc.

def fetch_trade_history_for_market(
    market_id: str, 
    market_name: str, 
    processed_signatures_log_file: Path,
    pool_base_mint_str: str,
    pool_quote_mint_str: str
) -> pd.DataFrame:
    # Implementation from original script
    pass

def process_single_transaction(
    client: Client,
    signature: Signature,
    market_name: str,
    pool_base_mint: str,
    pool_quote_mint: str,
    log_file: Path,
    processed_set: set[str]
) -> Optional[Dict[str, Any]]:
    # Implementation from original script
    pass

def worker_thread(market_queue, results, progress_bar=None):
    client = Client(RPC_ENDPOINT)
    
    while not market_queue.empty():
        try:
            market_config = market_queue.get_nowait()
        except queue.Empty:
            break
            
        try:
            result = fetch_and_save_market_data(market_config, client)
            results.append(result)
            if progress_bar:
                progress_bar.update(1)
        except Exception as e:
            logging.error(f"Error processing market {market_config['name']}: {str(e)}")
        finally:
            market_queue.task_done()

def fetch_and_save_market_data(market_config: Dict[str, Any], client: Client = None) -> str:
    """Modified to accept an existing client and handle thread safety"""
    market_name = market_config["name"]
    market_id = market_config["id"]
    base_mint = market_config["base_mint"]
    quote_mint = market_config["quote_mint"]
    
    if client is None:
        client = Client(RPC_ENDPOINT)
    
    logging.info(f"Starting data fetch for market: {market_name}")
    
    # Create data directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Prepare output file paths
    processed_log_file = Path(f"data/processed/{market_name.lower()}_processed_signatures.log")
    output_file = Path(f"data/raw/{market_name.lower()}_trades.csv")
    ohlcv_file = Path(f"data/processed/DATA_{base_mint}_{quote_mint}_OHLCV_5min.csv")
    
    # Fetch and process trade history
    trades_df = fetch_trade_history_for_market(
        market_id=market_id,
        market_name=market_name,
        processed_signatures_log_file=processed_log_file,
        pool_base_mint_str=base_mint,
        pool_quote_mint_str=quote_mint
    )
    
    if not trades_df.empty:
        # Save raw trades
        trades_df.to_csv(output_file, index=False)
        logging.info(f"Saved {len(trades_df)} trades to {output_file}")
        
        # Aggregate to OHLCV and save
        ohlcv_df = aggregate_trades_to_ohlcv(trades_df, CANDLE_INTERVAL_SECONDS, market_name)
        
        # Standardize column names: rename the timestamp index to 'date' for consistent naming
        ohlcv_df.index.name = 'date'
        
        ohlcv_df.to_csv(ohlcv_file)
        logging.info(f"Saved OHLCV data with standardized column 'date' to {ohlcv_file}")
        return f"Processed {len(trades_df)} trades for {market_name}"
    else:
        logging.info(f"No trades found for {market_name}")
        return f"No trades found for {market_name}"

def process_markets_threaded(markets_to_process: List[Dict[str, Any]], max_workers: int = MAX_WORKER_THREADS):
    """Process multiple markets in parallel using threads"""
    market_queue = queue.Queue()
    results = []
    
    # Add all markets to the queue
    for market in markets_to_process:
        market_queue.put(market)
    
    # Adjust workers if fewer markets than max_workers
    actual_workers = min(max_workers, len(markets_to_process))
    
    logging.info(f"Processing {len(markets_to_process)} markets with {actual_workers} worker threads")
    
    # Create progress bar
    progress_bar = tqdm(total=len(markets_to_process), desc="Processing Markets")
    
    # Start worker threads
    threads = []
    for _ in range(actual_workers):
        thread = threading.Thread(
            target=worker_thread,
            args=(market_queue, results, progress_bar)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    progress_bar.close()
    return results

def main():
    parser = argparse.ArgumentParser(description='Fetch historical OHLCV data from Solana')
    parser.add_argument('--days', type=int, default=NUM_DAYS_TO_FETCH,
                      help=f'Number of days of historical data to fetch (default: {NUM_DAYS_TO_FETCH})')
    parser.add_argument('--token', type=str, help='Specific token symbol to fetch (e.g., SOL)')
    parser.add_argument('--threads', type=int, default=MAX_WORKER_THREADS,
                      help=f'Number of threads to use (default: {MAX_WORKER_THREADS})')
    args = parser.parse_args()
    
    global NUM_DAYS_TO_FETCH
    NUM_DAYS_TO_FETCH = args.days
    max_workers = args.threads
    
    # Load market configurations
    markets_to_process = []
    pool_addresses_file = Path("pool_addresses.json")
    
    if args.token:
        token_symbol = args.token.upper()
        logging.info(f"Fetching data for specific token: {token_symbol}")
        
        # Load market configurations
        market_configs = load_market_configs_from_json(pool_addresses_file)
        
        # Find markets that match the token symbol
        for market_name, config in market_configs.items():
            base_symbol = market_name.split('-')[0]
            if base_symbol.upper() == token_symbol:
                markets_to_process.append(config)
                logging.info(f"Added market {market_name} for token {token_symbol}")
    else:
        # Process all markets
        market_configs = load_market_configs_from_json(pool_addresses_file)
        markets_to_process = list(market_configs.values())
    
    # Add manual configs if any match the criteria
    for market_name, config in MANUAL_MARKET_CONFIGS.items():
        if not args.token or market_name.split('-')[0].upper() == args.token.upper():
            markets_to_process.append(config)
            logging.info(f"Added manual market config: {market_name}")
    
    if not markets_to_process:
        logging.error("No markets to process. Check your token symbol or pool_addresses.json file.")
        return
    
    # Process markets in parallel
    results = process_markets_threaded(markets_to_process, max_workers)
    
    # Log summary
    logging.info("--- Processing Summary ---")
    for result in results:
        logging.info(result)
    
    logging.info("OHLCV data fetching completed.")

if __name__ == "__main__":
    main() 