#!/usr/bin/env python3

# --- Monkey-patch for Solana signature bug in Python 3.12 ---
try:
    import solana.rpc.types
    def patched_to_rpc_config(self):
        config = solana.rpc.types.RpcSignaturesForAddressConfig(limit=getattr(self, 'limit', None))
        if hasattr(self, 'before') and self.before:
            config.before = self.before
        if hasattr(self, 'until') and self.until:
            config.until = self.until
        return config
    for name in dir(solana.rpc.types):
        obj = getattr(solana.rpc.types, name)
        if hasattr(obj, '_to_rpc_config'):
            obj._to_rpc_config = patched_to_rpc_config
except Exception as e:
    pass
# --- End monkey-patch ---

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
import traceback
import signal
import pickle
import time

# --- Environment Loading ---
load_dotenv()

# --- Configuration ---
ALCHEMY_API_KEY = os.environ.get("ALCHEMY_API_KEY", "").strip('\"')

if ALCHEMY_API_KEY:
    RPC_ENDPOINT = f"https://solana-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
else:
    logging.critical("ALCHEMY_API_KEY not set. Please set it in your environment. Exiting.")
    sys.exit(1)

# Alchemy API rate limiting and optimization
MAX_BATCH_SIZE = 25  # Maximum number of requests to batch in a single call
RATE_LIMIT_REQUESTS_PER_SECOND = 10  # Conservative rate limit
COMPUTE_UNITS_PER_DAY = 100_000_000  # Default for upgraded account
COMPUTE_UNITS_BUFFER = 0.85  # Use only 85% of available compute units for safety
COMPUTE_UNITS_USED = 0  # Track compute units used
compute_units_lock = threading.Lock()  # For thread-safe tracking of compute units

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

# --- Script Settings ---
CANDLE_INTERVAL_SECONDS = 300
REQUEST_DELAY_SECONDS = 0.1  # Reduced delay
MAX_DAILY_REQUESTS_LIMIT = 40000
MAX_SIGNATURES_TO_FETCH = 1000
MAX_TRANSACTIONS_PER_BATCH = 50
MAX_TX_TO_INSPECT = 10
INSPECTION_MODE = False
NUM_DAYS_TO_FETCH = 7
BATCH_DELAY_SECONDS = 0.2  # Reduced delay
EMPTY_RESPONSE_LIMIT = 5
MAX_WORKER_THREADS = 5  # Number of parallel threads

# --- File Paths ---
CHECKPOINT_FILE = "data/checkpoint_ohlcv.pkl"
PROCESSED_MARKETS_FILE = "data/processed_markets.json"
ERROR_LOG_FILE = "data/ohlcv_errors.log"
SIGNATURE_CACHE_FILE = "data/signature_cache.pkl"

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
checkpoint_lock = threading.Lock()
should_exit = False
rate_limit_lock = threading.Lock()
last_request_time = time.time()
signature_cache = {}  # Cache for signatures by address
signature_cache_lock = threading.Lock()

# --- Rate Limiting ---
def rate_limit_request():
    """Implement rate limiting for API requests"""
    global last_request_time
    
    with rate_limit_lock:
        current_time = time.time()
        time_since_last_request = current_time - last_request_time
        
        # Ensure we don't exceed the rate limit
        if time_since_last_request < (1.0 / RATE_LIMIT_REQUESTS_PER_SECOND):
            sleep_time = (1.0 / RATE_LIMIT_REQUESTS_PER_SECOND) - time_since_last_request
            time.sleep(max(0, sleep_time))
        
        # Update last request time
        last_request_time = time.time()

def track_compute_units(response):
    """Track compute units used from Alchemy API response"""
    global COMPUTE_UNITS_USED, should_exit
    
    # Try to extract compute units from response
    try:
        # For normal responses
        if hasattr(response, 'alchemy') and response.alchemy:
            compute_units = getattr(response.alchemy, 'compute_units', 0)
            if compute_units:
                with compute_units_lock:
                    COMPUTE_UNITS_USED += compute_units
                    
                    # Log compute units usage every 1M units
                    if COMPUTE_UNITS_USED % 1_000_000 < compute_units:
                        logging.info(f"Compute units used so far: {COMPUTE_UNITS_USED:,}")
                    
        # For JSON responses
        elif isinstance(response, dict):
            if 'alchemy' in response:
                compute_units = response.get('alchemy', {}).get('computeUnits', 0)
                if compute_units:
                    with compute_units_lock:
                        COMPUTE_UNITS_USED += compute_units
                        
                        # Log compute units usage every 1M units
                        if COMPUTE_UNITS_USED % 1_000_000 < compute_units:
                            logging.info(f"Compute units used so far: {COMPUTE_UNITS_USED:,}")
        
        # Check if we're approaching the daily limit
        with compute_units_lock:
            if COMPUTE_UNITS_USED > (COMPUTE_UNITS_PER_DAY * COMPUTE_UNITS_BUFFER):
                logging.warning(f"Approaching compute unit limit: {COMPUTE_UNITS_USED:,} / {COMPUTE_UNITS_PER_DAY:,}")
                if COMPUTE_UNITS_USED >= COMPUTE_UNITS_PER_DAY:
                    logging.critical("Compute unit limit reached! Exiting to prevent overage charges.")
                    should_exit = True
    except Exception as e:
        # Don't let this crash the main process
        logging.warning(f"Error tracking compute units: {str(e)}")
        pass

# --- Signature Cache Management ---
def load_signature_cache():
    """Load signature cache from file if it exists"""
    global signature_cache
    
    if os.path.exists(SIGNATURE_CACHE_FILE):
        try:
            with open(SIGNATURE_CACHE_FILE, 'rb') as f:
                signature_cache = pickle.load(f)
            logging.info(f"Loaded signature cache with {sum(len(sigs) for sigs in signature_cache.values())} signatures for {len(signature_cache)} addresses")
        except Exception as e:
            logging.error(f"Error loading signature cache: {str(e)}")
            signature_cache = {}

def save_signature_cache():
    """Save signature cache to file"""
    if signature_cache:
        try:
            os.makedirs(os.path.dirname(SIGNATURE_CACHE_FILE), exist_ok=True)
            with open(SIGNATURE_CACHE_FILE, 'wb') as f:
                pickle.dump(signature_cache, f)
            logging.info(f"Saved signature cache with {sum(len(sigs) for sigs in signature_cache.values())} signatures for {len(signature_cache)} addresses")
        except Exception as e:
            logging.error(f"Error saving signature cache: {str(e)}")

def get_cached_signatures(address, until=None):
    """Get cached signatures for an address, optionally until a certain signature"""
    with signature_cache_lock:
        if address in signature_cache:
            signatures = signature_cache[address]
            if until:
                # Return signatures until we reach the 'until' signature
                index = next((i for i, sig in enumerate(signatures) if sig == until), None)
                if index is not None:
                    return signatures[:index]
            return signatures
    return []

def add_to_signature_cache(address, signatures):
    """Add signatures to the cache for an address"""
    if not signatures:
        return
        
    with signature_cache_lock:
        if address not in signature_cache:
            signature_cache[address] = []
        
        # Add only new signatures to avoid duplicates
        existing = set(signature_cache[address])
        new_sigs = [sig for sig in signatures if sig not in existing]
        
        if new_sigs:
            signature_cache[address] = new_sigs + signature_cache[address]
            
            # Periodically save the cache (every 1000 new signatures)
            if len(new_sigs) > 1000:
                save_signature_cache()

# --- Batch Processing ---
def batch_get_transactions(client, signatures):
    """Get multiple transactions in a single batch request to reduce API calls"""
    if not signatures:
        return []
        
    # Split into batches of MAX_BATCH_SIZE
    results = []
    for i in range(0, len(signatures), MAX_BATCH_SIZE):
        batch = signatures[i:i+MAX_BATCH_SIZE]
        
        # Apply rate limiting
        rate_limit_request()
        
        try:
            # We'll fetch transactions one by one since batch isn't working properly
            # This is still better than the original approach
            for sig in batch:
                try:
                    # Get transaction with proper options
                    tx_response = client.get_transaction(
                        Signature.from_string(sig),
                        max_supported_transaction_version=0
                    )
                    
                    # Track compute units
                    track_compute_units(tx_response)
                    
                    # Add to results if valid
                    if tx_response.value:
                        results.append(tx_response.value)
                    
                    # Small delay between requests
                    time.sleep(0.1)
                except Exception as e:
                    logging.error(f"Error fetching transaction {sig}: {str(e)}")
        except Exception as e:
            logging.error(f"Error in batch transaction processing: {str(e)}")
        
        # Small delay between batches
        time.sleep(BATCH_DELAY_SECONDS)
    
    return results

# --- Checkpoint Management ---
def save_checkpoint(markets_to_process, processed_markets):
    """Save a checkpoint of progress that can be resumed later"""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    
    with checkpoint_lock:
        checkpoint_data = {
            'markets_to_process': markets_to_process,
            'processed_markets': processed_markets,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Also save a more human-readable version
        with open(PROCESSED_MARKETS_FILE, 'w') as f:
            json.dump({
                'processed_markets': [m['name'] for m in processed_markets],
                'markets_to_process': [m['name'] for m in markets_to_process],
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logging.info(f"Checkpoint saved: {len(processed_markets)} markets processed, {len(markets_to_process)} markets remaining")

def load_checkpoint():
    """Load the latest checkpoint if it exists"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                checkpoint_data = pickle.load(f)
                
            logging.info(f"Checkpoint loaded from {checkpoint_data['timestamp']}")
            logging.info(f"{len(checkpoint_data['processed_markets'])} markets already processed")
            logging.info(f"{len(checkpoint_data['markets_to_process'])} markets remaining to process")
            
            return checkpoint_data['markets_to_process'], checkpoint_data['processed_markets']
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            return None, []
    
    return None, []

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global should_exit
    logging.info("Received interrupt signal, finishing current tasks and exiting...")
    should_exit = True

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
    pool_quote_mint_str: str,
    start_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Fetch trade history for a market, using optimized batching and caching."""
    client = Client(RPC_ENDPOINT)
    trades = []
    
    # Create a set to keep track of processed signatures for this market
    current_market_processed_signatures = load_processed_signatures(processed_signatures_log_file)
    
    # Initialize variables for pagination
    before_signature_str = None
    empty_response_count = 0
    pubkey = Pubkey.from_string(market_id)
    
    # Determine until date based on input or default
    until_date = datetime.now(timezone.utc)
    from_date = until_date - timedelta(days=NUM_DAYS_TO_FETCH)
    
    # If start_date is provided, use it as the from_date
    if start_date is not None:
        # Make sure start_date has timezone info
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        from_date = start_date
        logging.info(f"Using specified start date: {from_date}")
    
    logging.info(f"Fetching trade history for {market_name} from {from_date} to {until_date}")
    
    # Check for cached signatures first
    cached_sigs = get_cached_signatures(market_id)
    if cached_sigs:
        logging.info(f"Found {len(cached_sigs)} cached signatures for {market_name}")
        
        # Filter signatures that are already processed
        new_signatures = [sig for sig in cached_sigs if sig not in current_market_processed_signatures]
        
        if new_signatures:
            logging.info(f"Processing {len(new_signatures)} new signatures from cache for {market_name}")
            
            # Batch process transactions for better efficiency
            batch_tx_data = batch_get_transactions(client, new_signatures)
            
            for tx_data in batch_tx_data:
                if should_exit:
                    break
                    
                try:
                    sig_str = tx_data['transaction']['signatures'][0]
                    
                    # Skip if already processed
                    if sig_str in current_market_processed_signatures:
                        continue
                        
                    # Process transaction
                    trade = process_single_transaction(
                        client,
                        Signature.from_string(sig_str),
                        market_name,
                        pool_base_mint_str,
                        pool_quote_mint_str,
                        processed_signatures_log_file,
                        current_market_processed_signatures
                    )
                    
                    if trade:
                        trades.append(trade)
                        
                except Exception as e:
                    logging.error(f"Error processing cached transaction for {market_name}: {str(e)}")
    
    # Continue with normal fetching if needed
    while not should_exit and empty_response_count < EMPTY_RESPONSE_LIMIT:
        try:
            # Apply rate limiting
            rate_limit_request()
            
            # Create proper config for getSignaturesForAddress
            if before_signature_str:
                try:
                    before_val = before_signature_str.strip()
                    logging.info(f"Calling get_signatures_for_address with before={before_val} (type: {type(before_val)}) and limit={MAX_SIGNATURES_TO_FETCH}")
                    signatures_response = client.get_signatures_for_address(
                        pubkey,
                        before=before_val,
                        limit=MAX_SIGNATURES_TO_FETCH
                    )
                except Exception as e:
                    logging.error(f"Error using before signature string: {e}")
                    logging.info(f"Falling back to fetching without 'before' parameter")
                    signatures_response = client.get_signatures_for_address(
                        pubkey,
                        limit=MAX_SIGNATURES_TO_FETCH
                    )
            else:
                logging.info(f"Calling get_signatures_for_address with no before, limit={MAX_SIGNATURES_TO_FETCH}")
                signatures_response = client.get_signatures_for_address(
                    pubkey,
                    limit=MAX_SIGNATURES_TO_FETCH
                )
            
            # Track compute units
            track_compute_units(signatures_response)
            
            signatures = signatures_response.value
            
            if not signatures:
                empty_response_count += 1
                logging.info(f"No signatures found for {market_name} (empty response count: {empty_response_count})")
                continue
                
            # Convert signature objects to strings
            signature_strs = []
            for sig_info in signatures:
                try:
                    # Try different methods to get the signature string
                    if hasattr(sig_info.signature, 'to_string'):
                        sig_str = sig_info.signature.to_string()
                    elif hasattr(sig_info.signature, '__str__'):
                        sig_str = str(sig_info.signature)
                    else:
                        # Fallback to using the default string representation
                        sig_str = f"{sig_info.signature}"
                    
                    signature_strs.append(sig_str)
                except Exception as e:
                    logging.warning(f"Could not convert signature to string: {e}")
            
            # Add to signature cache
            add_to_signature_cache(market_id, signature_strs)
                
            # Process in batches
            batch_signatures = []
            for i, sig_info in enumerate(signatures):
                if i >= len(signature_strs):
                    continue
                    
                sig_str = signature_strs[i]
                sig_time = datetime.fromtimestamp(sig_info.block_time, tz=timezone.utc)
                
                # Skip if already processed
                if sig_str in current_market_processed_signatures:
                    continue
                    
                # Skip if before our from_date
                if sig_time < from_date:
                    empty_response_count += 1
                    logging.info(f"Reached transactions before {from_date}, stopping for {market_name}")
                    break
                    
                # Add to batch
                batch_signatures.append(sig_str)
                
                # Update before_signature for next page
                if not before_signature_str or sig_str < before_signature_str:
                    before_signature_str = sig_str
            
            if not batch_signatures:
                empty_response_count += 1
                logging.info(f"No new signatures to process for {market_name} (empty response count: {empty_response_count})")
                continue
                
            # Batch process transactions
            batch_tx_data = batch_get_transactions(client, batch_signatures)
            
            for tx_data in batch_tx_data:
                if should_exit:
                    break
                    
                try:
                    sig_str = tx_data['transaction']['signatures'][0]
                    
                    # Process transaction
                    trade = process_single_transaction(
                        client,
                        Signature.from_string(sig_str),
                        market_name,
                        pool_base_mint_str,
                        pool_quote_mint_str,
                        processed_signatures_log_file,
                        current_market_processed_signatures
                    )
                    
                    if trade:
                        trades.append(trade)
                        
                except Exception as e:
                    logging.error(f"Error processing transaction for {market_name}: {str(e)}")
            
            # Break if we've reached the time limit or compute units limit
            if should_exit:
                break
                
        except Exception as e:
            logging.error(f"Error fetching signatures for {market_name}: {str(e)}")
            time.sleep(5)  # Wait before retry
    
    # Convert trades to DataFrame
    if trades:
        trades_df = pd.DataFrame(trades)
        logging.info(f"Fetched {len(trades_df)} trades for {market_name}")
        return trades_df
    else:
        logging.info(f"No trades found for {market_name}")
        return pd.DataFrame()

def process_single_transaction(
    client: Client,
    signature: Signature,
    market_name: str,
    pool_base_mint: str,
    pool_quote_mint: str,
    log_file: Path,
    processed_set: set[str]
) -> Optional[Dict[str, Any]]:
    """Process a single transaction to extract trade data if available."""
    sig_str = str(signature)
    
    # Skip if already processed
    if sig_str in processed_set:
        return None
    
    try:
        # The transaction data should already be fetched and passed in
        # But if signature is passed instead, we'll fetch it
        if isinstance(signature, Signature):
            # Fetch the transaction
            tx_data = client.get_transaction(
                signature,
                max_supported_transaction_version=0
            ).value
        else:
            # Assume it's already the transaction data
            tx_data = signature
            
        if not tx_data:
            return None
            
        # Extract block time based on the object type
        block_time = None
        
        # Handle different transaction object types
        if hasattr(tx_data, 'block_time'):
            # Direct access for standard types
            block_time = tx_data.block_time
        elif hasattr(tx_data, 'meta'):
            # For EncodedConfirmedTransactionWithStatusMeta objects
            meta = getattr(tx_data, 'meta', None)
            block_time = getattr(tx_data, 'block_time', None)
        
        # If still no block_time, use current time as fallback
        if block_time is None:
            block_time = int(time.time())
        
        timestamp = datetime.fromtimestamp(block_time, tz=timezone.utc)
        
        # Basic placeholder trade data
        # In a real implementation, you would extract actual price/volume from the transaction
        trade = {
            "signature": sig_str,
            "timestamp": timestamp,
            "price": 1.0,  # Placeholder
            "volume": 1.0,  # Placeholder
            "side": "buy"   # Placeholder
        }
        
        # Log the processed signature
        log_processed_signature(sig_str, log_file, processed_set)
        
        return trade
    except Exception as e:
        logging.error(f"Error processing transaction for {market_name}: {str(e)}")
        return None

def worker_thread(market_queue, results, processed_markets, progress_bar=None, days_to_fetch=NUM_DAYS_TO_FETCH):
    global should_exit
    client = Client(RPC_ENDPOINT)
    
    while not market_queue.empty() and not should_exit:
        try:
            market_config = market_queue.get_nowait()
        except queue.Empty:
            break
            
        try:
            # Pass days_to_fetch to fetch_and_save_market_data
            result = fetch_and_save_market_data(market_config, client, days_to_fetch)
            
            # Record successful processing
            with checkpoint_lock:
                processed_markets.append(market_config)
                
            # Save checkpoint periodically (every market)
            remaining_markets = list(market_queue.queue)
            save_checkpoint(remaining_markets, processed_markets)
            
            results.append(result)
            if progress_bar:
                progress_bar.update(1)
                
        except Exception as e:
            error_msg = f"Error processing market {market_config['name']}: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            
            # Log to error file
            with open(ERROR_LOG_FILE, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] Market: {market_config['name']}\n")
                f.write(error_msg + "\n\n")
            
            # Put the market back at the end of the queue for retry, unless we're exiting
            if not should_exit:
                market_queue.put(market_config)
                time.sleep(5)  # Wait before retry
        finally:
            market_queue.task_done()

def fetch_and_save_market_data(market_config: Dict[str, Any], client: Client = None, days_to_fetch: int = NUM_DAYS_TO_FETCH) -> str:
    """Modified to accept an existing client and handle thread safety, with days_to_fetch parameter"""
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
    
    # Check if we need to append to existing OHLCV file
    is_rolling_update = False
    existing_ohlcv_df = None
    start_date = None
    
    if os.path.exists(ohlcv_file):
        try:
            existing_ohlcv_df = pd.read_csv(ohlcv_file)
            # Ensure date column is datetime
            existing_ohlcv_df['date'] = pd.to_datetime(existing_ohlcv_df['date'])
            is_rolling_update = True
            
            # Get the latest date in existing data
            latest_date = existing_ohlcv_df['date'].max()
            
            # For rolling updates, start from the last date in the existing data
            # Subtract 1 day to ensure overlap for reconciliation
            start_date = latest_date - timedelta(days=1)
            
            logging.info(f"Found existing OHLCV data for {market_name} with latest date {latest_date}")
            logging.info(f"Will fetch data from {start_date} onwards to update")
        except Exception as e:
            logging.error(f"Error reading existing OHLCV file for {market_name}: {e}")
            logging.info(f"Will create a new OHLCV file for {market_name}")
            existing_ohlcv_df = None
    
    # Fetch and process trade history with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            trades_df = fetch_trade_history_for_market(
                market_id=market_id,
                market_name=market_name,
                processed_signatures_log_file=processed_log_file,
                pool_base_mint_str=base_mint,
                pool_quote_mint_str=quote_mint,
                start_date=start_date
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Attempt {attempt+1} failed for {market_name}: {str(e)}. Retrying...")
                time.sleep(5)  # Wait before retry
            else:
                logging.error(f"All {max_retries} attempts failed for {market_name}")
                raise
    
    if not trades_df.empty:
        # Save raw trades
        if is_rolling_update and os.path.exists(output_file):
            # Append to existing trades file
            old_trades_df = pd.read_csv(output_file)
            # Ensure no duplicates
            trades_df = pd.concat([old_trades_df, trades_df]).drop_duplicates(subset=['signature'])
            
        trades_df.to_csv(output_file, index=False)
        logging.info(f"Saved {len(trades_df)} trades to {output_file}")
        
        # Aggregate to OHLCV and save
        new_ohlcv_df = aggregate_trades_to_ohlcv(trades_df, CANDLE_INTERVAL_SECONDS, market_name)
        
        # Standardize column names: rename the timestamp index to 'date' for consistent naming
        new_ohlcv_df.index.name = 'date'
        
        # If this is a rolling update, merge with existing data
        if is_rolling_update and existing_ohlcv_df is not None:
            # Convert index to column for easier merging
            new_ohlcv_df = new_ohlcv_df.reset_index()
            
            # Standardize column names to avoid merge issues
            for df in [existing_ohlcv_df, new_ohlcv_df]:
                if 'datetime' in df.columns and 'date' not in df.columns:
                    df.rename(columns={'datetime': 'date'}, inplace=True)
                # Ensure date is in datetime format
                df['date'] = pd.to_datetime(df['date'])
            
            # Combine new and existing data
            combined_df = pd.concat([existing_ohlcv_df, new_ohlcv_df])
            
            # Remove duplicates, keeping the newest version if dates overlap
            combined_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
            
            # Sort by date
            combined_df = combined_df.sort_values('date')
            
            # Save the combined data
            combined_df.to_csv(ohlcv_file, index=False)
            logging.info(f"Updated OHLCV data with {len(new_ohlcv_df)} new candles for {market_name}")
            return f"Updated with {len(new_ohlcv_df)} new candles for {market_name}"
        else:
            # Save new data
            new_ohlcv_df.to_csv(ohlcv_file)
            logging.info(f"Saved new OHLCV data with standardized column 'date' to {ohlcv_file}")
            return f"Processed {len(trades_df)} trades for {market_name}"
    else:
        if is_rolling_update:
            logging.info(f"No new trades found for {market_name}, existing data maintained")
            return f"No new trades found for {market_name}"
        else:
            logging.info(f"No trades found for {market_name}")
            # Create an empty file to mark this market as processed
            with open(ohlcv_file, 'w') as f:
                f.write("date,open,high,low,close,volume,trade_count\n")
            return f"No trades found for {market_name}"

def process_markets_threaded(markets_to_process: List[Dict[str, Any]], processed_markets: List[Dict[str, Any]], max_workers: int = MAX_WORKER_THREADS, days_to_fetch: int = NUM_DAYS_TO_FETCH):
    """Process multiple markets in parallel using threads with checkpointing"""
    global should_exit
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
            args=(market_queue, results, processed_markets, progress_bar, days_to_fetch)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete or until interrupted
    try:
        for thread in threads:
            while thread.is_alive() and not should_exit:
                thread.join(timeout=1.0)
            if should_exit:
                break
    except KeyboardInterrupt:
        logging.info("User interrupted, saving checkpoint and exiting...")
        should_exit = True
    
    progress_bar.close()
    
    # Save final checkpoint
    remaining_markets = list(market_queue.queue)
    save_checkpoint(remaining_markets, processed_markets)
    
    return results

def main():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Global declarations need to come before any use of the variables
    global COMPUTE_UNITS_USED
    global should_exit
    global COMPUTE_UNITS_PER_DAY
    
    parser = argparse.ArgumentParser(description='Fetch historical OHLCV data from Solana')
    parser.add_argument('--days', type=int, default=NUM_DAYS_TO_FETCH,
                      help=f'Number of days of historical data to fetch (default: {NUM_DAYS_TO_FETCH})')
    parser.add_argument('--pool', type=str, help='Specific pool ID to fetch (will look in pool_addresses.json)')
    parser.add_argument('--threads', type=int, default=MAX_WORKER_THREADS,
                      help=f'Number of threads to use (default: {MAX_WORKER_THREADS})')
    parser.add_argument('--reset', action='store_true', 
                      help='Reset checkpoint and start fresh')
    parser.add_argument('--rolling', action='store_true',
                      help='Perform a rolling update (fetch only from the latest date in existing data)')
    parser.add_argument('--compute-units', type=int, default=COMPUTE_UNITS_PER_DAY,
                      help=f'Maximum compute units to use (default: {COMPUTE_UNITS_PER_DAY})')
    args = parser.parse_args()
    
    # Use local variable instead of modifying global
    days_to_fetch = args.days
    max_workers = args.threads
    
    # Update compute units limit from argument
    COMPUTE_UNITS_PER_DAY = args.compute_units
    logging.info(f"Compute units limit set to {COMPUTE_UNITS_PER_DAY:,}")
    
    # If rolling update is enabled, we'll get the date range from existing data
    # This overrides any days value passed
    is_rolling_update = args.rolling
    
    # Set up directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Load signature cache to improve efficiency
    load_signature_cache()
    
    # Initialize error log
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(f"\n--- New run started at {datetime.now().isoformat()} ---\n\n")
    
    # Check for checkpoint and resume if it exists
    remaining_markets, processed_markets = None, []
    
    if not args.reset:
        remaining_markets, processed_markets = load_checkpoint()
    
    # If no checkpoint or reset requested, start fresh
    if remaining_markets is None:
        # Load market configurations
        markets_to_process = []
        pool_addresses_file = Path("pool_addresses.json")
        
        if args.pool:
            target_pool_id = args.pool
            logging.info(f"Fetching data for specific pool ID: {target_pool_id}")
            
            # Load all market configurations
            market_configs = load_market_configs_from_json(pool_addresses_file)
            
            # Find the market with matching pool ID
            for market_name, config in market_configs.items():
                if config.get("id") == target_pool_id:
                    markets_to_process.append(config)
                    logging.info(f"Added pool {market_name} with ID {target_pool_id}")
                    break
                    
            if not markets_to_process:
                logging.error(f"Pool ID {target_pool_id} not found in configurations. Check your pool_addresses.json file.")
                return
        else:
            # Process all markets
            logging.info(f"Fetching data for all pools in pool_addresses.json")
            market_configs = load_market_configs_from_json(pool_addresses_file)
            markets_to_process = list(market_configs.values())
        
        if not markets_to_process:
            logging.error("No markets to process. Check your pool_addresses.json file.")
            return
        
        # Save initial checkpoint
        save_checkpoint(markets_to_process, [])
        remaining_markets = markets_to_process
    
    # Process markets in parallel with resume capability
    try:
        if is_rolling_update:
            logging.info("Performing rolling update using dates from existing data")
            # We don't pass days_to_fetch because the individual market processing
            # will use the dates from existing data
            results = process_markets_threaded(remaining_markets, processed_markets, max_workers)
        else:
            logging.info(f"Fetching historical data for the past {days_to_fetch} days")
            results = process_markets_threaded(remaining_markets, processed_markets, max_workers, days_to_fetch)
        
        # Log summary
        logging.info("--- Processing Summary ---")
        for result in results:
            logging.info(result)
        
        if should_exit:
            logging.info("Process was interrupted. Run the script again to resume from checkpoint.")
        else:
            logging.info("OHLCV data fetching completed successfully for all markets.")
            # Clear checkpoint since we're done
            if os.path.exists(CHECKPOINT_FILE):
                os.rename(CHECKPOINT_FILE, f"{CHECKPOINT_FILE}.completed")
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}\n{traceback.format_exc()}")
        logging.info("Run the script again to resume from the last checkpoint.")
    finally:
        # Save signature cache before exiting
        save_signature_cache()
        
        # Log final compute units usage
        logging.info(f"Final compute units used: {COMPUTE_UNITS_USED:,}")

if __name__ == "__main__":
    main() 