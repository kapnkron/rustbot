import sys
print(f"Python executable: {sys.executable}")
print(f"Python sys.path: {sys.path}")

import time
import pandas as pd
from solana.rpc.api import Client
from solders.pubkey import Pubkey
from solders.signature import Signature # Import Signature from solders
from datetime import datetime, timezone, timedelta
import logging
import os
import json # For pretty printing transaction details
import requests # For fetching Raydium API
from typing import Dict, Any, Optional, Tuple, List # Added for type hinting

# --- Configuration ---
RPC_ENDPOINT = os.getenv("SOLANA_RPC_ENDPOINT", "https://mainnet.helius-rpc.com/?api-key=YOUR_API_KEY_HERE") 
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY") # Ensure HELIUS_API_KEY is set in your environment
HELIUS_TRANSACTIONS_API_URL = "https://api.helius.xyz/v0/addresses"

USDC_MINT_STR = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" # Remains relevant for some pairs
RAYDIUM_LIQUIDITY_API_URL = "https://api.raydium.io/v2/sdk/liquidity/mainnet.json"
JUP_MINT_STR = "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"

# --- Market Configurations ---
# Each dict should have 'id' (market ID string) and 'name' (e.g., "SOL-USDC")
MARKET_CONFIGS = [
    {"id": "8BnEgHoWFysVcuFFX7QztDmzuH8r5ZFvyP3sYwn1XTh6", 
     "name": "SOL-USDC",
     "base_mint": "So11111111111111111111111111111111111111112", # WSOL
     "quote_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", # USDC
     "base_decimals": 9,
     "quote_decimals": 6
    },
    {"id": "C1MgLojNLWBKADvu9BHdtgzz1oZX4dZ5zGdGcgvvW8Wz", 
     "name": "JUP-SOL-Orca",
     "base_mint": JUP_MINT_STR, # JUP is base
     "quote_mint": "So11111111111111111111111111111111111111112", # WSOL is quote
     "base_decimals": 6, # JUP has 6 decimals
     "quote_decimals": 9 # SOL has 9 decimals
    },
]

# --- General Script Settings ---
CANDLE_INTERVAL_SECONDS = 300  # 5 minutes
REQUEST_DELAY_SECONDS = 0.02  # Delay between Helius API requests
MAX_DAILY_REQUESTS_LIMIT = 900000 
MAX_TX_TO_INSPECT = 10 # Inspect 10 transactions
INSPECTION_MODE = False # Set to True to inspect a few txs for a *single* market (see main block)
NUM_DAYS_TO_FETCH = 1 # Test with 1 day

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("data_fetcher.log"), # General log for the script run
                        logging.StreamHandler()
                    ])

# --- Global State (for the entire script run) ---
request_count_today = 0 # Tracks total API calls for the session
last_reset_date = datetime.now(timezone.utc).date()
processed_signatures_global_set_for_run = set() # To avoid reprocessing a sig if it appears for multiple markets (unlikely but a safeguard)


# --- Type Hinting for Market Details ---
class MarketDetails(Dict[str, Any]):
    pool_base_mint: Pubkey
    pool_quote_mint: Pubkey
    pool_base_vault: Pubkey
    pool_quote_vault: Pubkey
    pool_base_decimals: int
    pool_quote_decimals: int

# --- Utility Functions ---
def get_program_account_public_key(program_id_str: str) -> Pubkey:
    cleaned_id_str = program_id_str.strip()
    return Pubkey.from_string(cleaned_id_str)

def load_processed_signatures(processed_signatures_log_file: str) -> set[Signature]:
    """Loads already processed signatures from a market-specific log file."""
    processed_market_signatures = set()
    if os.path.exists(processed_signatures_log_file):
        with open(processed_signatures_log_file, 'r') as f:
            valid_signatures = set()
            for line_num, line in enumerate(f):
                stripped_line = line.strip()
                if stripped_line:
                    try:
                        valid_signatures.add(Signature.from_string(stripped_line))
                    except ValueError:
                        logging.warning(f"Skipping invalid signature string in {processed_signatures_log_file} at line {line_num+1}: '{stripped_line}'")
            processed_market_signatures = valid_signatures
        logging.info(f"Loaded {len(processed_market_signatures)} processed signatures from {processed_signatures_log_file}.")
    return processed_market_signatures

def log_processed_signature(signature: Signature, processed_signatures_log_file: str, current_market_processed_set: set[Signature]):
    """Logs a signature as processed to the market-specific file and updates sets."""
    current_market_processed_set.add(signature)
    processed_signatures_global_set_for_run.add(signature) # Add to global set as well
    with open(processed_signatures_log_file, 'a') as f:
        f.write(str(signature) + '\n')

def check_and_reset_daily_request_count():
    """Resets the daily request counter if a new day has started."""
    global request_count_today, last_reset_date
    current_date = datetime.now(timezone.utc).date()
    if current_date > last_reset_date:
        logging.info(f"New day detected. Resetting daily request count from {request_count_today} to 0.")
        request_count_today = 0
        last_reset_date = current_date

def fetch_market_details_from_api(market_id_str: str, market_name: str) -> Optional[MarketDetails]:
    logging.info(f"[{market_name}] Fetching market details for {market_id_str} from Raydium API: {RAYDIUM_LIQUIDITY_API_URL}")
    temp_file_path = f"temp_raydium_api_response_{market_name}.json" # Market-specific temp file
    
    market_details_obj: MarketDetails = {} # type: ignore

    try:
        response = requests.get(RAYDIUM_LIQUIDITY_API_URL, timeout=(60, 1800), stream=True) 
        response.raise_for_status()

        logging.info(f"[{market_name}] Streaming Raydium API response to {temp_file_path}")
        downloaded_size = 0
        with open(temp_file_path, 'wb') as f:
            for chunk_idx, chunk in enumerate(response.iter_content(chunk_size=8192)):
                f.write(chunk)
                downloaded_size += len(chunk)
                if (chunk_idx + 1) % 1000 == 0:
                    logging.info(f"[{market_name}] Downloaded {downloaded_size / (1024*1024):.2f} MB so far...")
        logging.info(f"[{market_name}] Finished streaming Raydium API response. Total size: {downloaded_size / (1024*1024):.2f} MB to {temp_file_path}")

        logging.info(f"[{market_name}] Attempting to parse JSON from {temp_file_path}")
        with open(temp_file_path, 'r') as f:
            liquidity_data = json.load(f)
        logging.info(f"[{market_name}] Successfully parsed JSON from file.")
        
        try:
            os.remove(temp_file_path)
            logging.info(f"[{market_name}] Removed temporary file {temp_file_path}")
        except OSError as e:
            logging.warning(f"[{market_name}] Error removing temporary file {temp_file_path}: {e}")

        logging.debug(f"[{market_name}] Starting search for market ID {market_id_str} in Raydium API response.")
        market_info = None
        for pool_list_type in ['official', 'unOfficial']:
            logging.debug(f"[{market_name}] Checking pool list type: {pool_list_type}")
            if pool_list_type in liquidity_data:
                current_list_size = len(liquidity_data[pool_list_type])
                logging.debug(f"[{market_name}]  List '{pool_list_type}' found with {current_list_size} pools.")
                pool_counter = 0
                for pool in liquidity_data[pool_list_type]:
                    pool_counter += 1
                    if pool_counter % 500 == 0: 
                        logging.debug(f"[{market_name}]    Processing pool {pool_counter}/{current_list_size} in '{pool_list_type}'...")
                    
                    if pool.get('id') == market_id_str or \
                       pool.get('ammId') == market_id_str or \
                       pool.get('marketId') == market_id_str: 
                        market_info = pool
                        logging.debug(f"[{market_name}]  Market ID {market_id_str} found in '{pool_list_type}' list.")
                        break
            else:
                logging.debug(f"[{market_name}]  List type '{pool_list_type}' not found in API response.")
            if market_info:
                break
        
        logging.debug(f"[{market_name}] Finished search for market ID {market_id_str}.")

        if market_info:
            logging.info(f"[{market_name}] Found market details for {market_id_str}: {market_info}")
            market_details_obj['pool_base_mint'] = Pubkey.from_string(market_info['baseMint'])
            market_details_obj['pool_quote_mint'] = Pubkey.from_string(market_info['quoteMint'])
            market_details_obj['pool_base_vault'] = Pubkey.from_string(market_info['baseVault'])
            market_details_obj['pool_quote_vault'] = Pubkey.from_string(market_info['quoteVault'])
            market_details_obj['pool_base_decimals'] = int(market_info['baseDecimals'])
            market_details_obj['pool_quote_decimals'] = int(market_info['quoteDecimals'])
            
            logging.info(f"[{market_name}]  Base Mint: {market_details_obj['pool_base_mint']}, Decimals: {market_details_obj['pool_base_decimals']}")
            logging.info(f"[{market_name}]  Quote Mint: {market_details_obj['pool_quote_mint']}, Decimals: {market_details_obj['pool_quote_decimals']}")
            return market_details_obj
        else:
            logging.error(f"[{market_name}] Market ID {market_id_str} not found in Raydium API response.")
            return None
            
    except requests.exceptions.RequestException as e:
        logging.error(f"[{market_name}] Error fetching Raydium liquidity API for {market_id_str}: {e}")
        return None
    except KeyError as e:
        logging.error(f"[{market_name}] Error parsing market details for {market_id_str} from API response (missing key {e}). Market Info: {market_info if 'market_info' in locals() else 'Not Found'}")
        return None
    except Exception as e:
        logging.error(f"[{market_name}] Unexpected error fetching/parsing market details for {market_id_str}: {e}")
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e_del:
                logging.warning(f"[{market_name}] Error removing temporary file {temp_file_path} after error: {e_del}")
        return None

def inspect_trade_history_for_market(market_id_str: str, market_name: str, market_details: MarketDetails):
    global request_count_today
    logging.info(f"[{market_name}] Starting Helius Enhanced API inspection for Market ID: {market_id_str}")
    market_pk = get_program_account_public_key(market_id_str)
    
    params = { "api-key": HELIUS_API_KEY, "limit": MAX_TX_TO_INSPECT }
    request_url = f"{HELIUS_TRANSACTIONS_API_URL}/{str(market_pk)}/transactions"
    
    try:
        logging.debug(f"[{market_name}] Requesting Helius for inspection: {request_url} with params: {{'limit': {MAX_TX_TO_INSPECT} }}")
        response = requests.get(request_url, params=params, timeout=30)
        request_count_today +=1
        response.raise_for_status()
        transactions = response.json()

        if not transactions:
            logging.info(f"[{market_name}] No transactions found for inspection.")
            return

        logging.info(f"[{market_name}] Fetched {len(transactions)} transactions for inspection:")
        for i, tx in enumerate(transactions):
            logging.info(f"[{market_name}] --- Transaction {i+1}/{len(transactions)} (Signature: {tx.get('signature', 'N/A')}) ---")
            logging.info(json.dumps(tx, indent=2))
            tx_dt = datetime.fromtimestamp(tx['timestamp'], timezone.utc) if tx.get('timestamp') else "N/A"
            logging.info(f"  Timestamp: {tx_dt}")
            if tx.get('events') and tx['events'].get('swap'):
                swap_event = tx['events']['swap']
                logging.info(f"  Swap Event: {swap_event}")
            logging.info(f"[{market_name}] --- End Transaction ---")
            if i + 1 >= MAX_TX_TO_INSPECT: break
        
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"[{market_name}] Helius API HTTP error during inspection: {http_err} - Response: {http_err.response.text if http_err.response else 'N/A'}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"[{market_name}] Helius API Request error during inspection: {req_err}")
    except Exception as e:
        logging.exception(f"[{market_name}] Unexpected error during Helius inspection: {e}")

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

def fetch_trade_history_for_market(
    market_id_pk: Pubkey, 
    market_name: str, 
    market_details: MarketDetails, 
    start_time_dt: datetime, 
    end_time_dt: datetime,
    processed_market_signatures: set[Signature],
    processed_signatures_log_file: str
    ):
    global request_count_today
    logging.info(f"[{market_name}] Starting Helius Enhanced API trade fetch for market {str(market_id_pk)} from {start_time_dt} to {end_time_dt}")
    all_trades = []
    
    pool_base_mint_str = str(market_details['pool_base_mint'])
    pool_quote_mint_str = str(market_details['pool_quote_mint'])
    pool_base_decimals = market_details['pool_base_decimals']
    pool_quote_decimals = market_details['pool_quote_decimals']

    current_before_signature = None
    processed_in_this_run_count = 0
    consecutive_empty_responses = 0
    MAX_CONSECUTIVE_EMPTY_RESPONSES = 3

    while True:
        check_and_reset_daily_request_count()
        if request_count_today >= MAX_DAILY_REQUESTS_LIMIT:
            logging.warning(f"[{market_name}] Reached daily request limit. Stopping fetch for this market.")
            break

        params = {"api-key": HELIUS_API_KEY, "limit": 100}
        if current_before_signature:
            params["before"] = current_before_signature
        
        request_url = f"{HELIUS_TRANSACTIONS_API_URL}/{str(market_id_pk)}/transactions"
        
        try:
            log_params = {k: v for k,v in params.items() if k != 'api-key'}
            logging.debug(f"[{market_name}] Requesting Helius: {request_url} with params: {log_params}")
            response = requests.get(request_url, params=params, timeout=60)
            request_count_today += 1
            
            transactions = []
            if response.status_code != 200:
                error_text = response.text
                try:
                    error_data = response.json()
                    if "error" in error_data and "Failed to find events within the search period" in error_data["error"]:
                        logging.info(f"[{market_name}] Helius: {error_data['error']} (End of history). Stopping.")
                    else:
                        logging.error(f"[{market_name}] Helius API non-200 response: {response.status_code} - {error_text}")
                        response.raise_for_status()
                except json.JSONDecodeError: 
                    logging.error(f"[{market_name}] Helius API non-200 response (not JSON): {response.status_code} - {error_text}")
                    response.raise_for_status()
            else:
                 transactions = response.json()

            if not transactions:
                logging.info(f"[{market_name}] No transactions from Helius API (url: {request_url} {'before: '+current_before_signature if current_before_signature else ''}).")
                consecutive_empty_responses +=1
                if consecutive_empty_responses >= MAX_CONSECUTIVE_EMPTY_RESPONSES or not current_before_signature:
                    logging.warning(f"[{market_name}] Assuming end of relevant data for this market after {consecutive_empty_responses} empty responses or first call empty.")
                    break
                time.sleep(REQUEST_DELAY_SECONDS * 5)
                continue
            
            consecutive_empty_responses = 0
            logging.info(f"[{market_name}] Fetched {len(transactions)} txs. Oldest sig: {transactions[-1].get('signature', 'N/A')}, Oldest ts: {datetime.fromtimestamp(transactions[-1]['timestamp'], timezone.utc) if transactions[-1].get('timestamp') else 'N/A'}")
            current_before_signature = transactions[-1].get('signature')
            if not current_before_signature:
                 logging.error(f"[{market_name}] Last tx in batch has no signature. Stopping.")
                 break
            
            batch_had_relevant_transactions = False
            for tx_idx, tx_data in enumerate(transactions):
                tx_sig_str = tx_data.get('signature')
                if not tx_sig_str:
                    logging.warning(f"[{market_name}] Tx at index {tx_idx} missing 'signature'. Skipping. Data: {tx_data}")
                    continue
                tx_sig = Signature.from_string(tx_sig_str)
                if tx_sig in processed_market_signatures or tx_sig in processed_signatures_global_set_for_run:
                    logging.debug(f"[{market_name}] Skipping already processed sig: {tx_sig_str}")
                    continue
                if 'timestamp' not in tx_data:
                    logging.warning(f"[{market_name}] Tx {tx_sig_str} missing 'timestamp'. Skipping.")
                    continue
                tx_dt = datetime.fromtimestamp(tx_data['timestamp'], timezone.utc)

                if tx_dt < start_time_dt:
                    logging.info(f"[{market_name}] Tx {tx_sig_str} at {tx_dt} is older than start_time {start_time_dt}. Stopping.")
                    transactions = [] # Signal to break outer loop
                    break 
                if tx_dt > end_time_dt:
                    logging.debug(f"[{market_name}] Tx {tx_sig_str} at {tx_dt} newer than end_time {end_time_dt}. Skipping.")
                    continue
                
                batch_had_relevant_transactions = True
                # Initialize amounts and mints before attempting any parsing method
                amount_in, amount_out, mint_in, mint_out = 0.0, 0.0, None, None
                # Flag to indicate if trade was successfully parsed by any method
                trade_parsed_successfully = False
                parsed_from_description = False # Reset for current transaction

                if 'description' in tx_data and tx_data['description']: # Ensure description is not empty
                    try:
                        parts = tx_data['description'].lower().split(" swapped ")
                        if len(parts) == 2 and " for " in parts[1]:
                            asset_parts = parts[1].split(" for ")
                            amount_in_str, mint_in_symbol = asset_parts[0].strip().split(" ", 1)
                            amount_out_str, mint_out_symbol = asset_parts[1].strip().split(" ", 1)
                            
                            # Simplified symbol to mint mapping for brevity
                            # Important: This needs robust mapping or to be combined with pool mints
                            temp_mint_in, temp_mint_out = None, None
                            if mint_in_symbol == "sol": temp_mint_in = "So11111111111111111111111111111111111111112"
                            elif mint_in_symbol == "usdc": temp_mint_in = USDC_MINT_STR
                            elif mint_in_symbol == "jup": temp_mint_in = JUP_MINT_STR
                            # else check against pool_base_mint_str / pool_quote_mint_str by symbol match

                            if mint_out_symbol == "sol": temp_mint_out = "So11111111111111111111111111111111111111112"
                            elif mint_out_symbol == "usdc": temp_mint_out = USDC_MINT_STR
                            elif mint_out_symbol == "jup": temp_mint_out = JUP_MINT_STR
                            # else check against pool_base_mint_str / pool_quote_mint_str by symbol match

                            # Actual logic should determine mint_in/mint_out based on pool_base_mint_str and pool_quote_mint_str
                            # and ensure the parsed mints are one of them.
                            # This is a placeholder for your existing, more detailed description parsing.
                            # For this example, let's assume if temp_mint_in/out are found, they are valid for the pool.
                            
                            # Placeholder: For the actual script, you'd map these symbols to the specific
                            # pool_base_mint_str and pool_quote_mint_str to determine the actual amounts
                            # and ensure they align with the current market context.
                            # This simplified version just uses the symbols directly if matched.
                            if temp_mint_in == pool_base_mint_str and temp_mint_out == pool_quote_mint_str:
                                amount_in, mint_in = float(amount_in_str), pool_base_mint_str
                                amount_out, mint_out = float(amount_out_str), pool_quote_mint_str
                                parsed_from_description = True
                            elif temp_mint_in == pool_quote_mint_str and temp_mint_out == pool_base_mint_str:
                                amount_in, mint_in = float(amount_in_str), pool_quote_mint_str
                                amount_out, mint_out = float(amount_out_str), pool_base_mint_str
                                parsed_from_description = True
                            
                            if parsed_from_description:
                                logging.debug(f"[{market_name}] Parsed from desc: {amount_in_str} {mint_in_symbol} for {amount_out_str} {mint_out_symbol}")
                                trade_parsed_successfully = True

                    except Exception as e:
                        logging.debug(f"[{market_name}] Could not parse desc for {tx_sig_str}: '{tx_data['description']}'. Err: {e}. Fallback.")
                        parsed_from_description = False # Ensure it's reset on error
                        trade_parsed_successfully = False


                if not trade_parsed_successfully:
                    swap_event = tx_data.get('events', {}).get('swap')
                    if swap_event: # Primary parsing path if swap_event exists
                        found_in_inner_swap = False
                        if swap_event.get('innerSwaps'):
                            for inner_swap_idx, inner_swap in enumerate(swap_event.get('innerSwaps', [])):
                                inner_base_raw_chg, inner_quote_raw_chg = 0, 0
                                if inner_swap.get('nativeInput') and inner_swap['nativeInput'].get('amount'):
                                    if pool_base_mint_str == "So11111111111111111111111111111111111111112": inner_base_raw_chg += int(inner_swap['nativeInput']['amount'])
                                    elif pool_quote_mint_str == "So11111111111111111111111111111111111111112": inner_quote_raw_chg += int(inner_swap['nativeInput']['amount'])
                                if inner_swap.get('nativeOutput') and inner_swap['nativeOutput'].get('amount'):
                                    if pool_base_mint_str == "So11111111111111111111111111111111111111112": inner_base_raw_chg -= int(inner_swap['nativeOutput']['amount'])
                                    elif pool_quote_mint_str == "So11111111111111111111111111111111111111112": inner_quote_raw_chg -= int(inner_swap['nativeOutput']['amount'])
                                for ti in inner_swap.get('tokenInputs', []):
                                    if ti.get('mint') == pool_base_mint_str and ti.get('rawTokenAmount', {}).get('tokenAmount'): inner_base_raw_chg += int(ti['rawTokenAmount']['tokenAmount'])
                                    elif ti.get('mint') == pool_quote_mint_str and ti.get('rawTokenAmount', {}).get('tokenAmount'): inner_quote_raw_chg += int(ti['rawTokenAmount']['tokenAmount'])
                                for to_ in inner_swap.get('tokenOutputs', []): # Renamed to avoid conflict
                                    if to_.get('mint') == pool_base_mint_str and to_.get('rawTokenAmount', {}).get('tokenAmount'): inner_base_raw_chg -= int(to_['rawTokenAmount']['tokenAmount'])
                                    elif to_.get('mint') == pool_quote_mint_str and to_.get('rawTokenAmount', {}).get('tokenAmount'): inner_quote_raw_chg -= int(to_['rawTokenAmount']['tokenAmount'])

                                if inner_base_raw_chg != 0 and inner_quote_raw_chg != 0:
                                    if inner_base_raw_chg > 0 and inner_quote_raw_chg < 0:
                                        # Base IN, Quote OUT
                                        # amount_in refers to what the swapper provided, amount_out is what they received
                                        # Here, swapper provided QUOTE, received BASE
                                        amount_out, mint_out = inner_base_raw_chg / (10**pool_base_decimals), pool_base_mint_str
                                        amount_in, mint_in = abs(inner_quote_raw_chg) / (10**pool_quote_decimals), pool_quote_mint_str
                                        found_in_inner_swap = True
                                        trade_parsed_successfully = True
                                        logging.debug(f"[{market_name}] Tx {tx_sig_str} (innerSwap #{inner_swap_idx+1}) {mint_in_shorthand(mint_in)}_IN/{mint_out_shorthand(mint_out)}_OUT: {amount_in} for {amount_out}")
                                        break
                                    elif inner_base_raw_chg < 0 and inner_quote_raw_chg > 0:
                                        # Base OUT, Quote IN
                                        # Swapper provided BASE, received QUOTE
                                        amount_in, mint_in = abs(inner_base_raw_chg) / (10**pool_base_decimals), pool_base_mint_str
                                        amount_out, mint_out = inner_quote_raw_chg / (10**pool_quote_decimals), pool_quote_mint_str
                                        found_in_inner_swap = True
                                        trade_parsed_successfully = True
                                        logging.debug(f"[{market_name}] Tx {tx_sig_str} (innerSwap #{inner_swap_idx+1}) {mint_in_shorthand(mint_in)}_IN/{mint_out_shorthand(mint_out)}_OUT: {amount_in} for {amount_out}")
                                        break
                        
                        if not found_in_inner_swap: # If not in innerSwaps, try top-level of this swap_event
                            base_raw_chg, quote_raw_chg = 0, 0
                            if swap_event.get('nativeInput') and swap_event['nativeInput'].get('amount'):
                                if pool_base_mint_str == "So11111111111111111111111111111111111111112": base_raw_chg += int(swap_event['nativeInput']['amount'])
                                elif pool_quote_mint_str == "So11111111111111111111111111111111111111112": quote_raw_chg += int(swap_event['nativeInput']['amount'])
                            if swap_event.get('nativeOutput') and swap_event['nativeOutput'].get('amount'):
                                if pool_base_mint_str == "So11111111111111111111111111111111111111112": base_raw_chg -= int(swap_event['nativeOutput']['amount'])
                                elif pool_quote_mint_str == "So11111111111111111111111111111111111111112": quote_raw_chg -= int(swap_event['nativeOutput']['amount'])
                            for ti in swap_event.get('tokenInputs', []):
                                if ti.get('mint') == pool_base_mint_str and ti.get('rawTokenAmount', {}).get('tokenAmount'): base_raw_chg += int(ti['rawTokenAmount']['tokenAmount'])
                                elif ti.get('mint') == pool_quote_mint_str and ti.get('rawTokenAmount', {}).get('tokenAmount'): quote_raw_chg += int(ti['rawTokenAmount']['tokenAmount'])
                            for to_ in swap_event.get('tokenOutputs', []): # Renamed to avoid conflict
                                if to_.get('mint') == pool_base_mint_str and to_.get('rawTokenAmount', {}).get('tokenAmount'): base_raw_chg -= int(to_['rawTokenAmount']['tokenAmount'])
                                elif to_.get('mint') == pool_quote_mint_str and to_.get('rawTokenAmount', {}).get('tokenAmount'): quote_raw_chg -= int(to_['rawTokenAmount']['tokenAmount'])

                            # Determine amounts based on who received what.
                            # If base_raw_chg > 0, user received base. They must have paid with quote.
                            if base_raw_chg > 0 and quote_raw_chg < 0: # User received base, paid with quote
                                amount_out, mint_out = base_raw_chg / (10**pool_base_decimals), pool_base_mint_str
                                amount_in, mint_in = abs(quote_raw_chg) / (10**pool_quote_decimals), pool_quote_mint_str
                                trade_parsed_successfully = True
                            elif base_raw_chg < 0 and quote_raw_chg > 0: # User received quote, paid with base
                                amount_in, mint_in = abs(base_raw_chg) / (10**pool_base_decimals), pool_base_mint_str
                                amount_out, mint_out = quote_raw_chg / (10**pool_quote_decimals), pool_quote_mint_str
                                trade_parsed_successfully = True
                
                # Fallback: If not parsed by description OR (swap_event was missing OR swap_event parsing failed),
                # AND source is ORCA/RAYDIUM, then try accountData.
                if not trade_parsed_successfully and tx_data.get("source") in ["ORCA", "RAYDIUM"]:
                    if not tx_data.get('events', {}).get('swap'): # Log if events.swap was actually missing
                        logging.debug(f"[{market_name}] Tx {tx_sig_str} missing 'events.swap' (Source: {tx_data.get('source')}). Attempting accountData fallback.")
                    else: # Log if events.swap was present but unparseable by above logic
                        logging.debug(f"[{market_name}] Tx {tx_sig_str} 'events.swap' present but unparseable (Source: {tx_data.get('source')}). Attempting accountData fallback.")

                    user_net_changes = {} # Key: user_account, Value: {'base_change': 0, 'quote_change': 0, 'involved_mints': set()}
                    for ad_idx, ad in enumerate(tx_data.get("accountData", [])):
                        user_account = ad.get("account")
                        if not user_account: continue
                        if user_account not in user_net_changes:
                            user_net_changes[user_account] = {'base_change': 0, 'quote_change': 0, 'involved_mints': set()}
                        native_change = ad.get("nativeBalanceChange", 0)
                        if pool_base_mint_str == "So11111111111111111111111111111111111111112":
                            user_net_changes[user_account]['base_change'] += native_change
                            if native_change != 0: user_net_changes[user_account]['involved_mints'].add("So11111111111111111111111111111111111111112")
                        elif pool_quote_mint_str == "So11111111111111111111111111111111111111112":
                            user_net_changes[user_account]['quote_change'] += native_change
                            if native_change != 0: user_net_changes[user_account]['involved_mints'].add("So11111111111111111111111111111111111111112")
                        for tbc_idx, tbc in enumerate(ad.get("tokenBalanceChanges", [])):
                            mint = tbc.get("mint")
                            raw_amount_str = tbc.get("rawTokenAmount", {}).get("tokenAmount")
                            if not raw_amount_str: continue
                            raw_amount = int(raw_amount_str)
                            if mint == pool_base_mint_str:
                                user_net_changes[user_account]['base_change'] += raw_amount
                                if raw_amount != 0: user_net_changes[user_account]['involved_mints'].add(pool_base_mint_str)
                            elif mint == pool_quote_mint_str:
                                user_net_changes[user_account]['quote_change'] += raw_amount
                                if raw_amount != 0: user_net_changes[user_account]['involved_mints'].add(pool_quote_mint_str)
                    potential_swaps = []
                    for acc, changes in user_net_changes.items():
                        if pool_base_mint_str not in changes['involved_mints'] or pool_quote_mint_str not in changes['involved_mints']:
                            continue
                        bc, qc = changes['base_change'], changes['quote_change']
                        if (bc > 0 and qc < 0) or (bc < 0 and qc > 0):
                            potential_swaps.append({'account': acc, 'base_change': bc, 'quote_change': qc})
                    if len(potential_swaps) == 1:
                        swapper_data = potential_swaps[0]
                        final_base_change = swapper_data['base_change']
                        final_quote_change = swapper_data['quote_change']
                        if final_base_change > 0 and final_quote_change < 0: # User received Base, spent Quote
                            amount_out, mint_out = final_base_change / (10**pool_base_decimals), pool_base_mint_str
                            amount_in, mint_in = abs(final_quote_change) / (10**pool_quote_decimals), pool_quote_mint_str
                            trade_parsed_successfully = True
                        elif final_base_change < 0 and final_quote_change > 0: # User spent Base, received Quote
                            amount_in, mint_in = abs(final_base_change) / (10**pool_base_decimals), pool_base_mint_str
                            amount_out, mint_out = final_quote_change / (10**pool_quote_decimals), pool_quote_mint_str
                            trade_parsed_successfully = True
                        if trade_parsed_successfully:
                            logging.info(f"[{market_name}] Tx {tx_sig_str} (accountData fallback) {mint_in_shorthand(mint_in)}_IN/{mint_out_shorthand(mint_out)}_OUT: {amount_in} for {amount_out} by {swapper_data['account']}")
                    elif len(potential_swaps) > 1:
                        logging.debug(f"[{market_name}] Tx {tx_sig_str} (accountData fallback) found multiple potential swapper accounts: {potential_swaps}. Skipping due to ambiguity.")
                
                if not trade_parsed_successfully:
                    # Determine the most relevant part of the event to log if parsing failed
                    event_info_for_log = tx_data.get('description', 'No description')
                    if tx_data.get('events', {}).get('swap'):
                        event_info_for_log = tx_data['events']['swap']
                    elif not event_info_for_log: # if description was empty and no swap event
                         event_info_for_log = f"Type: {tx_data.get('type', 'N/A')}, Source: {tx_data.get('source', 'N/A')}"

                    logging.warning(f"[{market_name}] Indeterminable trade for {tx_sig_str} after all parsing attempts. Details: {event_info_for_log}. Full Tx JSON in DEBUG if enabled. Skip.")
                    logging.debug(f"[{market_name}] Full JSON for unparsed tx {tx_sig_str}: {json.dumps(tx_data, indent=1)}") # Log full JSON at DEBUG
                    continue

                if not (amount_in > 0 and amount_out > 0 and mint_in and mint_out):
                    logging.warning(f"[{market_name}] Invalid amounts/mints post-parsing for {tx_sig_str}. In: {amount_in} {mint_in}, Out: {amount_out} {mint_out}. Skip.")
                    continue

                price, volume_base_currency = 0.0, 0.0
                if mint_in == pool_base_mint_str and mint_out == pool_quote_mint_str:
                    price, volume_base_currency = amount_out / amount_in, amount_in
                elif mint_in == pool_quote_mint_str and mint_out == pool_base_mint_str:
                    price, volume_base_currency = amount_in / amount_out, amount_out
                else:
                    logging.warning(f"[{market_name}] Trade mints ({mint_in}, {mint_out}) != pool mints ({pool_base_mint_str}, {pool_quote_mint_str}) for {tx_sig_str}. Skip.")
                    continue
                
                if price <= 0 or volume_base_currency <= 0:
                    logging.warning(f"[{market_name}] Invalid price ({price}) or volume ({volume_base_currency}) for {tx_sig_str}. Skip.")
                    continue

                all_trades.append({'timestamp': tx_dt, 'price': price, 'volume': volume_base_currency, 'signature': tx_sig_str})
                log_processed_signature(tx_sig, processed_signatures_log_file, processed_market_signatures)
                processed_in_this_run_count += 1
            
            if not transactions: # Broke from inner loop due to age
                logging.info(f"[{market_name}] Reached tx older than start_time. Halting fetch for this market.")
                break
            if batch_had_relevant_transactions: consecutive_empty_responses = 0
            logging.info(f"[{market_name}] Processed {processed_in_this_run_count} new trades for this market so far. Last sig: {current_before_signature}")
            time.sleep(REQUEST_DELAY_SECONDS)

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"[{market_name}] Helius HTTP err: {http_err} - URL: {request_url} - Resp: {http_err.response.text if http_err.response else 'N/A'}")
            if hasattr(http_err, 'response') and http_err.response and http_err.response.status_code == 429:
                logging.warning(f"[{market_name}] Rate limited. Sleeping 60s.")
                time.sleep(60)
            else: time.sleep(10)
        except requests.exceptions.RequestException as req_err:
            logging.error(f"[{market_name}] Helius Req err: {req_err} - URL: {request_url}. Sleeping 20s.")
            time.sleep(20)
        except json.JSONDecodeError as json_err:
            logging.error(f"[{market_name}] JSON Decode err: {json_err} - URL: {request_url} - Resp: {response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'}. Sleeping 10s.")
            time.sleep(10)
        except Exception as e:
            logging.exception(f"[{market_name}] Unexpected err (URL: {request_url}): {e}. Sleeping 10s.")
            time.sleep(10)
            
    logging.info(f"[{market_name}] Finished Helius trade fetch. Total trades collected in this run for this market: {processed_in_this_run_count}")
    return pd.DataFrame(all_trades)

def find_jup_usdc_raydium_pools():
    logging.info("--- Starting JUP/USDC Raydium Pool Discovery ---")
    temp_file_path = "temp_raydium_liquidity_discovery.json"
    
    try:
        logging.info(f"Fetching Raydium liquidity data from {RAYDIUM_LIQUIDITY_API_URL}")
        response = requests.get(RAYDIUM_LIQUIDITY_API_URL, timeout=(60, 1800), stream=True) 
        response.raise_for_status()

        logging.info(f"Streaming Raydium API response to {temp_file_path}")
        downloaded_size = 0
        with open(temp_file_path, 'wb') as f:
            for chunk_idx, chunk in enumerate(response.iter_content(chunk_size=8192)):
                f.write(chunk)
                downloaded_size += len(chunk)
                if (chunk_idx + 1) % 1000 == 0:
                    logging.info(f"Downloaded {downloaded_size / (1024*1024):.2f} MB so far...")
        logging.info(f"Finished streaming Raydium API response. Total size: {downloaded_size / (1024*1024):.2f} MB to {temp_file_path}")

        logging.info(f"Attempting to parse JSON from {temp_file_path}")
        with open(temp_file_path, 'r') as f:
            liquidity_data = json.load(f)
        logging.info("Successfully parsed JSON from file.")
        
        try:
            os.remove(temp_file_path)
            logging.info(f"Removed temporary file {temp_file_path}")
        except OSError as e:
            logging.warning(f"Error removing temporary file {temp_file_path}: {e}")

        found_pools = []
        for pool_list_type in ['official', 'unOfficial']:
            if pool_list_type in liquidity_data:
                logging.info(f"Searching in '{pool_list_type}' list...")
                for pool in liquidity_data[pool_list_type]:
                    base_mint = pool.get('baseMint')
                    quote_mint = pool.get('quoteMint')
                    
                    is_jup_usdc_pair = (base_mint == JUP_MINT_STR and quote_mint == USDC_MINT_STR) or \
                                       (base_mint == USDC_MINT_STR and quote_mint == JUP_MINT_STR)
                    
                    if is_jup_usdc_pair:
                        pool_details = {
                            "id": pool.get("id"),
                            "ammId": pool.get("ammId"),
                            "lpMint": pool.get("lpMint"),
                            "baseMint": base_mint,
                            "quoteMint": quote_mint,
                            "marketId": pool.get("marketId"), # This is often the Openbook market ID used by the AMM
                            "liquidity": pool.get("liquidity"),
                            "volume24h": pool.get("volume24h"),
                            "lpPrice": pool.get("lpPrice"),
                            "pool_list_type": pool_list_type
                        }
                        found_pools.append(pool_details)
                        logging.info(f"Found JUP/USDC Pool in '{pool_list_type}':")
                        for key, value in pool_details.items():
                            logging.info(f"  {key}: {value}")
            else:
                logging.info(f"List type '{pool_list_type}' not found in API response.")
        
        if not found_pools:
            logging.info("No JUP/USDC Raydium pools found in the liquidity data.")
        else:
            logging.info(f"Found {len(found_pools)} potential JUP/USDC Raydium pool(s). Details logged above.")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Raydium liquidity API: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during JUP/USDC pool discovery: {e}")
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e_del:
                logging.warning(f"Error removing temporary file {temp_file_path} after error: {e_del}")
    logging.info("--- Finished JUP/USDC Raydium Pool Discovery ---")


# --- Main Execution ---
if __name__ == "__main__":
    logging.info(f"Script starting for {len(MARKET_CONFIGS)} market(s). Default fetch: {NUM_DAYS_TO_FETCH} days.")
    
    if not HELIUS_API_KEY:
        logging.error("CRITICAL: HELIUS_API_KEY environment variable not set. Please set it and rerun.")
        sys.exit(1)
    
    # Initialize Solana client (used for Pubkey conversions, not RPC calls here)
    try:
        client = Client(RPC_ENDPOINT) 
        logging.info(f"Successfully initialized Solana client (primarily for Pubkey utils) with endpoint: {RPC_ENDPOINT if RPC_ENDPOINT else 'Default'}")
    except Exception as e:
        logging.warning(f"Failed to initialize Solana client with {RPC_ENDPOINT}: {e}. Script will continue if client is not strictly needed for Helius calls.")
        client = None 

    overall_start_time_dt = datetime.now(timezone.utc) - timedelta(days=NUM_DAYS_TO_FETCH)
    overall_end_time_dt = datetime.now(timezone.utc)
    
    logging.info(f"Global data fetch window: {overall_start_time_dt.isoformat()} to {overall_end_time_dt.isoformat()}")

    # Corrected main processing loop starts here
    # # --- TEMPORARY MODIFICATION FOR DISCOVERY --- (Ensure this is commented out for normal runs)
    # find_jup_usdc_raydium_pools() 
    # logging.info("Discovery run complete. Exiting.")
    # sys.exit(0) # Exit after discovery
    # # --- END TEMPORARY MODIFICATION ---

    for market_config in MARKET_CONFIGS:
        market_id_str = market_config["id"]
        market_name = market_config["name"]
        
        logging.info(f"[{market_name}] Processing market ID: {market_id_str}")

        market_details = None # Initialize market_details

        # Check if all necessary details are predefined in market_config
        if (market_config.get("base_mint") and 
            market_config.get("quote_mint") and
            market_config.get("base_decimals") is not None and # Check for None explicitly
            market_config.get("quote_decimals") is not None): # Check for None explicitly
            
            logging.info(f"[{market_name}] Using fully predefined market details from MARKET_CONFIGS.")
            market_details = { # type: ignore
                'pool_base_mint': Pubkey.from_string(market_config["base_mint"]),
                'pool_quote_mint': Pubkey.from_string(market_config["quote_mint"]),
                'pool_base_decimals': int(market_config["base_decimals"]),
                'pool_quote_decimals': int(market_config["quote_decimals"]),
                'pool_base_vault': Pubkey.from_string(market_config.get("base_vault", "11111111111111111111111111111111")), 
                'pool_quote_vault': Pubkey.from_string(market_config.get("quote_vault", "11111111111111111111111111111111"))
            }
        else:
            # If not fully predefined, attempt to fetch from API
            logging.info(f"[{market_name}] Details not fully predefined in MARKET_CONFIGS. Attempting to fetch from API...")
            market_details = fetch_market_details_from_api(market_id_str, market_name)
            
            # Fallback if API fails AND partial predefinition exists (e.g. mints but no decimals)
            # This is a secondary fallback. The primary is the direct use of fully predefined details above.
            if not market_details and market_config.get("base_mint") and market_config.get("quote_mint"):
                 logging.warning(f"[{market_name}] API fetch failed. Falling back to any predefined mints/decimals. Completeness check will follow.")
                 # Construct with whatever is available, decimals might be missing if not in config
                 market_details = { # type: ignore
                    'pool_base_mint': Pubkey.from_string(market_config["base_mint"]),
                    'pool_quote_mint': Pubkey.from_string(market_config["quote_mint"]),
                    'pool_base_decimals': int(market_config.get("base_decimals")) if market_config.get("base_decimals") is not None else None,
                    'pool_quote_decimals': int(market_config.get("quote_decimals")) if market_config.get("quote_decimals") is not None else None,
                    'pool_base_vault': Pubkey.from_string(market_config.get("base_vault", "11111111111111111111111111111111")), 
                    'pool_quote_vault': Pubkey.from_string(market_config.get("quote_vault", "11111111111111111111111111111111"))
                 }

        # Critical check for valid and complete market_details after attempting all methods
        if not market_details or \
           not market_details.get('pool_base_mint') or \
           not market_details.get('pool_quote_mint') or \
           market_details.get('pool_base_decimals') is None or \
           market_details.get('pool_quote_decimals') is None:
            logging.error(f"[{market_name}] CRITICAL: Could not obtain valid and complete market details (mints & decimals are essential). Skipping this market. Details found: {market_details}")
            continue 

        # Sanitize market_name for filename
        safe_market_name_for_file = market_name.replace('/', '_').replace('-', '_')
        processed_signatures_log_file = f"processed_signatures_{safe_market_name_for_file}_{market_id_str}.log"
        current_market_processed_signatures = load_processed_signatures(processed_signatures_log_file)

        if INSPECTION_MODE:
            # Logic to ensure inspection only happens for the *first* market in the list if multiple share a name or if user wants to target one.
            # For precise inspection, configure only that one market in MARKET_CONFIGS.
            is_first_market_in_config = (market_config == MARKET_CONFIGS[0])
            if is_first_market_in_config:
                 logging.info(f"[{market_name}] INSPECTION MODE: Inspecting a few transactions for {market_id_str} as it's the first in config (or only one)...")
                 inspect_trade_history_for_market(market_id_str, market_name, market_details)
                 logging.info(f"[{market_name}] INSPECTION MODE: Finished inspection for {market_id_str}. Script will now exit due to inspection mode for the first market.")
                 sys.exit(0) # Exit after inspecting the first market in inspection mode
            else:
                logging.info(f"[{market_name}] INSPECTION MODE: Skipping inspection for {market_id_str} as it is not the first market in config.")
                continue

        logging.info(f"[{market_name}] Fetching trade history...")
        trades_df = fetch_trade_history_for_market(
            get_program_account_public_key(market_id_str),
            market_name,
            market_details,
            overall_start_time_dt,
            overall_end_time_dt,
            current_market_processed_signatures,
            processed_signatures_log_file
        )

        if not trades_df.empty:
            logging.info(f"[{market_name}] Fetched {len(trades_df)} new trades.")
            # Add general metadata (once, correctly)
            trades_df['market_id_fetched_from'] = market_id_str
            trades_df['market_name_config'] = market_name
            trades_df['script_CANDLE_INTERVAL_SECONDS'] = CANDLE_INTERVAL_SECONDS
            trades_df['script_NUM_DAYS_TO_FETCH'] = NUM_DAYS_TO_FETCH

            logging.info(f"[{market_name}] Aggregating {len(trades_df)} trades to OHLCV candles ({CANDLE_INTERVAL_SECONDS}s interval)...")
            ohlcv_df = aggregate_trades_to_ohlcv(trades_df, CANDLE_INTERVAL_SECONDS, market_name)

            if not ohlcv_df.empty:
                output_filename = f"DATA_{safe_market_name_for_file}_OHLCV_{CANDLE_INTERVAL_SECONDS // 60}min.csv"
                logging.info(f"[{market_name}] Preparing to save/update OHLCV data to {output_filename}")
                combined_ohlcv_df = ohlcv_df # Default to new data

                if os.path.exists(output_filename):
                    logging.info(f"[{market_name}] Found existing data file: {output_filename}. Loading to append/update...")
                    try:
                        existing_ohlcv_df = pd.read_csv(output_filename, index_col='timestamp', parse_dates=True)
                        if existing_ohlcv_df.index.tz is None:
                            existing_ohlcv_df.index = existing_ohlcv_df.index.tz_localize('UTC')
                        elif existing_ohlcv_df.index.tz != ohlcv_df.index.tz:
                            existing_ohlcv_df.index = existing_ohlcv_df.index.tz_convert(ohlcv_df.index.tz)
                        
                        logging.info(f"[{market_name}] Appending {len(ohlcv_df)} new/updated candles to existing {len(existing_ohlcv_df)} candles.")
                        combined_ohlcv_df = pd.concat([existing_ohlcv_df, ohlcv_df])
                        combined_ohlcv_df = combined_ohlcv_df[~combined_ohlcv_df.index.duplicated(keep='last')]
                    except Exception as e:
                        logging.error(f"[{market_name}] Error reading or processing existing CSV {output_filename}: {e}. Will overwrite with new data from this run.")
                        # combined_ohlcv_df is already ohlcv_df (new data)
                else:
                    logging.info(f"[{market_name}] No existing data file found at {output_filename}. Creating new file.")
                
                combined_ohlcv_df.sort_index(inplace=True)
                logging.info(f"[{market_name}] Saving {len(combined_ohlcv_df)} total OHLCV candles to {output_filename}")
                combined_ohlcv_df.to_csv(output_filename)
                logging.info(f"[{market_name}] Successfully saved OHLCV data to {output_filename}")
            else:
                logging.info(f"[{market_name}] No OHLCV candles generated from {len(trades_df)} new trades. CSV file not updated.")
        else:
            logging.info(f"[{market_name}] No new trades fetched in the specified timeframe. CSV file not updated.")
            
    logging.info("Script finished processing all configured markets.")

# This is the explicit end of the if __name__ == "__main__": block.
# No other code should follow this in the script.