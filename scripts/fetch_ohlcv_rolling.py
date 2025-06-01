import os
import sys
import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import argparse

TOKEN_MINTS_FILE = "token_mints.txt"
POOL_MAP_SCRIPT = "scripts/fetch_pool_addresses.py"
DATA_DIR = "data/processed/"
OHLCV_INTERVAL = 300  # 5 minutes in seconds
ROLLING_DAYS = 15
INITIAL_DAYS = 30  # Default to 30 days, can be overridden by CLI

HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
if not HELIUS_API_KEY:
    print("Error: HELIUS_API_KEY environment variable not set.")
    sys.exit(1)

HELIUS_URL = f"https://api.helius.xyz/v0/addresses/{{address}}/transactions?api-key={HELIUS_API_KEY}"

# Helper: Read token mints
def read_token_mints() -> List[str]:
    if not os.path.exists(TOKEN_MINTS_FILE):
        print(f"{TOKEN_MINTS_FILE} not found. Running fetch_top_solana_tokens.py to generate it...")
        import subprocess
        result = subprocess.run([sys.executable, "scripts/fetch_top_solana_tokens.py"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error running fetch_top_solana_tokens.py:", result.stderr)
            sys.exit(1)
        print(f"{TOKEN_MINTS_FILE} created.")
    with open(TOKEN_MINTS_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]

# Helper: Find most liquid pool (SOL, fallback USDC)
def get_pool_map() -> dict:
    import importlib.util
    import subprocess
    # Run the pool mapping script and capture output as JSON
    result = subprocess.run([sys.executable, POOL_MAP_SCRIPT, "--json"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running pool mapping script:", result.stderr)
        sys.exit(1)
    return eval(result.stdout)  # Should be a dict: {mint: {"base":..., "quote":..., "pool":...}}

# Helper: Fetch swap/trade events from Helius
def fetch_helius_events(mint: str, quote: str, pool: str, start_time: int, end_time: int) -> List[dict]:
    url = f"https://api.helius.xyz/v0/addresses/{pool}/transactions?api-key={HELIUS_API_KEY}&type=SWAP"
    limit = 100
    all_events = []
    before_signature = None
    while True:
        params = {"limit": limit}
        if before_signature:
            params["before"] = before_signature
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code != 200:
            print(f"Helius error for {mint}-{quote}: {resp.text}")
            break
        data = resp.json()
        if not data:
            break
        filtered = []
        for tx in data:
            ts = tx.get("timestamp")
            if not ts or not (start_time <= ts <= end_time):
                continue
            swap = tx.get("events", {}).get("swap")
            if swap:
                try:
                    if swap.get("tokenInputs") and swap.get("tokenOutputs"):
                        input_amt = float(swap["tokenInputs"][0].get("tokenAmount", 0))
                        output_amt = float(swap["tokenOutputs"][0].get("tokenAmount", 0))
                        if input_amt > 0 and output_amt > 0:
                            price = output_amt / input_amt
                            filtered.append({
                                "timestamp": ts,
                                "price": price,
                                "amount": input_amt
                            })
                except Exception as e:
                    print(f"DEBUG: Failed to parse swap event: {e}")
                    continue
            else:
                # Broader logic: infer swap from tokenTransfers
                token_transfers = tx.get("tokenTransfers", [])
                if len(token_transfers) == 2:
                    t0, t1 = token_transfers[0], token_transfers[1]
                    # Must be different mints
                    if t0["mint"] != t1["mint"]:
                        # One is input, one is output (direction depends on pool address)
                        # Try to infer input/output by pool address
                        if t0["toUserAccount"] == pool:
                            input_amt = float(t0["tokenAmount"])
                            input_mint = t0["mint"]
                        elif t1["toUserAccount"] == pool:
                            input_amt = float(t1["tokenAmount"])
                            input_mint = t1["mint"]
                        else:
                            continue
                        if t0["fromUserAccount"] == pool:
                            output_amt = float(t0["tokenAmount"])
                            output_mint = t0["mint"]
                        elif t1["fromUserAccount"] == pool:
                            output_amt = float(t1["tokenAmount"])
                            output_mint = t1["mint"]
                        else:
                            continue
                        # Only keep if input/output mints match base/quote
                        if {input_mint, output_mint} == {mint, quote} and input_amt > 0 and output_amt > 0:
                            price = output_amt / input_amt
                            filtered.append({
                                "timestamp": ts,
                                "price": price,
                                "amount": input_amt
                            })
                            if len(filtered) == 1 and not all_events:
                                print(f"DEBUG: First inferred swap event: {filtered[0]}")
        all_events.extend(filtered)
        print(f"DEBUG: Fetched {len(data)} tx, {len(filtered)} valid swap/inferred events in window, first ts: {data[0]['timestamp'] if data else 'n/a'}, last ts: {data[-1]['timestamp'] if data else 'n/a'}")
        if data[-1]["timestamp"] < start_time:
            break
        if len(data) < limit:
            break
        before_signature = data[-1]["signature"]
        time.sleep(0.2)
    return all_events

# Helper: Aggregate events to OHLCV
def aggregate_ohlcv(events: List[dict], interval: int) -> pd.DataFrame:
    # Assume each event has 'timestamp' (unix), 'price', 'amount'
    df = pd.DataFrame(events)
    if df.empty:
        return df
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    ohlcv = df['price'].resample(f'{interval}s').ohlc()
    ohlcv['volume'] = df['amount'].resample(f'{interval}s').sum()
    ohlcv = ohlcv.dropna()
    return ohlcv.reset_index()

# Helper: Save and prune OHLCV
def save_and_prune_ohlcv(df: pd.DataFrame, base: str, quote: str):
    fname = os.path.join(DATA_DIR, f"DATA_{base}_{quote}_OHLCV_5min.csv")
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=ROLLING_DAYS)
    if os.path.exists(fname):
        old = pd.read_csv(fname, parse_dates=['datetime'])
        df = pd.concat([old, df]).drop_duplicates(subset=['datetime']).sort_values('datetime')
    df = df[df['datetime'] >= cutoff]
    df.to_csv(fname, index=False)

# Main logic
def main():
    parser = argparse.ArgumentParser(description="Fetch and aggregate 5-min OHLCV data for all pools.")
    parser.add_argument('--initial-days', type=int, default=30, help='Number of days to fetch for new pools (default: 30)')
    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    mints = read_token_mints()
    pool_map = get_pool_map()
    print("DEBUG: Pool map:", pool_map)
    print("DEBUG: Token mints:", mints)
    now = int(time.time())
    for mint in mints:
        info = pool_map.get(mint)
        print(f"DEBUG: Processing mint {mint} with pool info: {info}")
        if not info:
            print(f"No pool found for {mint}, skipping.")
            continue
        base, quote, pool = info['base'], info['quote'], info['pool']
        fname = os.path.join(DATA_DIR, f"DATA_{base}_{quote}_OHLCV_5min.csv")
        # Determine start time
        if os.path.exists(fname):
            df = pd.read_csv(fname, parse_dates=['datetime'])
            last = df['datetime'].max()
            start_time = int(last.timestamp())
            period = ROLLING_DAYS * 86400
        else:
            start_time = now - args.initial_days * 86400
            period = args.initial_days * 86400
        end_time = now
        print(f"DEBUG: Fetching {base}-{quote} ({pool}) from {datetime.utcfromtimestamp(start_time)} to {datetime.utcfromtimestamp(end_time)}")
        try:
            events = fetch_helius_events(mint, quote, pool, start_time, end_time)
            print(f"DEBUG: Number of events fetched for {base}-{quote}: {len(events)}")
            if not events:
                print(f"No events for {base}-{quote} in this period.")
                continue
            ohlcv = aggregate_ohlcv(events, OHLCV_INTERVAL)
            print(f"DEBUG: Number of OHLCV rows for {base}-{quote}: {len(ohlcv)}")
            if ohlcv.empty:
                print(f"No OHLCV for {base}-{quote} in this period.")
                continue
            save_and_prune_ohlcv(ohlcv, base, quote)
            print(f"Saved {len(ohlcv)} OHLCV rows for {base}-{quote}.")
        except Exception as e:
            print(f"ERROR: Exception while processing {base}-{quote}: {e}")

if __name__ == "__main__":
    main() 