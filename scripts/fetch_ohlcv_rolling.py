import os
import sys
import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

TOKEN_MINTS_FILE = "token_mints.txt"
POOL_MAP_SCRIPT = "fetch_pool_addresses.py"
DATA_DIR = "data/processed/"
OHLCV_INTERVAL = 300  # 5 minutes in seconds
ROLLING_DAYS = 15
INITIAL_DAYS = 60

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
    # Helius enhanced API: filter by program, pool, and time
    # This is a placeholder; you may need to adjust endpoint/params for your Helius plan
    url = f"https://api.helius.xyz/v0/addresses/{pool}/transactions?api-key={HELIUS_API_KEY}"
    params = {
        "before": end_time,
        "after": start_time,
        "limit": 1000
    }
    events = []
    while True:
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code != 200:
            print(f"Helius error for {mint}-{quote}: {resp.text}")
            break
        data = resp.json()
        if not data:
            break
        events.extend(data)
        if len(data) < 1000:
            break
        params["before"] = min(e["timestamp"] for e in data)
        time.sleep(0.2)
    return events

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
    os.makedirs(DATA_DIR, exist_ok=True)
    mints = read_token_mints()
    pool_map = get_pool_map()
    now = int(time.time())
    for mint in mints:
        info = pool_map.get(mint)
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
            start_time = now - INITIAL_DAYS * 86400
            period = INITIAL_DAYS * 86400
        end_time = now
        print(f"Fetching {base}-{quote} ({pool}) from {datetime.utcfromtimestamp(start_time)} to {datetime.utcfromtimestamp(end_time)}")
        events = fetch_helius_events(mint, quote, pool, start_time, end_time)
        if not events:
            print(f"No events for {base}-{quote} in this period.")
            continue
        ohlcv = aggregate_ohlcv(events, OHLCV_INTERVAL)
        if ohlcv.empty:
            print(f"No OHLCV for {base}-{quote} in this period.")
            continue
        save_and_prune_ohlcv(ohlcv, base, quote)
        print(f"Saved {len(ohlcv)} OHLCV rows for {base}-{quote}.")

if __name__ == "__main__":
    main() 