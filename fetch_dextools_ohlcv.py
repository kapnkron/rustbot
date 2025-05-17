import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import ast

API_KEY = "L8tBdVzmLNaVJfrmhlGr87efKnpkj1Ez4lO58nBi"
HEADERS = {"X-API-KEY": API_KEY}
BASE_URL = "https://public-api.dextools.io/trial"
CHAIN = "solana"
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

now = int(time.time())
six_months_ago = int((datetime.utcnow() - timedelta(days=180)).timestamp())
pool_params = {
    "sort": "creationTime",
    "order": "desc",
    "from": six_months_ago,
    "to": now
}

# 1. Extract unique (symbol, address) pairs from top token CSVs
top_lists = [
    "top_100_solana_gainers.csv",
    "top_100_solana_losers.csv"
]
tokens = set()
for fname in top_lists:
    if not os.path.exists(fname):
        continue
    df = pd.read_csv(fname)
    if 'data' in df.columns:
        for val in df['data'].dropna():
            try:
                d = ast.literal_eval(val)
                for key in ['mainToken', 'sideToken']:
                    tok = d.get(key)
                    if tok and 'symbol' in tok and 'address' in tok:
                        tokens.add((tok['symbol'].strip().upper(), tok['address']))
            except Exception:
                continue

tokens = list(tokens)
print(f"Found {len(tokens)} unique (symbol, address) pairs to process.")

# Error log setup
error_log_path = "fetch_dextools_ohlcv_errors.log"
with open(error_log_path, "w") as elog:
    elog.write("")  # Clear previous log

# Retry logic
max_retries = 3
retry_counts = {token: 0 for token in tokens}
remaining_tokens = tokens.copy()
failed_tokens = set()

while remaining_tokens:
    next_round = []
    for symbol, address in remaining_tokens:
        print(f"Processing {symbol} ({address})...")
        try:
            # Find pools for this token
            pools_url = f"{BASE_URL}/v2/token/{CHAIN}/{address}/pools"
            pools_resp = requests.get(pools_url, headers=HEADERS, params=pool_params)
            if pools_resp.status_code == 429:
                print(f"  Rate limited (429). Waiting 10 seconds and will retry later.")
                with open(error_log_path, "a") as elog:
                    elog.write(f"429 Rate limit for {symbol} ({address})\n")
                time.sleep(10)
                retry_counts[(symbol, address)] += 1
                if retry_counts[(symbol, address)] < max_retries:
                    next_round.append((symbol, address))
                else:
                    failed_tokens.add((symbol, address))
                continue
            time.sleep(1.1)
            if pools_resp.status_code != 200:
                print(f"  Failed to get pools: {pools_resp.status_code}")
                with open(error_log_path, "a") as elog:
                    elog.write(f"Failed to get pools for {symbol} ({address}): {pools_resp.status_code}\n")
                retry_counts[(symbol, address)] += 1
                if retry_counts[(symbol, address)] < max_retries:
                    next_round.append((symbol, address))
                else:
                    failed_tokens.add((symbol, address))
                continue
            pools = pools_resp.json().get("results", [])
            if not pools:
                print("  No pools found, skipping.")
                with open(error_log_path, "a") as elog:
                    elog.write(f"No pools found for {symbol} ({address})\n")
                # Not retrying for no pools
                continue
            # Pick the pool with highest liquidity
            pools_sorted = sorted(pools, key=lambda x: x.get("liquidity", 0), reverse=True)
            pool = pools_sorted[0]
            pool_address = pool["address"]
            # Try to fetch pool price history (if available)
            price_url = f"{BASE_URL}/v2/pool/{CHAIN}/{pool_address}/price"
            price_resp = requests.get(price_url, headers=HEADERS)
            if price_resp.status_code == 429:
                print(f"  Rate limited (429) on price. Waiting 10 seconds and will retry later.")
                with open(error_log_path, "a") as elog:
                    elog.write(f"429 Rate limit for {symbol} ({address}) on price\n")
                time.sleep(10)
                retry_counts[(symbol, address)] += 1
                if retry_counts[(symbol, address)] < max_retries:
                    next_round.append((symbol, address))
                else:
                    failed_tokens.add((symbol, address))
                continue
            time.sleep(1.1)
            if price_resp.status_code == 200:
                price_data = price_resp.json()
                # If price_data is a list of candles, save as OHLCV
                if isinstance(price_data, list) and price_data:
                    df = pd.DataFrame(price_data)
                    # Filter for last 6 months
                    if "timestamp" in df.columns:
                        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
                        cutoff = datetime.utcnow() - timedelta(days=180)
                        df = df[df["date"] >= cutoff]
                    df["address"] = address
                    out_path = os.path.join(RAW_DIR, f"{symbol}_usd_ohlcv.csv")
                    df.to_csv(out_path, index=False)
                    print(f"  Saved OHLCV to {out_path} ({len(df)} rows)")
                    continue
            # Fallback: fetch token price snapshot
            token_price_url = f"{BASE_URL}/v2/token/{CHAIN}/{address}/price"
            token_price_resp = requests.get(token_price_url, headers=HEADERS)
            if token_price_resp.status_code == 429:
                print(f"  Rate limited (429) on fallback price. Waiting 10 seconds and will retry later.")
                with open(error_log_path, "a") as elog:
                    elog.write(f"429 Rate limit for {symbol} ({address}) on fallback price\n")
                time.sleep(10)
                retry_counts[(symbol, address)] += 1
                if retry_counts[(symbol, address)] < max_retries:
                    next_round.append((symbol, address))
                else:
                    failed_tokens.add((symbol, address))
                continue
            time.sleep(1.1)
            if token_price_resp.status_code == 200:
                price_data = token_price_resp.json()
                df = pd.DataFrame([price_data])
                df["address"] = address
                out_path = os.path.join(RAW_DIR, f"{symbol}_usd_ohlcv.csv")
                df.to_csv(out_path, index=False)
                print(f"  Saved price snapshot to {out_path}")
            else:
                print(f"  Failed to fetch price for {symbol}")
                with open(error_log_path, "a") as elog:
                    elog.write(f"Failed to fetch price for {symbol} ({address}): {token_price_resp.status_code}\n")
                retry_counts[(symbol, address)] += 1
                if retry_counts[(symbol, address)] < max_retries:
                    next_round.append((symbol, address))
                else:
                    failed_tokens.add((symbol, address))
        except Exception as e:
            print(f"  Exception for {symbol}: {e}")
            with open(error_log_path, "a") as elog:
                elog.write(f"Exception for {symbol} ({address}): {e}\n")
            retry_counts[(symbol, address)] += 1
            if retry_counts[(symbol, address)] < max_retries:
                next_round.append((symbol, address))
            else:
                failed_tokens.add((symbol, address))
    if not next_round:
        break
    print(f"Retrying {len(next_round)} tokens that failed or were rate limited...")
    remaining_tokens = next_round

# Summary
if failed_tokens:
    print(f"\nFailed to fetch data for {len(failed_tokens)} tokens after {max_retries} attempts. See {error_log_path} for details.")
    with open(error_log_path, "a") as elog:
        elog.write(f"\nFailed tokens after {max_retries} attempts:\n")
        for symbol, address in failed_tokens:
            elog.write(f"{symbol} ({address})\n")
else:
    print("\nSuccessfully fetched data for all tokens.") 