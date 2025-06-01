import requests
import os
import subprocess
import time
import sys
import json
from typing import List, Dict

SOLANA_TRACKER_API_KEY = "ef4f484a-538c-42ac-92c9-3158d267f7e6"
BASE_URL = "https://data.solanatracker.io"
TOKEN_MINTS_FILE = "token_mints.txt"
FETCH_TOP_TOKENS_SCRIPT = "fetch_top_solana_tokens.py"
MIN_LIQUIDITY_USD = 10_000
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
HEADERS = {"x-api-key": SOLANA_TRACKER_API_KEY}
SLEEP_BETWEEN_REQUESTS = 1.5

def load_token_mints() -> List[str]:
    if os.path.exists(TOKEN_MINTS_FILE):
        print(f"Loading token mints from {TOKEN_MINTS_FILE}...", file=sys.stderr)
        with open(TOKEN_MINTS_FILE, "r") as f:
            mints = [line.strip() for line in f if line.strip()]
        return mints
    else:
        print(f"{TOKEN_MINTS_FILE} not found. Running {FETCH_TOP_TOKENS_SCRIPT} to generate it...", file=sys.stderr)
        result = subprocess.run(["python3", FETCH_TOP_TOKENS_SCRIPT], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to generate {TOKEN_MINTS_FILE}. Error:\n{result.stderr}", file=sys.stderr)
            exit(1)
        print(f"Created {TOKEN_MINTS_FILE} automatically.", file=sys.stderr)
        with open(TOKEN_MINTS_FILE, "r") as f:
            mints = [line.strip() for line in f if line.strip()]
        return mints

def has_viable_pool(token_mint: str, dex_filter="Raydium"):
    url = f"{BASE_URL}/tokens/{token_mint}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch pools for {token_mint}: {e}", file=sys.stderr)
        return None
    pools = data.get("pools", [])
    best_pool = None
    best_liquidity = -1
    for pool in pools:
        if pool.get("dex") != dex_filter:
            continue
        liquidity = pool.get("liquidity", {}).get("usd")
        try:
            liquidity = float(liquidity)
        except (TypeError, ValueError):
            continue
        if liquidity < MIN_LIQUIDITY_USD:
            continue
        quote_token = pool.get("quoteToken")
        base_token = pool.get("tokenAddress")
        if quote_token in [SOL_MINT, USDC_MINT]:
            if liquidity > best_liquidity:
                best_liquidity = liquidity
                best_pool = {
                    "base": base_token,
                    "quote": quote_token,
                    "pool": pool.get("poolId")
                }
    return best_pool

def main():
    output_json = "--json" in sys.argv
    token_mints = load_token_mints()
    if not output_json:
        print(f"Checking {len(token_mints)} tokens for viable Raydium pools on Solana Tracker...", file=sys.stderr)
    viable_mints = []
    pool_map = {}
    for i, mint in enumerate(token_mints):
        if not output_json:
            print(f"[{i+1}/{len(token_mints)}] Checking {mint}...", file=sys.stderr)
        pool_info = has_viable_pool(mint, dex_filter="Raydium")
        if pool_info:
            if not output_json:
                print(f"  -> Viable Raydium pool found.", file=sys.stderr)
            viable_mints.append(mint)
            pool_map[mint] = pool_info
        else:
            if not output_json:
                print(f"  -> No viable Raydium pool.", file=sys.stderr)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    if output_json:
        print(json.dumps(pool_map, indent=2))
    else:
        with open(TOKEN_MINTS_FILE, "w") as f:
            for mint in viable_mints:
                f.write(mint + "\n")
        print(f"\nFiltered to {len(viable_mints)} tokens with at least one Raydium SOL or USDC pool and >$10k liquidity. Overwrote {TOKEN_MINTS_FILE}.", file=sys.stderr)

if __name__ == "__main__":
    main() 