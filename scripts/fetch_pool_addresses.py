import requests
import os
import subprocess
import time
import sys
import json
from typing import List, Dict
from pathlib import Path

# Determine Project Root assuming this script is in a 'scripts' subdirectory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SOLANA_TRACKER_API_KEY = "ef4f484a-538c-42ac-92c9-3158d267f7e6"
BASE_URL = "https://data.solanatracker.io"
TOKEN_MINTS_FILE = PROJECT_ROOT / "token_mints.txt"
FETCH_TOP_TOKENS_SCRIPT = SCRIPT_DIR / "fetch_top_solana_tokens.py"
POOL_ADDRESSES_FILE = PROJECT_ROOT / "pool_addresses.json"
MIN_LIQUIDITY_USD = 10000
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
QUOTE_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
HEADERS = {"x-api-key": SOLANA_TRACKER_API_KEY}
SLEEP_BETWEEN_REQUESTS = 1.5

def load_token_mints() -> List[str]:
    if TOKEN_MINTS_FILE.exists():
        print(f"Loading token mints from {TOKEN_MINTS_FILE}...", file=sys.stderr)
        with open(TOKEN_MINTS_FILE, "r") as f:
            mints = [line.strip() for line in f if line.strip()]
        return mints
    else:
        print(f"{TOKEN_MINTS_FILE} not found. Running {FETCH_TOP_TOKENS_SCRIPT.name} to generate it...", file=sys.stderr)
        result = subprocess.run([sys.executable, str(FETCH_TOP_TOKENS_SCRIPT)], capture_output=True, text=True, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"Failed to generate {TOKEN_MINTS_FILE}. Error:\n{result.stderr}", file=sys.stderr)
            exit(1)
        print(f"Created {TOKEN_MINTS_FILE} automatically.", file=sys.stderr)
        with open(TOKEN_MINTS_FILE, "r") as f:
            mints = [line.strip() for line in f if line.strip()]
        return mints

def has_viable_pool(token_mint: str, debug_this_token_markets=False):
    url = f"{BASE_URL}/tokens/{token_mint}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch pools for {token_mint}: {e}", file=sys.stderr)
        return None, None
    pools = data.get("pools", [])
    print(f"Token {token_mint}: Found {len(pools)} pools.", file=sys.stderr)

    if debug_this_token_markets and pools:
        print(f"DEBUG: Market field for all pools of {token_mint} (liquidity > 0):", file=sys.stderr)
        for p_idx, p_item in enumerate(pools):
            liquidity_usd = p_item.get("liquidity", {}).get("usd", 0)
            try:
                if float(liquidity_usd) > 0:
                    print(f"  Pool {p_idx+1}: ID={p_item.get('poolId')}, Market='{p_item.get('market')}', LiquidityUSD={liquidity_usd}", file=sys.stderr)
            except ValueError:
                pass # ignore if liquidity is not a float

    best_pool = None
    best_liquidity = -1
    for pool in pools:
        liquidity = pool.get("liquidity", {}).get("usd")
        try:
            liquidity = float(liquidity)
        except (TypeError, ValueError):
            print(f"  Skipping pool {pool.get('poolId')} (invalid liquidity: {liquidity})", file=sys.stderr)
            continue
        if liquidity < MIN_LIQUIDITY_USD:
            print(f"  Skipping pool {pool.get('poolId')} (liquidity {liquidity} < {MIN_LIQUIDITY_USD})", file=sys.stderr)
            continue
        print(f"  Pool {pool.get('poolId')} is a candidate (liquidity: {liquidity})", file=sys.stderr)
        if liquidity > best_liquidity:
            best_liquidity = liquidity
            best_pool = pool
    if best_pool:
        print(f"  Selected pool {best_pool.get('poolId')} with liquidity {best_liquidity}", file=sys.stderr)
        return best_pool, data.get("token")
    print(f"  No viable pool found for {token_mint}", file=sys.stderr)
    return None, None

def main():
    output_json = "--json" in sys.argv
    token_mints = load_token_mints()
    if not output_json:
        print(f"Checking {len(token_mints)} tokens for viable pools on Solana Tracker...", file=sys.stderr)
    viable_mints = []
    pool_map = {}

    for i, mint in enumerate(token_mints):
        if not output_json:
            print(f"[{i+1}/{len(token_mints)}] Checking {mint}...", file=sys.stderr)
        
        # Debug market names for the first 5 tokens
        should_debug_this_token = (i < 5) 

        pool_info, token_info = has_viable_pool(mint, debug_this_token_markets=should_debug_this_token)
        if pool_info and token_info:
            base_symbol = token_info.get("symbol", "UNKNOWN")
            base_mint = token_info.get("mint")
            quote_mint = pool_info.get("quoteToken")
            
            if not all([base_symbol, base_mint, quote_mint]):
                print(f"  -> Skipping {mint} due to missing token info.", file=sys.stderr)
                continue

            quote_symbol = "USDC" if quote_mint == USDC_MINT else "SOL"
            market_name = f"{base_symbol}-{quote_symbol}"
            
            # --- Categorize Token ---
            category = "general_token"
            token_name_lower = token_info.get("name", "").lower()
            token_symbol_lower = token_info.get("symbol", "").lower()
            if "pump" in token_name_lower or "pump" in token_symbol_lower:
                category = "pump_graduate"
            
            print(f"  -> Viable pool found for {market_name} (Category: {category})", file=sys.stderr)

            pool_map[market_name] = {
                "poolId": pool_info.get("poolId"),
                "base": {"address": base_mint, "symbol": base_symbol},
                "quote": {"address": quote_mint, "symbol": quote_symbol},
                "category": category
            }
            viable_mints.append(mint)
        else:
            if not output_json:
                print(f"  -> No viable pool.", file=sys.stderr)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Always write the pool_map to pool_addresses.json
    with open(POOL_ADDRESSES_FILE, "w") as f:
        json.dump(pool_map, f, indent=2)
    print(f"Wrote pool address map to {POOL_ADDRESSES_FILE}", file=sys.stderr)

    if output_json:
        print(json.dumps(pool_map, indent=2))
    else:
        with open(TOKEN_MINTS_FILE, "w") as f:
            for mint in viable_mints:
                f.write(mint + "\n")
        print(f"\nFiltered to {len(viable_mints)} tokens with at least one pool and >$10k liquidity. Overwrote {TOKEN_MINTS_FILE}.", file=sys.stderr)

if __name__ == "__main__":
    main() 