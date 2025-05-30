import requests
import json
import os
import subprocess
from typing import List, Dict

RAYDIUM_POOLS_URL = "https://api.raydium.io/v2/sdk/liquidity/mainnet.json"
ORCA_POOLS_URL = "https://api.orca.so/v1/pools"
TOKEN_MINTS_FILE = "token_mints.txt"
FETCH_TOP_TOKENS_SCRIPT = "fetch_top_solana_tokens.py"


def fetch_raydium_pools() -> List[Dict]:
    resp = requests.get(RAYDIUM_POOLS_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Raydium pools are in a dict keyed by pool address
    return list(data.values())


def fetch_orca_pools() -> List[Dict]:
    resp = requests.get(ORCA_POOLS_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Orca pools are in a dict under 'pools'
    return list(data["pools"].values())


def find_pools_for_token(token_mint: str, raydium_pools: List[Dict], orca_pools: List[Dict]) -> Dict[str, List[Dict]]:
    result = {"raydium": [], "orca": []}
    # Raydium: check baseMint and quoteMint
    for pool in raydium_pools:
        if pool.get("baseMint") == token_mint or pool.get("quoteMint") == token_mint:
            result["raydium"].append(pool)
    # Orca: check tokenA and tokenB mints
    for pool in orca_pools:
        if pool.get("tokenA", {}).get("mint") == token_mint or pool.get("tokenB", {}).get("mint") == token_mint:
            result["orca"].append(pool)
    return result


def load_token_mints() -> List[str]:
    if os.path.exists(TOKEN_MINTS_FILE):
        print(f"Loading token mints from {TOKEN_MINTS_FILE}...")
        with open(TOKEN_MINTS_FILE, "r") as f:
            mints = [line.strip() for line in f if line.strip()]
        return mints
    else:
        print(f"{TOKEN_MINTS_FILE} not found. Running {FETCH_TOP_TOKENS_SCRIPT} to generate it...")
        result = subprocess.run(["python3", FETCH_TOP_TOKENS_SCRIPT], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to generate {TOKEN_MINTS_FILE}. Error:\n{result.stderr}")
            exit(1)
        print(f"Created {TOKEN_MINTS_FILE} automatically.")
        with open(TOKEN_MINTS_FILE, "r") as f:
            mints = [line.strip() for line in f if line.strip()]
        return mints


def main():
    token_mints = load_token_mints()
    print("Fetching Raydium pools...")
    raydium_pools = fetch_raydium_pools()
    print(f"Found {len(raydium_pools)} Raydium pools.")
    print("Fetching Orca pools...")
    orca_pools = fetch_orca_pools()
    print(f"Found {len(orca_pools)} Orca pools.")

    for mint in token_mints:
        pools = find_pools_for_token(mint, raydium_pools, orca_pools)
        print(f"\nToken mint: {mint}")
        print(f"  Raydium pools: {len(pools['raydium'])}")
        for pool in pools["raydium"]:
            print(f"    - Pool: {pool.get('id')} | base: {pool.get('baseMint')} | quote: {pool.get('quoteMint')}")
        print(f"  Orca pools: {len(pools['orca'])}")
        for pool in pools["orca"]:
            print(f"    - Pool: {pool.get('address')} | tokenA: {pool.get('tokenA', {}).get('mint')} | tokenB: {pool.get('tokenB', {}).get('mint')}")

if __name__ == "__main__":
    main() 