import requests
import time

SOLANA_TRACKER_API_KEY = "ef4f484a-538c-42ac-92c9-3158d267f7e6"
BASE_URL = "https://data.solanatracker.io"
TOKEN_MINTS_FILE = "token_mints.txt"
TOP_N = 100  # Number of tokens to fetch from each endpoint (max 100)
HEADERS = {"x-api-key": SOLANA_TRACKER_API_KEY}
SLEEP_BETWEEN_REQUESTS = 1.2  # 1 request every 1.2 seconds for extra safety


def fetch_tokens_by_volume(top_n=TOP_N):
    url = f"{BASE_URL}/tokens/volume/24h"
    params = {"limit": top_n}
    print(f"\n[DEBUG] Requesting: {url}")
    print(f"[DEBUG] Params: {params}")
    print(f"[DEBUG] Headers: {HEADERS}")
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        print("[DEBUG] Raw response:", data)
    except Exception as e:
        print(f"Error fetching top tokens by volume: {e}")
        return []
    tokens = data.get("tokens", []) if isinstance(data, dict) else data
    mints = []
    print("\n--- Top Solana Tokens by 24h Volume ---")
    for t in tokens[:top_n]:
        mint = t["token"]["mint"] if isinstance(t, dict) and "token" in t and "mint" in t["token"] else None
        symbol = t["token"].get("symbol") if isinstance(t, dict) and "token" in t else None
        name = t["token"].get("name") if isinstance(t, dict) and "token" in t else None
        if mint:
            print(f"{symbol} ({name}): {mint}")
            mints.append(mint)
    return mints

def fetch_trending_tokens(top_n=TOP_N, timeframe="1h"):
    url = f"{BASE_URL}/tokens/trending"
    params = {"limit": top_n, "timeframe": timeframe}
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    print(f"\n[DEBUG] Requesting: {url}")
    print(f"[DEBUG] Params: {params}")
    print(f"[DEBUG] Headers: {HEADERS}")
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        print("[DEBUG] Raw response:", data)
    except Exception as e:
        print(f"Error fetching trending tokens: {e}")
        return []
    tokens = data.get("tokens", []) if isinstance(data, dict) else data
    mints = []
    print(f"\n--- Trending Solana Tokens ({timeframe}) ---")
    for t in tokens[:top_n]:
        mint = t["token"]["mint"] if isinstance(t, dict) and "token" in t and "mint" in t["token"] else None
        symbol = t["token"].get("symbol") if isinstance(t, dict) and "token" in t else None
        name = t["token"].get("name") if isinstance(t, dict) and "token" in t else None
        if mint:
            print(f"{symbol} ({name}): {mint}")
            mints.append(mint)
    return mints

def main():
    print(f"Using Solana Tracker API key: {SOLANA_TRACKER_API_KEY}")
    volume_mints = fetch_tokens_by_volume(TOP_N)
    trending_mints = fetch_trending_tokens(TOP_N, timeframe="1h")
    # Merge and deduplicate
    merged_mints = list(dict.fromkeys(volume_mints + trending_mints))
    # Keep only the top 25 tokens
    filtered_mints = merged_mints[:25]
    print(f"\n--- Final Filtered Token Mint List ({len(filtered_mints)} tokens) ---")
    for mint in filtered_mints:
        print(mint)
    with open(TOKEN_MINTS_FILE, "w") as f:
        for mint in filtered_mints:
            f.write(mint + "\n")
    print(f"\nWrote {len(filtered_mints)} token mints to {TOKEN_MINTS_FILE}")

if __name__ == "__main__":
    main() 