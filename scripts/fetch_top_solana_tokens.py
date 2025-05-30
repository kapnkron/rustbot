import requests
import time

SOLANA_TRACKER_API_KEY = "566ba4b7-40b0-4a56-990a-f673d23d4b22"
SOLANA_TRACKER_TOP_TOKENS_URL = "https://api.solanatracker.io/tokens/volume/24h"
TOKEN_MINTS_FILE = "token_mints.txt"
TOP_N = 20  # Number of top tokens to fetch
HEADERS = {"x-api-key": SOLANA_TRACKER_API_KEY}
SLEEP_BETWEEN_REQUESTS = 0.34  # 3 requests per second

def fetch_top_solana_tokens(top_n=TOP_N):
    print(f"Using Solana Tracker API key: {SOLANA_TRACKER_API_KEY}")
    url = SOLANA_TRACKER_TOP_TOKENS_URL
    params = {"limit": top_n}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Error fetching from Solana Tracker: {e}")
        return []
    tokens = data.get("tokens", []) if isinstance(data, dict) else data
    mints = []
    print("\n--- Top Solana Tokens by 24h Volume ---")
    for i, t in enumerate(tokens[:top_n]):
        mint = t.get("mint") or t.get("address")
        symbol = t.get("symbol")
        name = t.get("name")
        if mint:
            print(f"{symbol} ({name}): {mint}")
            mints.append(mint)
        if i < top_n - 1:
            time.sleep(SLEEP_BETWEEN_REQUESTS)
    with open(TOKEN_MINTS_FILE, "w") as f:
        for mint in mints:
            f.write(mint + "\n")
    print(f"\nWrote {len(mints)} token mints to {TOKEN_MINTS_FILE}")
    return mints

if __name__ == "__main__":
    fetch_top_solana_tokens() 