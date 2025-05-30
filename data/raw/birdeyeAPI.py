import requests
import pandas as pd

API_KEY = "70aa31da27d64f3189aa671a546f3dc3"
url = "https://public-api.birdeye.so/defi/tokenlist"
headers = {"X-API-KEY": API_KEY}
response = requests.get(url, headers=headers)

print(response.status_code)
print(response.text[:500])  # Print only the first 500 chars for brevity

data = response.json()
if "data" in data and "tokens" in data["data"]:
    tokens = data["data"]["tokens"]
    df = pd.DataFrame(tokens)
    # Use the correct column names
    min_volume = 100_000      # $100k 24h volume
    min_liquidity = 500_000   # $500k liquidity
    filtered = df[(df['v24hUSD'] > min_volume) & (df['liquidity'] > min_liquidity)]
    filtered = filtered.sort_values('v24hUSD', ascending=False)
    filtered[['symbol', 'address', 'v24hUSD', 'liquidity']].to_csv("top_solana_tokens.csv", index=False)
    print("Saved top_solana_tokens.csv")
    print(filtered[['symbol', 'address', 'v24hUSD', 'liquidity']].head(20))
else:
    print("API response did not contain expected data structure.")
