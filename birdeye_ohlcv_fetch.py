import requests
import pandas as pd
import os
import time

API_KEY = "70aa31da27d64f3189aa671a546f3dc3"
input_csv = "top_solana_tokens.csv"
output_dir = "ohlcv"
interval = "1d"  # Use "1d" for daily, "1h" for hourly, etc.

os.makedirs(output_dir, exist_ok=True)

tokens = pd.read_csv(input_csv)

for idx, row in tokens.iterrows():
    symbol = row['symbol']
    address = row['address']
    print(f"Fetching OHLCV for {symbol} ({address})...")
    url = f"https://public-api.birdeye.so/defi/ohlcv?address={address}&interval={interval}"
    headers = {"X-API-KEY": API_KEY}
    for attempt in range(3):  # Try up to 3 times
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            break
        elif response.status_code == 429:
            print(f"Rate limited for {symbol}, waiting 10 seconds...")
            time.sleep(10)
        elif response.status_code == 401:
            print(f"Unauthorized for {symbol}: {response.text}")
            break
        else:
            print(f"Failed to fetch {symbol}: {response.status_code} {response.text}")
            break
    if response.status_code != 200:
        continue
    data = response.json()
    if "data" in data and "items" in data["data"]:
        ohlcv = data["data"]["items"]
        if not ohlcv:
            print(f"No OHLCV data for {symbol}")
            continue
        df = pd.DataFrame(ohlcv)
        df['symbol'] = symbol
        df['address'] = address
        out_path = os.path.join(output_dir, f"{symbol}_{address}_ohlcv.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {symbol} OHLCV to {out_path}")
    else:
        print(f"Unexpected response for {symbol}: {data}")
    time.sleep(1.1)  # Stay safely under 60 RPM

print("Done fetching OHLCV for all tokens.") 