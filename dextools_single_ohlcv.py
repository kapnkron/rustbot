import requests
import pandas as pd

API_KEY = "L8tBdVzmLNaVJfrmhlGr87efKnpkj1Ez4lO58nBi"
headers = {"X-API-KEY": API_KEY}

# SOL/USDC on Solana
base_token = "So11111111111111111111111111111111111111112"  # SOL
quote_token = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
interval = "1d"  # daily candles

url = f"https://public-api.dextools.io/tradingview/v1/solana/ohlcv?baseToken={base_token}&quoteToken={quote_token}&interval={interval}"

response = requests.get(url, headers=headers)
print("Status code:", response.status_code)
print("Response:", response.text[:500])  # Print first 500 chars for brevity

if response.status_code == 200:
    data = response.json()
    if "data" in data:
        df = pd.DataFrame(data["data"])
        df.to_csv(f"SOL_USDC_ohlcv.csv", index=False)
        print("Saved OHLCV data to SOL_USDC_ohlcv.csv")
    else:
        print("No data found in response.")
else:
    print("Failed to fetch data:", response.text) 