import requests
import pandas as pd
import time

API_KEY = "L8tBdVzmLNaVJfrmhlGr87efKnpkj1Ez4lO58nBi"  # Your DEXTools API key
headers = {"X-API-KEY": API_KEY}

endpoints = {
    "gainers": "https://public-api.dextools.io/trial/v2/ranking/solana/gainers",
    "losers": "https://public-api.dextools.io/trial/v2/ranking/solana/losers",
    "hotpools": "https://public-api.dextools.io/trial/v2/ranking/solana/hotpools"
}

for name, url in endpoints.items():
    print(f"Fetching {name}...")
    for attempt in range(3):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            top_100 = df.head(100)
            out_csv = f"top_100_solana_{name}.csv"
            top_100.to_csv(out_csv, index=False)
            print(f"Saved {out_csv} ({len(top_100)} rows)")
            # Print preview of main columns if present
            preview_cols = [col for col in ['mainToken', 'sideToken', 'price', 'variation24h'] if col in top_100.columns]
            print(top_100[preview_cols].head(3) if preview_cols else top_100.head(3))
            break
        elif response.status_code == 429:
            print(f"Rate limited on {name}, attempt {attempt+1}/3. Waiting 10 seconds...")
            time.sleep(10)
        else:
            print(f"Failed to fetch {name}: {response.status_code} {response.text}")
            break
    time.sleep(1.1)  # Wait between requests 