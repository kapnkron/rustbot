import os
import pandas as pd
from pycoingecko import CoinGeckoAPI
from datetime import datetime
import time

# Asset list: CoinGecko IDs
assets = {
    'solana': 'SOL',
    'jupiter-exchange': 'JUP',
    'chainlink': 'LINK',
    'dogwifcoin': 'WIF',
    'bonk': 'BONK',
    'tether': 'USDT',
    'ethereum': 'ETH',
    'cardano': 'ADA',
}

vs_currency = 'usd'
output_dir = 'data/raw'
os.makedirs(output_dir, exist_ok=True)

cg = CoinGeckoAPI()

def fetch_and_save_ohlcv(asset_id, symbol):
    print(f"Fetching {symbol} ({asset_id}) for last 365 days...")
    try:
        ohlc = cg.get_coin_ohlc_by_id(asset_id, vs_currency, days=365)
        if not ohlc:
            print(f"No OHLCV data for {symbol} (skipping)")
            return
        df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['date', 'open', 'high', 'low', 'close']]
        out_path = os.path.join(output_dir, f"{symbol.lower()}_usd_ohlcv.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {symbol} data to {out_path}")
    except Exception as e:
        print(f"Skipping {symbol}: {e}")

for asset_id, symbol in assets.items():
    fetch_and_save_ohlcv(asset_id, symbol)
    time.sleep(1)  # Be nice to the API 