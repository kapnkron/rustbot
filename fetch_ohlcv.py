import os
import pandas as pd
from pycoingecko import CoinGeckoAPI
from datetime import datetime
import time

# List of CSVs to read from project root
token_csvs = [
    'top_100_solana_gainers.csv',
    'top_100_solana_losers.csv',
    'top_solana_tokens.csv'
]

vs_currency = 'usd'
output_dir = 'data/raw'
os.makedirs(output_dir, exist_ok=True)

cg = CoinGeckoAPI()

# Get all CoinGecko coins for mapping
print("Fetching CoinGecko coins list...")
all_coins = cg.get_coins_list()
coingecko_symbol_to_id = {c['symbol'].upper(): c['id'] for c in all_coins}
coingecko_name_to_id = {c['name'].upper(): c['id'] for c in all_coins}

# Extract unique symbols from all CSVs
symbols = set()
for fname in token_csvs:
    if not os.path.exists(fname):
        print(f"CSV not found: {fname}")
        continue
    df = pd.read_csv(fname)
    # Try to extract from 'symbol' column, fallback to 'data' column if present
    if 'symbol' in df.columns:
        symbols.update(df['symbol'].dropna().str.upper())
    elif 'data' in df.columns:
        import ast
        for val in df['data'].dropna():
            try:
                d = ast.literal_eval(val)
                for key in ['mainToken', 'sideToken']:
                    tok = d.get(key)
                    if tok and 'symbol' in tok:
                        symbols.add(tok['symbol'].strip().upper())
            except Exception:
                continue

print(f"Found {len(symbols)} unique symbols from CSVs.")

# Map symbols to CoinGecko IDs
symbol_to_id = {}
unmapped_symbols = []
for symbol in symbols:
    if symbol in coingecko_symbol_to_id:
        symbol_to_id[symbol] = coingecko_symbol_to_id[symbol]
    elif symbol in coingecko_name_to_id:
        symbol_to_id[symbol] = coingecko_name_to_id[symbol]
    else:
        unmapped_symbols.append(symbol)

print(f"Mapped {len(symbol_to_id)} symbols to CoinGecko IDs. {len(unmapped_symbols)} could not be mapped.")

# Log unmapped symbols
with open('fetch_ohlcv_unmapped.log', 'w') as f:
    for s in unmapped_symbols:
        f.write(f"{s}\n")

# Fetch and save OHLCV data for each mapped token
def fetch_and_save_ohlcv(asset_id, symbol):
    print(f"Fetching {symbol} ({asset_id}) for last 365 days...")
    try:
        ohlc = cg.get_coin_ohlc_by_id(asset_id, vs_currency, days=365)
        if not ohlc:
            print(f"No OHLCV data for {symbol} (skipping)")
            return False
        df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['date', 'open', 'high', 'low', 'close']]
        out_path = os.path.join(output_dir, f"{symbol.lower()}_usd_ohlcv.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {symbol} data to {out_path}")
        return True
    except Exception as e:
        print(f"Skipping {symbol}: {e}")
        return False

failed_fetch = []
for symbol, asset_id in symbol_to_id.items():
    success = fetch_and_save_ohlcv(asset_id, symbol)
    if not success:
        failed_fetch.append(symbol)
    time.sleep(1.2)  # Be nice to the API, increased to 1.2s

# Log failed fetches
with open('fetch_ohlcv_failed.log', 'w') as f:
    for s in failed_fetch:
        f.write(f"{s}\n")

print(f"Done. {len(failed_fetch)} tokens failed to fetch OHLCV data. See fetch_ohlcv_failed.log for details.") 