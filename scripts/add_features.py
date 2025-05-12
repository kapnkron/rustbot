import os
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

raw_dir = 'data/raw'
features_dir = 'data/features'
os.makedirs(features_dir, exist_ok=True)

# List all OHLCV CSVs
for fname in os.listdir(raw_dir):
    if not fname.endswith('_usd_ohlcv.csv'):
        continue
    asset = fname.split('_')[0].upper()
    print(f'Processing {asset}...')
    df = pd.read_csv(os.path.join(raw_dir, fname), parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Price returns
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Moving averages
    df['sma_10'] = SMAIndicator(df['close'], window=10).sma_indicator()
    df['ema_10'] = EMAIndicator(df['close'], window=10).ema_indicator()
    df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()

    # RSI
    df['rsi_14'] = RSIIndicator(df['close'], window=14).rsi()

    # MACD
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_high'] - df['bb_low']

    # Volatility (rolling std of returns)
    df['volatility_10'] = df['return'].rolling(window=10).std()
    df['volatility_50'] = df['return'].rolling(window=50).std()

    # Lagged close prices
    for lag in [1, 2, 3]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)

    # Drop rows with NaNs from feature creation
    df = df.dropna().reset_index(drop=True)

    out_path = os.path.join(features_dir, fname.replace('_ohlcv', '_features'))
    df.to_csv(out_path, index=False)
    print(f'Saved features to {out_path}') 