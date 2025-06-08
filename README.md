# Beatrice Trading Bot

This is an AI-assisted trading bot project.

## Overview

(Details to be filled in later)

## Project Structure

(Details to be filled in later)

## Setup

(Details to be filled in later)

## Usage

(Details to be filled in later)

# Beatrice

Beatrice is a cryptocurrency trading bot designed for the Solana blockchain.

## OHLCV Data Collection

To collect historical OHLCV (Open, High, Low, Close, Volume) data for trading pools:

```bash
# Activate the virtual environment
source ~/.pytorch_venv/bin/activate

# Fetch OHLCV data for all pools in pool_addresses.json
python scripts/fetch_historical_ohlcv.py

# Fetch data for a specific pool ID
python scripts/fetch_historical_ohlcv.py --pool <POOL_ID>

# Configure number of days to fetch (default is 7)
python scripts/fetch_historical_ohlcv.py --days 30

# Configure number of threads (default is 5)
python scripts/fetch_historical_ohlcv.py --threads 10

# Reset checkpoint and start fresh
python scripts/fetch_historical_ohlcv.py --reset
```

The script has the following features:
- Automatic checkpointing for resuming interrupted runs
- Thread-safe data collection with configurable parallelism
- Error handling with automatic retries
- Progress tracking and detailed logging
- Graceful handling of interruptions (Ctrl+C)

Data will be saved to:
- Raw trades: `data/raw/<MARKET_NAME>_trades.csv`
- OHLCV data: `data/processed/DATA_<BASE_MINT>_<QUOTE_MINT>_OHLCV_5min.csv`
- Checkpoint: `data/checkpoint_ohlcv.pkl`
- Human-readable progress: `data/processed_markets.json`
- Error logs: `data/ohlcv_errors.log`

You can leave this script running unattended for days, and it will automatically resume from where it left off if interrupted. 

## ML Pipeline

To run the complete ML pipeline, which includes OHLCV data collection, feature engineering, and model training:

```bash
# Activate the virtual environment
source ~/.pytorch_venv/bin/activate

# Run the fixed pipeline for Python 3.12
./run_fixed_pipeline_py312.sh --days 3
```

### Sentiment Data

Sentiment analysis has been temporarily removed from the pipeline. The model will use neutral sentiment values (0) for all tokens.

### Command-line Options

The pipeline script accepts the following options:

- `--initial-run`: Perform an initial run with 30 days of data
- `--days N`: Override to fetch N days of data 
- `--token SYMBOL`: Process only a specific token
- `--min-data N`: Set minimum data points needed before first training (default: 10000)
- `--retrain-hours N`: Set hours between retraining runs (default: 6)

### Troubleshooting

If you encounter issues with the signature conversion in the OHLCV fetcher, you can run the fix separately:

```bash
python fix_signature_issue.py
```

This fixes the issue where RpcSignaturesForAddressConfig objects cannot be converted to Signature objects. 