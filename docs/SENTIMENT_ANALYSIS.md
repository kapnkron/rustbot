# Sentiment Analysis for Crypto Trading

This document describes the sentiment analysis component of our crypto trading system, how it works, and how to use it.

## Overview

Our sentiment analysis system collects and processes sentiment data from various sources:

1. **Twitter** - Using snscrape to collect mentions of tokens without requiring an API key
2. **DexScreener** - Web scraping token-specific pages for community comments
3. **Fear & Greed Index** - Market-wide sentiment indicator

The collected data is processed using TextBlob for sentiment analysis, which provides a polarity score between -1 (negative) and 1 (positive).

## Setup

### Prerequisites

- Python 3.8+
- System development packages (see installation script)
- The `.pytorch_venv` virtual environment must exist in your home directory

### Installation

1. Install the dependencies using the provided script:

```bash
cd scripts
./install_sentiment_deps.sh
```

This will:
- Install required system packages
- Use the existing `.pytorch_venv` in your home directory
- Install Python dependencies
- Install snscrape from GitHub (more reliable than PyPI version)

## Usage

### Running the Sentiment Crawler

Ensure the `.pytorch_venv` is activated before running the scripts:

```bash
source ~/.pytorch_venv/bin/activate
```

To collect sentiment data for all tokens in your `pool_addresses.json` file:

```bash
python scripts/sentiment_crawler.py
```

Options:
- `--days NUMBER` - Number of days of historical data to fetch (default: 30)
- `--token SYMBOL` - Process only a specific token symbol (e.g., "SOL")
- `--max-tweets NUMBER` - Maximum number of tweets to fetch per token (default: 500)

### Running the Full Pipeline

The `run_sentiment_and_pipeline.py` script automatically uses the correct Python interpreter from your `.pytorch_venv`:

```bash
python scripts/run_sentiment_and_pipeline.py
```

Options:
- Same options as the sentiment crawler, plus:
- `--skip-sentiment` - Skip running the sentiment crawler
- `--skip-pipeline` - Skip running the ML pipeline

## Data Storage

The sentiment data is stored in the following locations:

- Token-specific sentiment: `data/sentiment/{token_address}.csv`
- Market-wide sentiment: `data/sentiment/market_sentiment.csv`
- Sentiment plots: `data/sentiment/plots/`

## Visualizations

The system automatically generates the following plots:

1. Token-specific Twitter sentiment over time
2. Combined sentiment (from all sources) for each token
3. Market-wide sentiment indicators

These plots are saved in the `data/sentiment/plots/` directory.

## Integration with ML Pipeline

The sentiment data is integrated into the ML pipeline as additional features for prediction:

1. Token-specific sentiment
2. Market-wide sentiment
3. Sentiment change over time

## Troubleshooting

### Twitter Scraping Issues

If you encounter issues with Twitter scraping:

1. Ensure snscrape is installed from GitHub (the PyPI version may be outdated)
2. Check your network connection (some networks may block scraping activities)
3. Try reducing the number of tweets (`--max-tweets` option)

### Web Scraping Failures

If web scraping fails:

1. Check if the website structure has changed (may require updates to the scraping selectors)
2. Ensure you're not being rate-limited (increase sleep time between requests)
3. Try using a different user agent

## Maintenance

The sentiment analysis components may require periodic updates due to changes in:

1. Website structures (affecting web scraping)
2. Twitter's API policies (affecting snscrape)
3. New sentiment sources becoming available

Regular testing and monitoring of the data collection process is recommended. 