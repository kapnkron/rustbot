#!/bin/bash

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create a temporary file for the cron job
CRON_TEMP=$(mktemp)

# Add the cron job to run at 00:00 UTC every day
echo "0 0 * * * cd $PROJECT_DIR && python3 fetch_market_data.py >> market_data_fetch.log 2>&1" > "$CRON_TEMP"

# Install the cron job
crontab "$CRON_TEMP"

# Remove the temporary file
rm "$CRON_TEMP"

echo "Daily market data fetch has been scheduled to run at 00:00 UTC every day"
echo "You can check the logs in market_data_fetch.log" 