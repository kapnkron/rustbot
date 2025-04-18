#!/bin/bash

# Exit on error
set -e

# Configuration
APP_NAME="trading-bot"
APP_DIR="/var/lib/$APP_NAME"
BIN_DIR="/usr/local/bin"
BACKUP_DIR="$APP_DIR/backups"

# Check if backup exists
if [ ! -d "$BACKUP_DIR" ]; then
    echo "No backup directory found!"
    exit 1
fi

# Get latest backup
LATEST_BACKUP=$(ls -t "$BACKUP_DIR/$APP_NAME."* | head -n 1)
if [ -z "$LATEST_BACKUP" ]; then
    echo "No backup found!"
    exit 1
fi

# Stop service
echo "Stopping service..."
sudo systemctl stop $APP_NAME

# Restore from backup
echo "Restoring from backup: $LATEST_BACKUP"
sudo cp "$LATEST_BACKUP" "$BIN_DIR/$APP_NAME"
sudo chmod +x "$BIN_DIR/$APP_NAME"

# Restore configuration if backup exists
LATEST_CONFIG=$(ls -t "$BACKUP_DIR/config."*.toml | head -n 1)
if [ -n "$LATEST_CONFIG" ]; then
    echo "Restoring configuration..."
    sudo cp "$LATEST_CONFIG" "$APP_DIR/config/config.toml"
fi

# Restore database if backup exists
LATEST_DB=$(ls -t "$BACKUP_DIR/trading."*.db | head -n 1)
if [ -n "$LATEST_DB" ]; then
    echo "Restoring database..."
    sudo sqlite3 "$LATEST_DB" ".backup '$APP_DIR/data/trading.db'"
fi

# Start service
echo "Starting service..."
sudo systemctl start $APP_NAME

# Verify rollback
echo "Verifying rollback..."
sleep 5
if systemctl is-active --quiet $APP_NAME; then
    echo "Rollback successful!"
else
    echo "Rollback failed! Service is not running."
    exit 1
fi 