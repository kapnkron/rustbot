#!/bin/bash

# Exit on error
set -e

# Configuration
APP_NAME="trading-bot"
APP_DIR="/var/lib/$APP_NAME"
BIN_DIR="/usr/local/bin"
CONFIG_DIR="$APP_DIR/config"
DATA_DIR="$APP_DIR/data"
LOG_DIR="$APP_DIR/logs"
BACKUP_DIR="$APP_DIR/backups"

# Create directories if they don't exist
sudo mkdir -p $CONFIG_DIR $DATA_DIR $LOG_DIR $BACKUP_DIR

# Backup current version
if [ -f "$BIN_DIR/$APP_NAME" ]; then
    echo "Backing up current version..."
    sudo cp "$BIN_DIR/$APP_NAME" "$BACKUP_DIR/$APP_NAME.$(date +%Y%m%d%H%M%S)"
fi

# Backup configuration
if [ -f "$CONFIG_DIR/config.toml" ]; then
    echo "Backing up configuration..."
    sudo cp "$CONFIG_DIR/config.toml" "$BACKUP_DIR/config.$(date +%Y%m%d%H%M%S).toml"
fi

# Backup database
if [ -f "$DATA_DIR/trading.db" ]; then
    echo "Backing up database..."
    sudo sqlite3 "$DATA_DIR/trading.db" ".backup '$BACKUP_DIR/trading.$(date +%Y%m%d%H%M%S).db'"
fi

# Stop service
echo "Stopping service..."
sudo systemctl stop $APP_NAME

# Deploy new version
echo "Deploying new version..."
sudo cp target/release/$APP_NAME "$BIN_DIR/"
sudo chmod +x "$BIN_DIR/$APP_NAME"

# Update configuration if provided
if [ -f "config.toml" ]; then
    echo "Updating configuration..."
    sudo cp config.toml "$CONFIG_DIR/"
fi

# Set permissions
echo "Setting permissions..."
sudo chown -R $APP_NAME:$APP_NAME $APP_DIR
sudo chmod -R 750 $APP_DIR

# Start service
echo "Starting service..."
sudo systemctl start $APP_NAME

# Verify deployment
echo "Verifying deployment..."
sleep 5
if systemctl is-active --quiet $APP_NAME; then
    echo "Deployment successful!"
else
    echo "Deployment failed! Service is not running."
    exit 1
fi 