#!/bin/bash

# Exit on error
set -e

# Configuration
APP_NAME="trading-bot"
APP_DIR="/var/lib/$APP_NAME"
BIN_DIR="/usr/local/bin"
CONFIG_DIR="$APP_DIR/config"
DATA_DIR="$APP_DIR/data"
BACKUP_DIR="$APP_DIR/backups"
SERVICE_FILE="/etc/systemd/system/$APP_NAME.service"

# Create user and group
if ! id "$APP_NAME" &>/dev/null; then
    echo "Creating user and group..."
    sudo useradd -r -s /bin/false "$APP_NAME"
fi

# Create directories
echo "Creating directories..."
sudo mkdir -p "$APP_DIR" "$CONFIG_DIR" "$DATA_DIR" "$BACKUP_DIR"
sudo chown -R "$APP_NAME:$APP_NAME" "$APP_DIR"
sudo chmod 750 "$APP_DIR"

# Install binary
echo "Installing binary..."
sudo cp "target/release/$APP_NAME" "$BIN_DIR/"
sudo chown root:root "$BIN_DIR/$APP_NAME"
sudo chmod 755 "$BIN_DIR/$APP_NAME"

# Install configuration
echo "Installing configuration..."
if [ -f "config/config.toml" ]; then
    sudo cp "config/config.toml" "$CONFIG_DIR/"
    sudo chown "$APP_NAME:$APP_NAME" "$CONFIG_DIR/config.toml"
    sudo chmod 640 "$CONFIG_DIR/config.toml"
fi

# Install service file
echo "Installing service file..."
sudo cp "scripts/$APP_NAME.service" "$SERVICE_FILE"
sudo chown root:root "$SERVICE_FILE"
sudo chmod 644 "$SERVICE_FILE"

# Reload systemd
echo "Reloading systemd..."
sudo systemctl daemon-reload

# Enable and start service
echo "Enabling and starting service..."
sudo systemctl enable "$APP_NAME"
sudo systemctl start "$APP_NAME"

# Verify installation
echo "Verifying installation..."
sleep 5
if systemctl is-active --quiet "$APP_NAME"; then
    echo "Installation successful!"
else
    echo "Installation failed! Service is not running."
    exit 1
fi 