#!/bin/bash
# Install sentiment analysis dependencies

set -e  # Exit on error

echo "Installing sentiment analysis dependencies..."

# Define the virtual environment path
VENV_PATH="$HOME/.pytorch_venv"

# Install system dependencies
if [ -f "/etc/debian_version" ]; then
    echo "Detected Debian/Ubuntu system"
    sudo apt-get update
    sudo apt-get install -y python3-dev build-essential
elif [ -f "/etc/redhat-release" ]; then
    echo "Detected RHEL/CentOS/Fedora system"
    sudo yum install -y python3-devel gcc
elif [ -f "/etc/arch-release" ]; then
    echo "Detected Arch Linux system"
    sudo pacman -S --noconfirm python-pip base-devel
else
    echo "Unknown Linux distribution. Please install Python development packages manually."
fi

# Check if the virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH"
    echo "Please create the .pytorch_venv first"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r ../sentiment_dependencies.txt

# Install snscrape from git (more reliable than PyPI version)
echo "Installing snscrape from GitHub..."
pip install git+https://github.com/JustAnotherArchivist/snscrape.git

echo "Installation complete!"
echo "To activate the virtual environment, run: source $VENV_PATH/bin/activate" 