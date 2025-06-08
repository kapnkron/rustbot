#!/bin/bash

# Script to create a Python 3.11 virtual environment for Beatrice
# This fixes the snscrape compatibility issue with Python 3.12

echo "Setting up Python 3.11 virtual environment for Beatrice..."

# Check if Python 3.11 is installed
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 is not installed. Installing..."
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
    
    # Check if installation was successful
    if ! command -v python3.11 &> /dev/null; then
        echo "Failed to install Python 3.11. Please install it manually."
        exit 1
    fi
fi

echo "Python 3.11 found. Creating virtual environment..."

# Create the virtual environment
python3.11 -m venv ~/.py311_venv

# Activate the virtual environment
source ~/.py311_venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install pandas numpy matplotlib textblob bs4 requests python-dotenv snscrape solana-py solders solana tqdm torch

# Show installed packages
echo "Installed packages:"
pip list

echo "Python 3.11 virtual environment created at ~/.py311_venv"
echo ""
echo "To activate the environment:"
echo "  source ~/.py311_venv/bin/activate"
echo ""
echo "To run the pipeline with this environment:"
echo "  ./run_py311_pipeline.sh" 