#!/bin/bash

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt

echo "✅ Environment set up. To activate later: source .venv/bin/activate"
