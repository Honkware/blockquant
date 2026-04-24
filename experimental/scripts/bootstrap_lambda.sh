#!/bin/bash
# BlockQuant Lambda Bootstrap
# Runs once on a fresh Lambda instance to install dependencies.

set -e

BQ_DIR="/opt/blockquant"
EXL3_DIR="/opt/exllamav3"

echo "=== BlockQuant Lambda Bootstrap ==="

# System deps
echo "Updating packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-pip python3-venv

# Install exllamav3
echo "Installing ExLlamaV3..."
if [ ! -d "$EXL3_DIR" ]; then
    git clone --depth 1 https://github.com/turboderp-org/exllamav3 "$EXL3_DIR"
fi
cd "$EXL3_DIR"
pip3 install -q -e .

# Install BlockQuant backend
echo "Installing BlockQuant..."
if [ ! -d "$BQ_DIR" ]; then
    git clone --depth 1 https://github.com/Honkware/blockquant "$BQ_DIR" 2>/dev/null || true
fi
cd "$BQ_DIR/backend"
pip3 install -q -r requirements.txt

echo "=== Bootstrap complete ==="
echo "ExLlamaV3: $EXL3_DIR"
echo "BlockQuant: $BQ_DIR"
