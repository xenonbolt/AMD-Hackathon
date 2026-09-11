#!/bin/bash
# setup_swap.sh
# Dynamically creates a 16GB swap file if one doesn't exist to prevent PyTorch OOMs during initial load.
# Requires root privileges (run via sudo).

set -e

SWAPFILE="/swapfile"
SWAP_SIZE="16G"

# Check if swap is already active
if swapon --show | grep -q "^$SWAPFILE\b"; then
    echo "Swap file $SWAPFILE is already active."
    exit 0
fi

# Check if the file exists but isn't active
if [ -f "$SWAPFILE" ]; then
    echo "Swap file $SWAPFILE exists but is not active. Activating now..."
    sudo swapon $SWAPFILE
    exit 0
fi

echo "Creating a $SWAP_SIZE swap file at $SWAPFILE..."
# fallocate is faster than dd, but if it fails on some filesystems, fallback to dd
sudo fallocate -l $SWAP_SIZE $SWAPFILE || sudo dd if=/dev/zero of=$SWAPFILE bs=1M count=16384 status=progress

# Secure the swap file
sudo chmod 600 $SWAPFILE

# Format it as swap
sudo mkswap $SWAPFILE

# Enable the swap
sudo swapon $SWAPFILE

echo "Swap file created and activated successfully."
free -h
