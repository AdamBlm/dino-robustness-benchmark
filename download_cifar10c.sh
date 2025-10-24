#!/bin/bash
#
# Download and extract CIFAR-10-C into ./data/CIFAR-10-C
#

set -e

# Move to repo root (this script lives in slurm/)
cd "$(dirname "$0")/.."

DATA_DIR="./data"
TARGET_DIR="$DATA_DIR/CIFAR-10-C"
TAR_PATH="$DATA_DIR/CIFAR-10-C.tar"

mkdir -p "$DATA_DIR"

if [ -d "$TARGET_DIR" ]; then
    echo "Found $TARGET_DIR; nothing to do."
    exit 0
fi

echo "Downloading CIFAR-10-C to $TAR_PATH ..."

URLS=(
    "https://zenodo.org/record/3555552/files/CIFAR-10-C.tar?download=1"
    "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
)

downloaded=false
for url in "${URLS[@]}"; do
    echo "Attempting: $url"
    if command -v wget >/dev/null 2>&1; then
        if wget -O "$TAR_PATH" "$url"; then
            downloaded=true; break
        fi
    elif command -v curl >/dev/null 2>&1; then
        if curl -L "$url" -o "$TAR_PATH"; then
            downloaded=true; break
        fi
    else
        echo "Error: neither wget nor curl is available. Please install one." >&2
        exit 1
    fi
done

if [ "$downloaded" != true ]; then
    echo "Failed to download CIFAR-10-C from known URLs. Please download manually from https://github.com/hendrycks/robustness and place it under $TARGET_DIR" >&2
    exit 1
fi

echo "Extracting $TAR_PATH ..."
tar -xf "$TAR_PATH" -C "$DATA_DIR"

if [ -d "$TARGET_DIR" ]; then
    echo "CIFAR-10-C is ready at $TARGET_DIR"
else
    echo "Extraction did not produce $TARGET_DIR. Please check the archive contents." >&2
    exit 1
fi

echo "Done."


