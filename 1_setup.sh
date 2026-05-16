#!/bin/bash
# STEP 1: Run this first on your AMD MI300X cloud instance
# Takes ~10-15 minutes (mostly download time)
#
# Usage:  bash 1_setup.sh
#
# This script:
#   1. Installs ROCm-compatible PyTorch + all dependencies
#   2. Downloads BindingDB drug-target binding data
#   3. Verifies GPU is visible

set -euo pipefail

echo "============================================================"
echo "  ALCHEMY — Step 1: Environment Setup (MI300X / ROCm)"
echo "============================================================"

# ── 1. ROCm environment ──────────────────────────────────────────
export HSA_OVERRIDE_GFX_VERSION=9.4.2

echo ""
echo "=== [1/4] Installing ROCm PyTorch ==="
pip install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1

echo ""
echo "=== [2/4] Installing ML dependencies ==="
pip install --quiet \
    transformers>=4.36.0 \
    accelerate \
    datasets \
    tqdm \
    matplotlib \
    numpy \
    scipy

# faiss-gpu for MI300X; fall back to CPU if wheel unavailable
pip install --quiet faiss-gpu --no-cache-dir 2>/dev/null \
    || pip install --quiet faiss-cpu

# Install full ALCHEMY requirements (some may overlap; pip deduplicates)
if [ -f requirements.txt ]; then
    pip install --quiet -r requirements.txt 2>/dev/null || true
fi

# ── 2. Verify GPU ────────────────────────────────────────────────
echo ""
echo "=== [3/4] GPU verification ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    # Quick compute test
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    print(f'GPU compute test: PASSED (matmul {y.shape})')
else:
    print('WARNING: No GPU detected! Training will be extremely slow on CPU.')
"

# ── 3. Download BindingDB ────────────────────────────────────────
echo ""
echo "=== [4/4] Downloading BindingDB dataset ==="
mkdir -p ~/alchemy_training/data
cd ~/alchemy_training/data

# BindingDB TSV — all drug-target binding measurements
# Current release is BindingDB_All_202605.tsv.zip (~2-4 GB compressed)
# URL pattern: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/bind/downloads/BindingDB_All_YYYYMM.tsv.zip
#
# We try multiple URLs in case the version rolls over:
DOWNLOADED=0
for DATE_TAG in 202605 202604 202603 202505 202504 202401; do
    URL="https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/bind/downloads/BindingDB_All_${DATE_TAG}.tsv.zip"
    OUTFILE="bindingdb_raw.tsv.zip"

    echo "  Trying BindingDB_All_${DATE_TAG}.tsv.zip ..."
    if wget -q --show-progress --timeout=30 -O "$OUTFILE" "$URL" 2>/dev/null; then
        # Check if downloaded file is actually valid (>1MB)
        FILE_SIZE=$(stat -f%z "$OUTFILE" 2>/dev/null || stat -c%s "$OUTFILE" 2>/dev/null || echo 0)
        if [ "$FILE_SIZE" -gt 1000000 ]; then
            echo "  ✅ Downloaded BindingDB_All_${DATE_TAG}.tsv.zip ($(numfmt --to=iec-i --suffix=B $FILE_SIZE 2>/dev/null || echo ${FILE_SIZE} bytes))"
            DOWNLOADED=1
            break
        else
            echo "  ⚠ File too small (${FILE_SIZE} bytes), trying next..."
            rm -f "$OUTFILE"
        fi
    else
        rm -f "$OUTFILE"
    fi
done

# Fallback: try the old .tsv.gz URL format
if [ "$DOWNLOADED" -eq 0 ]; then
    echo "  Trying legacy .tsv.gz format..."
    for DATE_TAG in 202401 202310 202307; do
        URL="https://www.bindingdb.org/bind/downloads/BindingDB_All_2D_${DATE_TAG}.tsv.gz"
        OUTFILE="bindingdb_raw.tsv.gz"
        if wget -q --show-progress --timeout=30 -O "$OUTFILE" "$URL" 2>/dev/null; then
            FILE_SIZE=$(stat -f%z "$OUTFILE" 2>/dev/null || stat -c%s "$OUTFILE" 2>/dev/null || echo 0)
            if [ "$FILE_SIZE" -gt 1000000 ]; then
                echo "  ✅ Downloaded legacy format: BindingDB_All_2D_${DATE_TAG}.tsv.gz"
                DOWNLOADED=1
                break
            fi
            rm -f "$OUTFILE"
        fi
    done
fi

if [ "$DOWNLOADED" -eq 0 ]; then
    echo ""
    echo "  ❌ ERROR: Could not download BindingDB. Manual download required:"
    echo "     1. Go to: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?all_download=yes"
    echo "     2. Download the 'All data' TSV file (BindingDB_All_YYYYMM.tsv.zip)"
    echo "     3. Place it at: ~/alchemy_training/data/bindingdb_raw.tsv.zip"
    echo ""
    exit 1
fi

# Unzip if needed (.zip format)
if [ -f "bindingdb_raw.tsv.zip" ]; then
    echo "  Extracting .zip archive..."
    unzip -o -q bindingdb_raw.tsv.zip -d .
    # Find the actual TSV file inside
    TSV_FILE=$(find . -maxdepth 1 -name "BindingDB_All*.tsv" -print -quit)
    if [ -n "$TSV_FILE" ]; then
        mv "$TSV_FILE" bindingdb_raw.tsv
        echo "  ✅ Extracted to bindingdb_raw.tsv"
    else
        echo "  ❌ Could not find TSV inside zip. Contents:"
        ls -la .
        exit 1
    fi
fi

# Decompress if .gz format
if [ -f "bindingdb_raw.tsv.gz" ] && [ ! -f "bindingdb_raw.tsv" ]; then
    echo "  Decompressing .gz archive..."
    gunzip -k bindingdb_raw.tsv.gz
    echo "  ✅ Decompressed to bindingdb_raw.tsv"
fi

echo ""
echo "============================================================"
echo "  ✅ Setup complete!"
echo ""
echo "  Next step:  python 2_prepare_data.py"
echo "============================================================"
