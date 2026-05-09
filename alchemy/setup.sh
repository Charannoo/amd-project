#!/usr/bin/env bash
# setup.sh — ALCHEMY environment (Ubuntu 22.04 + ROCm MI300X)

set -e
echo "======================================"
echo "  ALCHEMY Setup (AMD MI300X / ROCm)"
echo "======================================"

export HSA_OVERRIDE_GFX_VERSION=9.4.2
export HIP_VISIBLE_DEVICES=0

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1 -q
pip install -r requirements.txt -q
pip install vllm -q || echo "Optional: vLLM install skipped or failed — install on GPU node."

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA (ROCm) available:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('GPU:', p.name, 'Memory GB:', round(p.total_memory / 1e9, 1))
"

echo "Next: (1) vLLM Qwen server (2) python data/build_drug_index.py (3) python main.py"
