#!/bin/bash
# One-time environment setup on the university server.
# Run this once after cloning the repo:
#   bash slurm/setup_env.sh

set -e

echo "=== Creating conda environment 'flllm' ==="
conda create -n flllm python=3.11 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flllm

echo "=== Installing requirements ==="
pip install -r requirements.txt

echo "=== Verifying GPU ==="
python - <<'EOF'
import torch
print(f"PyTorch   : {torch.__version__}")
print(f"CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU       : {torch.cuda.get_device_name(0)}")
EOF

echo "=== Pre-downloading MiniLM model (needed by LLMFed) ==="
python - <<'EOF'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2")
print("MiniLM loaded OK")
EOF

echo ""
echo "Setup complete. Activate with:  conda activate flllm"
