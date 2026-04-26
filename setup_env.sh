#!/usr/bin/env bash
# Setup script for the FL project.
# Creates a virtualenv that inherits the system torch/torchvision/numpy from
# /opt/pytorch, then installs the remaining project dependencies.
# Matplotlib is installed last as a separate step.
#
# Usage:
#   chmod +x setup_env.sh && ./setup_env.sh
#   source .venv/bin/activate

set -euo pipefail

PYTHON=/opt/pytorch/bin/python3
VENV_DIR="$(cd "$(dirname "$0")" && pwd)/.venv"

echo "========================================"
echo " FL Project Environment Setup"
echo "========================================"
echo "Python  : $PYTHON ($($PYTHON --version))"
echo "Venv    : $VENV_DIR"
echo ""

# ── 1. Create virtual environment ────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "[1/4] Virtual environment already exists — skipping creation."
else
    echo "[1/4] Creating virtual environment with --system-site-packages..."
    $PYTHON -m venv --system-site-packages "$VENV_DIR"
    echo "      Done."
fi

source "$VENV_DIR/bin/activate"

# ── 2. Upgrade pip ───────────────────────────────────────────────────────────
echo ""
echo "[2/4] Upgrading pip..."
pip install --quiet --upgrade pip
echo "      pip $(pip --version | awk '{print $2}')"

# ── 3. Install project dependencies ─────────────────────────────────────────
echo ""
echo "[3/4] Installing project dependencies..."
echo "      (torch/torchvision/numpy inherited from system — skipped)"
echo ""

pip install \
    "flwr[simulation]>=1.8.0" \
    "sentence-transformers>=2.7.0" \
    "scikit-learn>=1.4.0" \
    "pyyaml>=6.0" \
    "medmnist>=3.0.0" \
    "tqdm>=4.66.0" \
    "rich>=13.0.0" \
    "wandb>=0.17.0"

echo ""
echo "      Core dependencies installed."

# ── 4. Matplotlib (separate step) ────────────────────────────────────────────
echo ""
echo "[4/4] Installing matplotlib..."
pip install "matplotlib>=3.8.0"
echo "      matplotlib $(python -c 'import matplotlib; print(matplotlib.__version__)')"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Setup complete."
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run an experiment:"
echo "  python main.py --config configs/cifar10_crossdevice.yaml --method fedavg"
echo ""
