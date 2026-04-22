#!/bin/bash
# Run a single FL method on a SLURM GPU node.
# Usage:
#   sbatch --export=METHOD=fedavg slurm/run_method.sh
#   sbatch --export=METHOD=llmfed,CONFIG=configs/cifar10_crossdevice.yaml slurm/run_method.sh
#
# Or use run_all.sh to submit all 5 methods at once.

#SBATCH --job-name=flllm_${METHOD:-unknown}
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
# Uncomment and set your GPU partition name if needed:
# #SBATCH --partition=gpu

set -e

# ── Environment ────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flllm

cd "$(dirname "$0")/.."   # repo root

# ── Arguments ─────────────────────────────────────────────────────────────────
METHOD=${METHOD:-fedavg}
CONFIG=${CONFIG:-configs/cifar10_crossdevice.yaml}
PLOT=results/${METHOD}_crossdevice.png

mkdir -p results logs

echo "========================================"
echo "  Method : $METHOD"
echo "  Config : $CONFIG"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Date   : $(date)"
echo "========================================"

python test_run.py \
    --config "$CONFIG" \
    --method "$METHOD" \
    --plot   "$PLOT"

echo "Done: $METHOD  →  $PLOT"
