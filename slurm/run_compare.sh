#!/bin/bash
# Run the full compare_methods.py pipeline sequentially on one SLURM node.
# Use this if you want a single job that produces comparison.png + JSON.
# Methods run one-after-another (not parallel), so request longer walltime.
#
# Usage:
#   sbatch slurm/run_compare.sh
#   sbatch --export=CONFIG=configs/cifar100_crossdevice.yaml slurm/run_compare.sh

#SBATCH --job-name=flllm_compare
#SBATCH --output=logs/slurm_compare_%j.out
#SBATCH --error=logs/slurm_compare_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
# #SBATCH --partition=gpu

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flllm

cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/cifar10_crossdevice.yaml}
PLOT=results/comparison_crossdevice.png
RESULTS_JSON=results/comparison_crossdevice.json

mkdir -p results logs

echo "========================================"
echo "  Config : $CONFIG"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Date   : $(date)"
echo "========================================"

python experiments/compare_methods.py \
    --config  "$CONFIG" \
    --plot    "$PLOT" \
    --results "$RESULTS_JSON"

echo "Done. Plot: $PLOT  |  JSON: $RESULTS_JSON"
