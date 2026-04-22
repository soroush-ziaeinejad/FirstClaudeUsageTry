#!/bin/bash
# Submit all 5 FL methods as independent SLURM jobs (run in parallel).
# Usage:
#   bash slurm/run_all.sh
#   bash slurm/run_all.sh configs/cifar100_crossdevice.yaml

set -e

CONFIG=${1:-configs/cifar10_crossdevice.yaml}
METHODS=(fedavg fedprox poc oort llmfed)

cd "$(dirname "$0")/.."
mkdir -p results logs

echo "Submitting ${#METHODS[@]} jobs with config: $CONFIG"
echo ""

for METHOD in "${METHODS[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --export=METHOD=$METHOD,CONFIG=$CONFIG \
        slurm/run_method.sh)
    echo "  Submitted $METHOD  →  job $JOB_ID"
done

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       logs/slurm_flllm_<method>_<jobid>.out"
