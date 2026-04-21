"""
Quick launcher — delegates to experiments/run_experiment.py.

Examples:
    python main.py --config configs/cifar10_crossdevice.yaml --method fedavg
    python main.py --config configs/medmnist_crosssilo.yaml  --method llmfed
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run_experiment import main

if __name__ == "__main__":
    main()