# FLLLM — Federated Learning with LLM-Embedding-Based Client Selection

> **Research project** — PhD work targeting MLSys / FL@ICML  
> *Using LLM embeddings to encode client state and improve FL convergence speed via diversity-aware client selection.*

---

## Idea

Standard FL client selection (random, loss-biased, system-aware) treats clients as scalar metrics and misses the rich multi-dimensional state each client carries — its data distribution, training dynamics, availability patterns, and loss trends.

This project encodes all of that into a single **structured text descriptor** per client, embeds it with a lightweight LLM (`all-MiniLM-L6-v2`), clusters the embedding space each round, and samples **diverse + informative** clients across clusters. The result: faster convergence versus random and greedy baselines on heterogeneous FL benchmarks.

---

## Method

```
Each FL round:
  1. Each client reports:  local loss, gradient norm, latency,
                           label histogram (class fractions only),
                           activity pattern (no raw data ever sent)

  2. Server builds descriptor:
     "Client 7: active 80% of last 20 rounds, last_gap=1, loss=0.42 (↓ trend),
      class_dist=[0.40, 0.10, 0.30, ...], data_entropy=1.42, grad_norm=1.2"

  3. Embed descriptors → 384-dim vectors via all-MiniLM-L6-v2 (CPU, server-side)

  4. K-Means cluster → sample proportionally across clusters
     Within each cluster: prefer highest-loss clients (PoC-style tiebreak)

  5. Selected clients train → FedAvg aggregation
```

**Privacy**: no raw data ever leaves the client — only statistics. No formal DP mechanism required (aggregation-only approach).

---

## Baselines

| Method | Description |
|---|---|
| **FedAvg** | Uniform random client selection (McMahan et al., 2017) |
| **FedProx** | FedAvg + proximal term `(μ/2)‖w − w_global‖²` (Li et al., 2020) |
| **PoC** | Power-of-Choice: 2× oversample, top-K by local loss (Lai et al., 2021) |
| **OORT** | Utility = `√loss × availability × staleness_penalty` (Lai et al., 2021) |
| **LLMFed** | **Ours** — MiniLM embedding + K-Means clustering + loss tiebreak |

---

## Datasets

| Dataset | Task | Setting | Classes |
|---|---|---|---|
| CIFAR-10 | Image classification | Cross-device (200 clients) | 10 |
| CIFAR-100 | Image classification | Cross-device (200 clients) | 100 |
| FEMNIST | Handwriting recognition | Cross-device (200 clients) | 62 |
| Shakespeare | Next-char prediction | Cross-device (100 clients) | 95 |
| MedMNIST (PathMNIST) | Medical imaging | Cross-silo (20 clients) | 9 |
| ISIC 2019 | Skin lesion classification | Cross-silo (20 clients) | 8 |

All image datasets use **Dirichlet(α) partitioning** for non-IID heterogeneity. α ∈ {0.1, 0.5, 1.0}.

---

## Project Structure

```
.
├── client/
│   ├── fl_client.py          # Flower NumPyClient — training, FedProx term, MPS support
│   ├── models.py             # SmallCNN (vision) + CharLSTM (text)
│   ├── profile_builder.py    # Client state descriptor builder
│   └── device_utils.py       # MPS / CUDA / CPU detection + cache clearing
├── datasets/
│   ├── dataset_factory.py    # Single entry point — get_dataset(name, ...)
│   ├── partitioning.py       # Dirichlet non-IID partitioning
│   ├── cifar.py / femnist.py / shakespeare.py / medmnist.py / isic.py
├── strategy/
│   ├── llm_fed_strategy.py   # Base FedAvg strategy (selection hook + logging)
│   ├── llm_cluster_strategy.py  # Ours: MiniLM + K-Means selection
│   ├── embedding_module.py   # MiniLM singleton (CPU-only)
│   ├── clustering.py         # K-Means + proportional diversity sampling
│   ├── fedprox_strategy.py   # FedProx baseline
│   ├── poc_strategy.py       # Power-of-Choice baseline
│   ├── oort_strategy.py      # OORT baseline
│   └── client_registry.py    # Server-side client profile store
├── simulation/
│   └── run_simulation.py     # Flower start_simulation entry point
├── experiments/
│   └── run_experiment.py     # Multi-method experiment runner
├── utils/
│   └── logger.py             # Dated log files + rich console output
├── configs/                  # YAML configs for each dataset/scale
├── logs/                     # Auto-generated dated log files
├── test_run.py               # Quick test runner with rich visuals + plots
├── main.py                   # Top-level launcher
└── requirements.txt
```

---

## Installation

```bash
# Create environment
conda create -n flllm python=3.11 -y
conda activate flllm

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: `flwr[simulation]`, `torch`, `torchvision`, `sentence-transformers`, `scikit-learn`, `medmnist`, `rich`, `matplotlib`, `wandb`, `tqdm`, `pyyaml`

---

## Usage

### Quick smoke test (5 rounds, 10 clients)
```bash
python test_run.py --config configs/cifar10_test.yaml --method fedavg
```

### Run any method
```bash
python test_run.py --config configs/cifar10_test.yaml --method llmfed  --plot llmfed.png
python test_run.py --config configs/cifar10_test.yaml --method poc     --plot poc.png
python test_run.py --config configs/cifar10_test.yaml --method oort    --plot oort.png
python test_run.py --config configs/cifar10_test.yaml --method fedprox --plot fedprox.png
```

### Full cross-device experiment
```bash
python test_run.py --config configs/cifar10_crossdevice.yaml --method llmfed
```

### Cross-silo medical imaging
```bash
python test_run.py --config configs/medmnist_crosssilo.yaml --method llmfed
```

---

## Config Parameters

```yaml
dataset: cifar10          # cifar10 | cifar100 | femnist | shakespeare | pathmnist | isic
num_clients: 200          # total virtual clients
clients_per_round: 20     # selected each round
num_rounds: 100           # total FL rounds
alpha: 0.5                # Dirichlet non-IID degree (lower = more heterogeneous)
local_epochs: 1           # local SGD epochs per round
lr: 0.01                  # learning rate
batch_size: 32
data_dir: ./data
mu: 0.01                  # FedProx proximal coefficient (fedprox only)
ray_object_store_mb: 2048 # Ray object store memory limit
```

---

## Logging

Every run automatically creates a timestamped log file:
```
logs/2026-04-20_21-30-00_llmfed.log
```

Log entries include:
- Experiment config at start
- Per-round: loss, accuracy, trained samples, failures, wall time
- Per-client (debug): local loss, grad norm, latency
- LLM embedding time and cluster sizes (llmfed only)
- Best accuracy at end

---

## Results (smoke test — 5 rounds, CIFAR-10, 10 clients, α=0.5)

| Round | Loss | Accuracy |
|---|---|---|
| 1 | 2.0439 | 24.4% |
| 2 | 1.6550 | 36.4% |
| 3 | 1.5711 | 40.4% |
| 4 | 1.5339 | 42.4% |
| 5 | 1.3879 | **49.9%** |

*Full experiment results (100 rounds, all baselines) to be added after experiments complete.*

---

## Hardware

- Supports **MPS** (Apple Silicon), **CUDA**, and **CPU**
- Device auto-detected via `client/device_utils.py`
- MiniLM embedding always runs on CPU (MPS doesn't support all sentence-transformer ops)
- Ray object store memory configurable via `ray_object_store_mb` in config

---

## Citation

*To be added upon publication.*
