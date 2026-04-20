# Research Plan: LLM-Embedding-Based Client Selection for Federated Learning

## Core Thesis
Current FL client selection methods (random, loss-biased, system-aware) treat clients as scalar metrics. We propose encoding rich multi-dimensional client state into LLM embeddings and using clustering-based diversity sampling to accelerate global model convergence — no explicit DP required, no raw data shared.

---

## 1. Client Information Stack

Each client maintains and reports a **structured profile** (no raw data):

| Feature Group | What's Encoded | Privacy |
|---|---|---|
| **Availability pattern** | Binary active/inactive history per round → summary stats (fraction active, max consecutive gap, recency) | Harmless metadata |
| **Local loss history** | Per-round local loss → current value, trend slope, variance over last K rounds | No data content |
| **Data distribution** | Normalized label-count histogram (class fractions only) — never raw samples | Aggregated server-side; no individual histogram stored long-term |
| **Training dynamics** | Gradient norm magnitude, local epochs completed, round latency | System-level only |

These are serialized into a **structured text descriptor** per client, e.g.:
```
"Client active 73% of last 20 rounds, last gap=3, loss=0.42 (↓ trend),
 classes=[0.4, 0.1, 0.3, 0.1, 0.1], grad_norm=1.2"
```

---

## 2. LLM Embedding Module

- **Model**: `all-MiniLM-L6-v2` (sentence-transformers) — 22M params, fast, runs server-side
  - TinyBERT is also viable but MiniLM gives better semantic embeddings for structured text
- **Where it runs**: FL server only — clients never touch the LLM
- **Output**: 384-dim embedding per client per round
- **Update frequency**: Re-embed all available clients at the start of each round

---

## 3. Selection Algorithm (Cluster-then-Sample)

```
Each round:
1. Collect profiles from all available clients
2. Embed via MiniLM → 384-dim vectors
3. K-Means cluster (k = num_selected / cluster_size, e.g. k=10 for 100 clients selecting 20)
4. Sample ceil(budget/k) clients per cluster
5. Within each cluster: prefer clients with highest recent local loss (PoC-style tiebreak)
6. Return selected set to Flower configure_fit()
```

This guarantees **diversity** (via clusters) + **informativeness** (via loss tiebreak) in a single unified policy.

---

## 4. Flower Implementation Architecture

```
project/
├── strategy/
│   ├── llm_fed_strategy.py      # Custom Strategy (extends FedAvg)
│   ├── client_registry.py       # Server-side client profile store
│   ├── embedding_module.py      # MiniLM wrapper
│   └── clustering.py            # KMeans + sampling logic
├── client/
│   ├── fl_client.py             # Flower NumPyClient base
│   └── profile_builder.py       # Builds client descriptor each round
├── datasets/
│   ├── dataset_factory.py       # Config-driven loader (one entry point)
│   ├── cifar.py
│   ├── femnist.py
│   ├── shakespeare.py
│   ├── isic.py
│   └── medmnist.py
├── simulation/
│   └── run_simulation.py        # Flower simulation entry point
├── configs/
│   ├── cifar10_crossdevice.yaml
│   ├── femnist_crossdevice.yaml
│   ├── shakespeare.yaml
│   ├── medmnist_crosssilo.yaml
│   └── isic_crosssilo.yaml
└── experiments/
    └── run_experiment.py        # Sweeps baselines + proposed method
```

**Key design**: `dataset_factory.py` is the only place you touch for new datasets — all strategy/client code is dataset-agnostic.

---

## 5. Experiment Design

**Variables:**

| Axis | Values |
|---|---|
| Non-IID degree | Dirichlet α ∈ {0.1, 0.5, 1.0} |
| Scale | Cross-device (200 clients, 20/round) + Cross-silo (20 clients, 8/round) |
| Datasets | CIFAR-10, FEMNIST, Shakespeare, MedMNIST, ISIC |
| Methods | FedAvg, FedProx, PoC, OORT, **Ours** |

**Metrics:**
- Rounds to reach target accuracy (primary — convergence speed claim)
- Final accuracy at fixed round budget
- Client participation fairness (variance across clients)
- Wall-clock time per round (for MLSys venue)

**Ablations** (important for rebuttal-readiness):
- Remove availability features
- Remove loss history
- Remove distribution summary
- Random sampling within clusters (vs. loss-tiebreak)
- Embedding dim sensitivity

---

## 6. Writeup Structure (MLSys/FL@ICML)

1. **Introduction** — FL heterogeneity problem, gap in existing selection, LLM embeddings as unified profiler
2. **Related Work** — Client selection (PoC, OORT, Favor), FL heterogeneity, LLMs for system optimization
3. **Method** — Client profile stack → embedding → clustering → selection algorithm
4. **Privacy Analysis** — Informal argument: no raw data shared, only aggregated summaries
5. **Experiments** — Convergence curves, rounds-to-target table, ablations, scale results
6. **Discussion** — Overhead analysis (embedding cost), limitations, future work

---

## 7. Implementation Milestones

| Phase | Tasks | Est. Time |
|---|---|---|
| **Phase 1** | Flower base + dataset factory + FedAvg baseline | 1 week |
| **Phase 2** | Client profile builder + server-side registry | 3 days |
| **Phase 3** | MiniLM embedding + clustering strategy | 3 days |
| **Phase 4** | All baselines (FedProx, PoC, OORT) | 1 week |
| **Phase 5** | Experiment sweeps + result logging (W&B) | 1–2 weeks |
| **Phase 6** | Ablations + writeup | 2 weeks |

---

## Baselines
- FedAvg (random selection)
- FedProx
- Power-of-Choice (PoC)
- OORT

## Target Venue
MLSys / FL@ICML
