"""
K-Means clustering + diversity sampling for LLM-based client selection.

Algorithm each round:
  1. Embed available clients → 384-dim vectors
  2. K-Means cluster (k = budget // cluster_size)
  3. Sample proportionally from each cluster
  4. Within each cluster: prefer clients with highest recent loss (PoC tiebreak)
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Tuple

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


def cluster_and_sample(
    embeddings: Dict[str, np.ndarray],
    losses: Dict[str, float],
    budget: int,
    min_clusters: int = 2,
    random_state: int = 42,
) -> List[str]:
    """
    Args:
        embeddings: {client_id: 384-dim unit vector}
        losses:     {client_id: latest local loss}
        budget:     number of clients to select
    Returns:
        List of selected client_ids
    """
    cids = list(embeddings.keys())
    n = len(cids)
    budget = min(budget, n)

    if n <= budget:
        return cids

    # Stack embedding matrix
    X = np.stack([embeddings[c] for c in cids])  # (n, 384)
    X = normalize(X, norm="l2")

    k = max(min_clusters, min(budget, n // 2))
    km = MiniBatchKMeans(n_clusters=k, random_state=random_state,
                         n_init=3, max_iter=100, batch_size=min(256, n))
    labels = km.fit_predict(X)

    # Group clients by cluster
    clusters: Dict[int, List[str]] = {}
    for idx, cid in enumerate(cids):
        cl = int(labels[idx])
        clusters.setdefault(cl, []).append(cid)

    # Proportional budget per cluster, minimum 1
    cluster_ids = sorted(clusters.keys())
    sizes = [len(clusters[cl]) for cl in cluster_ids]
    total = sum(sizes)
    alloc = _proportional_alloc(sizes, budget)

    selected: List[str] = []
    for cl, quota in zip(cluster_ids, alloc):
        candidates = clusters[cl]
        # Sort by descending loss (highest loss = most informative), fallback to 0
        candidates.sort(key=lambda c: losses.get(c, 0.0), reverse=True)
        selected.extend(candidates[:quota])

    return selected


def _proportional_alloc(sizes: List[int], budget: int) -> List[int]:
    """Allocate budget proportionally to cluster sizes, each ≥ 1."""
    total = sum(sizes)
    raw = [budget * s / total for s in sizes]
    alloc = [max(1, math.floor(r)) for r in raw]
    # Distribute remaining slots to clusters with largest fractional parts
    remaining = budget - sum(alloc)
    fracs = [(raw[i] - math.floor(raw[i]), i) for i in range(len(sizes))]
    fracs.sort(reverse=True)
    for _, i in fracs[:remaining]:
        alloc[i] += 1
    return alloc
