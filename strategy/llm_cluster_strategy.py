"""
LLMClusterStrategy: Phase 3 — LLM embedding + K-Means clustering client selection.
Extends LLMFedStrategy, only overrides select_clients().
"""
from __future__ import annotations
import random
from typing import List

from flwr.server.client_proxy import ClientProxy

from strategy.llm_fed_strategy import LLMFedStrategy
from strategy.embedding_module import EmbeddingModule
from strategy.clustering import cluster_and_sample


class LLMClusterStrategy(LLMFedStrategy):
    """
    Selects clients by:
      1. Fetching descriptors from the registry for known clients
      2. Embedding via MiniLM
      3. Clustering + loss-tiebreak sampling
      4. Falling back to random for clients not yet in registry
    """

    def select_clients(
        self,
        available: List[ClientProxy],
        server_round: int,
    ) -> List[ClientProxy]:
        budget = min(self.num_clients_per_round, len(available))
        cid_to_proxy = {c.cid: c for c in available}

        descriptors = self.registry.all_descriptors()
        losses = self.registry.all_losses()

        # Clients with known descriptors
        known = {cid: desc for cid, desc in descriptors.items() if cid in cid_to_proxy}
        unknown = [c for c in available if c.cid not in known]

        if len(known) < budget:
            # Not enough history yet — random fallback for warm-up rounds
            return random.sample(available, budget)

        # Embed known clients
        embedder = EmbeddingModule.get()
        embeddings = embedder.embed(known)

        selected_cids = cluster_and_sample(
            embeddings=embeddings,
            losses={cid: losses.get(cid, 0.0) for cid in known},
            budget=budget,
        )

        # Map back to proxies; fill any gap with unknown clients
        selected = [cid_to_proxy[cid] for cid in selected_cids if cid in cid_to_proxy]
        if len(selected) < budget:
            gap = budget - len(selected)
            selected += random.sample(unknown, min(gap, len(unknown)))

        return selected[:budget]
