"""
Power-of-Choice (PoC): biased client selection toward high-loss clients.

Algorithm (Lai et al., 2021):
  1. Sample d * K clients uniformly at random (d = oversampling factor, default 2)
  2. From those, pick top-K by highest local loss reported in the last round
  3. For clients with no loss history, assign mean loss (exploration)
"""
from __future__ import annotations
import random
from typing import List

from flwr.server.client_proxy import ClientProxy

from strategy.llm_fed_strategy import LLMFedStrategy


class PoCStrategy(LLMFedStrategy):
    def __init__(self, oversample_factor: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.d = oversample_factor

    def select_clients(
        self,
        available: List[ClientProxy],
        server_round: int,
    ) -> List[ClientProxy]:
        budget = min(self.num_clients_per_round, len(available))
        candidate_k = min(len(available), max(budget, int(budget * self.d)))

        # Step 1: random oversample
        candidates = random.sample(available, candidate_k)

        # Step 2: rank by loss
        losses = self.registry.all_losses()
        known_losses = [losses[c.cid] for c in candidates if c.cid in losses]
        mean_loss = sum(known_losses) / len(known_losses) if known_losses else 1.0

        candidates.sort(
            key=lambda c: losses.get(c.cid, mean_loss),
            reverse=True,  # highest loss first
        )
        return candidates[:budget]
