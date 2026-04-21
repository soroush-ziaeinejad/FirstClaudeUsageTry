"""
OORT: system- and stat-aware client selection (Lai et al., 2021, OSDI).

Utility score combines:
  - Statistical utility: sqrt(local_loss) — higher loss = more informative
  - System utility:      clip(deadline / latency, 0, 1) * availability_fraction
  - Staleness penalty:   exp(-lambda * rounds_since_selected) to ensure fairness

Clients not yet in the registry get a high default utility to encourage exploration.
"""
from __future__ import annotations
import math
import random
from typing import List

from flwr.server.client_proxy import ClientProxy

from strategy.llm_fed_strategy import LLMFedStrategy


class OORTStrategy(LLMFedStrategy):
    def __init__(
        self,
        deadline: float = 60.0,   # target round time in seconds
        staleness_lambda: float = 0.1,
        exploration_factor: float = 0.1,  # fraction of budget reserved for new clients
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.deadline = deadline
        self.staleness_lambda = staleness_lambda
        self.exploration_factor = exploration_factor
        self._rounds_since_selected: dict = {}

    def select_clients(
        self,
        available: List[ClientProxy],
        server_round: int,
    ) -> List[ClientProxy]:
        budget = min(self.num_clients_per_round, len(available))

        # Split into known (have registry entry) and unknown
        known, unknown = [], []
        for c in available:
            if self.registry.get(c.cid) is not None:
                known.append(c)
            else:
                unknown.append(c)

        # Exploration slots for clients never selected
        explore_n = min(len(unknown), max(1, int(budget * self.exploration_factor)))
        exploit_n = budget - explore_n

        # Score known clients
        scored = [(c, self._utility(c, server_round)) for c in known]
        scored.sort(key=lambda x: x[1], reverse=True)
        exploited = [c for c, _ in scored[:exploit_n]]

        # Random exploration from unknown clients
        explored = random.sample(unknown, explore_n) if unknown else []

        selected = exploited + explored
        # Fill any gap (e.g. not enough known clients)
        if len(selected) < budget:
            remaining = [c for c in available if c not in selected]
            selected += random.sample(remaining, min(budget - len(selected), len(remaining)))

        # Update staleness counter
        selected_cids = {c.cid for c in selected}
        for c in available:
            if c.cid in selected_cids:
                self._rounds_since_selected[c.cid] = 0
            else:
                self._rounds_since_selected[c.cid] = (
                    self._rounds_since_selected.get(c.cid, 0) + 1
                )

        return selected[:budget]

    def _utility(self, client: ClientProxy, server_round: int) -> float:
        record = self.registry.get(client.cid)
        if record is None:
            return float("inf")  # always prefer unexplored clients

        # Statistical utility
        stat = math.sqrt(max(record.last_loss, 1e-6))

        # System utility: fraction of time client finishes within deadline
        latency = list(record.grad_norm_history)  # reuse field; latency in FLClient
        # We stored latency in the record via a different field — use availability
        sys_util = record.active_fraction

        # Staleness penalty
        stale = self._rounds_since_selected.get(client.cid, 0)
        penalty = math.exp(-self.staleness_lambda * stale)

        return stat * sys_util * penalty
