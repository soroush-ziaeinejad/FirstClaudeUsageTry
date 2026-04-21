"""
FedProx: FedAvg + proximal term (mu) passed to clients via fit config.
The proximal term (mu/2)||w - w_global||^2 is applied client-side in fl_client.py.
"""
from __future__ import annotations
from typing import List, Tuple

from flwr.common import FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from strategy.llm_fed_strategy import LLMFedStrategy


class FedProxStrategy(LLMFedStrategy):
    def __init__(self, mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        pairs = super().configure_fit(server_round, parameters, client_manager)
        # Inject mu into each client's fit config
        return [
            (client, FitIns(fit_ins.parameters, {**fit_ins.config, "mu": self.mu}))
            for client, fit_ins in pairs
        ]
