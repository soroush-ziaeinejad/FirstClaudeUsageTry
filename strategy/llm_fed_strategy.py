"""
LLMFedStrategy: Flower strategy with pluggable client selection.
Phase 1 — ships with plain FedAvg aggregation.
Phase 3 — will swap in the LLM+clustering selector (see embedding_module.py / clustering.py).
"""
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import (FitIns, FitRes, EvaluateIns, EvaluateRes,
                          Parameters, Scalar, ndarrays_to_parameters,
                          parameters_to_ndarrays)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box

from strategy.client_registry import ClientRegistry
from utils.logger import ExperimentLogger

console = Console()


class LLMFedStrategy(fl.server.strategy.Strategy):
    """
    Base strategy that handles FedAvg aggregation and exposes
    `select_clients()` as the single override point for Phase 3.
    """

    def __init__(
        self,
        num_clients_per_round: int = 10,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        local_epochs: int = 1,
        lr: float = 0.01,
        num_classes: int = 10,
        initial_parameters: Optional[Parameters] = None,
        logger: Optional[ExperimentLogger] = None,
    ):
        self.num_clients_per_round = num_clients_per_round
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.local_epochs = local_epochs
        self.lr = lr
        self.num_classes = num_classes
        self.initial_parameters = initial_parameters
        self.registry = ClientRegistry()
        self._current_round = 0
        self._history: List[Dict] = []  # [{round, loss, accuracy}]
        self._round_start: float = 0.0
        self._num_rounds: int = 0  # set externally for progress display
        self.logger: Optional[ExperimentLogger] = logger

    # ------------------------------------------------------------------
    # Selection hook — override in subclass for LLM-based selection
    # ------------------------------------------------------------------

    def select_clients(
        self,
        available: List[ClientProxy],
        server_round: int,
    ) -> List[ClientProxy]:
        """Phase 1: uniform random. Phase 3: LLM+cluster."""
        k = min(self.num_clients_per_round, len(available))
        return random.sample(available, k)

    # ------------------------------------------------------------------
    # Flower strategy interface
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        self._current_round = server_round
        self._round_start = time.time()
        client_manager.wait_for(self.min_available_clients)
        available = list(client_manager.all().values())

        if self.logger:
            self.logger.round_start(server_round, len(available))

        selected = self.select_clients(available, server_round)

        if self.logger:
            self.logger.selection(
                method=self.__class__.__name__,
                selected_cids=[c.cid for c in selected],
                server_round=server_round,
            )

        fit_config = {
            "server_round": server_round,
            "local_epochs": self.local_epochs,
            "lr": self.lr,
        }
        return [(client, FitIns(parameters, fit_config)) for client in selected]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Update registry with per-client metrics
        for client, fit_res in results:
            self.registry.update(
                client_id=client.cid,
                metrics=fit_res.metrics,
                current_round=server_round,
                num_classes=self.num_classes,
            )
            if self.logger:
                self.logger.client_fit(
                    cid=client.cid,
                    loss=fit_res.metrics.get("train_loss", 0),
                    grad_norm=fit_res.metrics.get("grad_norm", 0),
                    latency=fit_res.metrics.get("latency", 0),
                    num_samples=fit_res.num_examples,
                )

        # FedAvg weighted aggregation
        weights = [(parameters_to_ndarrays(r.parameters), r.num_examples)
                   for _, r in results]
        aggregated = aggregate(weights)
        avg_loss = float(np.mean([r.metrics.get("train_loss", 0) for _, r in results]))
        return ndarrays_to_parameters(aggregated), {"train_loss": avg_loss}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        k = min(self.min_evaluate_clients, len(client_manager.all()))
        if k == 0:
            return []
        clients = random.sample(list(client_manager.all().values()), k)
        return [(c, EvaluateIns(parameters, {})) for c in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        loss = weighted_loss_avg([(r.num_examples, r.loss) for _, r in results])
        accuracy = float(np.mean([r.metrics.get("accuracy", 0) for _, r in results]))
        round_time = time.time() - self._round_start

        num_trained = sum(r.num_examples for _, r in results)
        self._history.append({"round": server_round, "loss": loss, "accuracy": accuracy})

        if self.logger:
            self.logger.round_end(
                server_round=server_round, loss=loss, accuracy=accuracy,
                elapsed=round_time, num_trained=num_trained, failures=len(failures),
            )

        self._print_round_summary(server_round, loss, accuracy, round_time)
        return loss, {"accuracy": accuracy}

    def _print_round_summary(self, server_round: int, loss: float,
                             accuracy: float, elapsed: float):
        total = self._num_rounds or "?"
        bar_len = 20
        filled = int(bar_len * server_round / self._num_rounds) if self._num_rounds else 0
        bar = "█" * filled + "░" * (bar_len - filled)

        # Trend arrows
        if len(self._history) >= 2:
            prev = self._history[-2]
            loss_arrow = "↓" if loss < prev["loss"] else "↑"
            acc_arrow = "↑" if accuracy > prev["accuracy"] else "↓"
        else:
            loss_arrow = acc_arrow = " "

        console.print(
            f"  [bold cyan]Round {server_round:>3}/{total}[/]  "
            f"[{bar}]  "
            f"loss [yellow]{loss:.4f}[/] {loss_arrow}  "
            f"acc [green]{accuracy:.4f}[/] {acc_arrow}  "
            f"[dim]{elapsed:.1f}s[/]"
        )

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None
